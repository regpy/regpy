from __future__ import annotations
import math
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
from typing import Optional
from scipy.special import lpmv, factorial

from .numpy import MeasureSpaceFcts,GridFcts
from regpy.operators import Operator
from regpy.util import Errors


def _angles_and_frame(directions: np.ndarray):
    """Return spherical angles and the orthonormal frame for unit directions."""
    directions = np.asarray(directions, dtype=float)
    if directions.ndim < 1 or directions.shape[-1] != 3:
        raise ValueError(
            f"directions must have shape (..., 3). Got {directions.shape}."
        )
    norms = np.linalg.norm(directions, axis=-1)
    if np.any(norms == 0):
        raise ValueError("directions must not contain the zero vector.")
    if not np.allclose(norms, 1.0, atol=1e-12, rtol=1e-12):
        raise ValueError("directions must consist of normalized vectors.")

    z = np.clip(directions[..., 2], -1.0, 1.0)
    theta = np.arccos(z)
    phi = np.mod(np.arctan2(directions[..., 1], directions[..., 0]), 2*np.pi)
    st, ct = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi), np.cos(phi)
    erho = directions
    etheta = np.stack((ct*cp, ct*sp, -st), axis=-1)
    ephi = np.stack((-sp, cp, np.zeros_like(sp)), axis=-1)
    return theta, phi, st, erho, etheta, ephi


def _complex_spherical_harmonics(
        degree: int, directions: np.ndarray, derivatives: bool = False):
    """Complex spherical harmonics in ``l*l + l + m`` ordering.

    If ``derivatives`` is true, the returned derivatives are with respect to
    theta and the unit ``Ephi`` direction, i.e.
    ``(1/sin(theta))*d/dphi``.
    """
    from scipy.special import sph_harm_y

    theta, phi, st, _, etheta, ephi = _angles_and_frame(directions)
    shape = theta.shape + ((degree+1)**2,)
    values = np.empty(shape, dtype=complex)
    if derivatives:
        dtheta = np.empty(shape, dtype=complex)
        dphi_tilde = np.empty(shape, dtype=complex)

    for l in range(degree+1):
        for m in range(-l, l+1):
            idx = l*l + l + m
            if derivatives:
                y, yp = sph_harm_y(l, m, theta, phi, diff_n=1)
                values[..., idx] = y
                dtheta[..., idx] = yp[..., 0]
                np.divide(
                    yp[..., 1],
                    st,
                    out=dphi_tilde[..., idx],
                    where=st >= 1e-12,
                )
            else:
                values[..., idx] = sph_harm_y(l, m, theta, phi)

    if derivatives:
        pole_mask = st < 1e-12
        if np.any(pole_mask):
            directions = np.asarray(directions, dtype=float)
            eps = 1e-7
            c, s = np.cos(eps), np.sin(eps)
            for tangent, target in ((etheta, dtheta), (ephi, dphi_tilde)):
                plus = c*directions[pole_mask] + s*tangent[pole_mask]
                minus = c*directions[pole_mask] - s*tangent[pole_mask]
                target[pole_mask] = (
                    _complex_spherical_harmonics(degree, plus)
                    - _complex_spherical_harmonics(degree, minus)
                )/(2*eps)
        return values, dtheta, dphi_tilde
    return values


def _real_spherical_harmonics(
        degree: int, directions: np.ndarray, derivatives: bool = False):
    """Real basis used by :class:`StarSHSurfaceSpc`."""
    if derivatives:
        yc, ytheta, yphi = _complex_spherical_harmonics(
            degree, directions, derivatives=True
        )
        basis = np.empty_like(yc.real)
        btheta = np.empty_like(basis)
        bphi = np.empty_like(basis)
    else:
        yc = _complex_spherical_harmonics(degree, directions)
        basis = np.empty_like(yc.real)

    fac = np.sqrt(2.0)
    for l in range(degree+1):
        i0 = l*l + l
        basis[..., i0] = yc[..., i0].real
        if derivatives:
            btheta[..., i0] = ytheta[..., i0].real
            bphi[..., i0] = yphi[..., i0].real
        for m in range(1, l+1):
            ip = l*l + l + m
            im = l*l + l - m
            basis[..., ip] = fac*yc[..., ip].real
            basis[..., im] = fac*yc[..., ip].imag
            if derivatives:
                btheta[..., ip] = fac*ytheta[..., ip].real
                btheta[..., im] = fac*ytheta[..., ip].imag
                bphi[..., ip] = fac*yphi[..., ip].real
                bphi[..., im] = fac*yphi[..., ip].imag
    if derivatives:
        return basis, btheta, bphi
    return basis


def _divide_by_sin_with_pole_limit(
        value: np.ndarray,
        dvalue_dtheta: np.ndarray,
        theta: np.ndarray,
        sin_theta: np.ndarray) -> np.ndarray:
    """Return ``value/sin(theta)`` with the differentiable pole limits.

    This is useful for axisymmetric parameterizations written as
    ``(x(theta), h(theta) sin(phi), h(theta) cos(phi))``.  If ``h`` vanishes
    at a pole, then h/sin(theta) tends to h'(0) at the north pole and to
    -h'(pi) at the south pole.
    """
    value = np.asarray(value, dtype=float)
    dvalue_dtheta = np.asarray(dvalue_dtheta, dtype=float)
    theta = np.asarray(theta, dtype=float)
    sin_theta = np.asarray(sin_theta, dtype=float)

    result = np.empty_like(value)
    regular = np.abs(sin_theta) > 1e-12
    np.divide(value, sin_theta, out=result, where=regular)

    north = ~regular & (theta <= np.pi/2)
    south = ~regular & ~north
    result[north] = dvalue_dtheta[north]
    result[south] = -dvalue_dtheta[south]
    return result


class S2Surface(ABC):
    """ Surfaces parameterized by the unit sphere :math:`S^2`. 
    More precisely, the surface is given by :math:`q(S^2)` for some smooth and smoothly invertible function 
    :math:`q:S^2\to R^3`. 
    This function is prepresented by its values on a Gauss-Legendre grid of :math:`(n+1) (2n+2)` 
    quadrature points in :math:`(theta,phi)` coordinates. 

    Parameters:
    -----------
    n: int

    Properties:
    --------
    n: int
        Determines the number of quadrature points, :math:`2(n+1)**2`
    xhat: np.ndarray
        quadrature points, shape (n+1,2n+2,3)
    theta: np.ndarray
        theta coordinates of quadrature points, shape (n+1,)
    phi: np.ndarray
        phi coordinates of quadrature points, shape (2*n+2,)
    Theta,Tphi: np.nd_array
        broadcast arrays of phi and theta to shape (n+1,2n+2)
    st,ct: no.ndarray
        sin(theta) and cos(theta), rsp
    sp,cp: no.ndarray
        sin(phi) and cos(phi), rsp 
    Etheta,Ephi,Erho: np.ndarray
        standard orthonormal basis vectors in spherical coordinates on the grid of quadrature points. 
        shape (n+1,2n+2,3)
    q: np.ndarray
        values o the function :math:`q` defining the surface on the quadrature points.
        shape (n+1,2n+2,3)
    Ttheta,Tphi: np.ndarray
        derivatives of q in the orthonormal ``Etheta`` and ``Ephi`` directions.
        Thus ``Tphi=(1/sin(theta))*d q/dphi``. Shape (n+1,2n+2,3).
    normal: np.ndarray
        unit normal vector on the surface.  shape (n+1,2n+2,3)
    Jacobian: nd_array
        determinant of the Jacobian.  shape (n+1,2n+2)


    """
    def __init__(self, n: int):
        # During construction the grid must be initialized before the
        # subclass-specific geometry can be evaluated.  Subsequent assignments
        # to ``n`` go through the same setter and additionally refresh all
        # geometry derived from the grid.
        self._surface_initialized = False
        self.n = n
        self._evaluate()
        self._complete_evaluation()
        self._surface_initialized = True

    class sphere_quad_data:
        """ compute data for Gauss-Legendre quadrature on grid of :math:`(n+1) (2n+2)` 
        quadrature points in :math:`(theta,phi)` coordinates. 
        For parametric surfaces these data are set up only once in the parameter vector space.
        """
        def __init__(self, n:int):
            sh = (n+1,2*n+2)
            # Gauss–Legendre in μ = cos(θ)
            mu, self._alpha = np.polynomial.legendre.leggauss(n+1)
            self._theta = np.arccos(mu[::-1])               # θ ∈ (0, π)
            self._phi = np.linspace(0, 2*np.pi, 2*n+2, endpoint=False)        
            self._st = np.sin(self._theta)
            self._ct = np.cos(self._theta)
            self._sp = np.sin(self._phi)
            self._cp = np.cos(self._phi)
            self._Erho   = np.stack((np.outer(self._st,self._cp), np.outer(self._st,self._sp), 
                                     np.broadcast_to(self._ct[:,None],sh)),axis=2)
            self._Etheta = np.stack((np.outer(self._ct,self._cp), np.outer(self._ct,self._sp),
                                     np.broadcast_to(-self._st[:,None],sh)),axis=2)
            self._Ephi   = np.stack((np.broadcast_to(-self._sp[None,:],sh), 
                                     np.broadcast_to(self._cp[None,:],sh), 
                                     np.zeros(sh)),axis=2)

    @property
    def n(self)->int:
        return self._n
    @n.setter
    def n(self,n_new:int):
        if not isinstance(n_new, (int, np.integer)) or n_new < 0:
            raise ValueError("n must be a non-negative integer.")

        self._n = int(n_new)
        dat = S2Surface.sphere_quad_data(self._n)
        self.__dict__.update(dat.__dict__)

        # The fallback arbitrary-direction evaluator caches spherical-harmonic
        # coefficients of q on the current grid.  They are invalid as soon as
        # the grid changes.
        self.__dict__.pop("_q_sh_coeff", None)

        # On construction, subclasses have not necessarily initialized their
        # parameters yet, so S2Surface.__init__ performs the first evaluation.
        # Later changes of n must refresh every grid-dependent geometry field.
        if getattr(self, "_surface_initialized", False):
            self._evaluate()
            self._complete_evaluation()

    @property
    def grid_shape(self)->tuple[int]:
        return (self.n+1,2*self.n+2)
    @property
    def theta(self)->np.ndarray:
        return self._theta
    @property
    def phi(self)->np.ndarray:
        return self._phi
    @property
    def xhat(self)->np.ndarray:
        return self._Erho
    @property
    def theta_coo(self)->np.ndarray:
        return np.broadcast_to(self._theta[:,None],self.grid_shape)
    @property
    def phi_coo(self)->np.ndarray:
        return np.broadcast_to(self._phi[None,:],self.grid_shape)        
    @property
    def quadweights(self)->np.ndarray:
        return np.broadcast_to((np.pi/(self.n+1))*self._alpha[:,None],self.grid_shape)
    @property
    def st(self)->np.ndarray:
        return self._st
    @property
    def ct(self)->np.ndarray:
        return self._ct
    @property
    def sp(self)->np.ndarray:
        return self._sp
    @property
    def cp(self)->np.ndarray:
        return self._cp
    @property
    def Erho(self)->np.ndarray:
        return self._Erho
    @property
    def Ephi(self)->np.ndarray:
        return self._Ephi
    @property
    def Etheta(self)->np.ndarray:
        return self._Etheta            
    @property
    def q(self)->np.ndarray:
        return self._q   
    @property
    def Jacobian(self)->np.ndarray:
        return self._Jacobian
    @property
    def Ttheta(self)->np.ndarray:
        return self._Ttheta           
    @property
    def Tphi(self)->np.ndarray:
        return self._Tphi
    @property
    def normal(self)->np.ndarray:
        return self._normal 

    @abstractmethod
    def _evaluate(self)->None:
        """Initialize ``q``, ``Ttheta`` and ``Tphi`` on the standard grid."""

    def evaluate_directions(self, directions: np.ndarray, full: bool = False):
        r"""Evaluate the surface geometry at arbitrary points of :math:`S^2`.

        ``Ttheta`` is the derivative in the unit ``Etheta`` direction and
        ``Tphi`` is the derivative in the unit ``Ephi`` direction, i.e.
        ``(1/sin(theta))*d/dphi``.  Consequently ``Jacobian`` is the surface
        Jacobian relative to the area element on the unit sphere.  

        The base implementation uses the degree-``n`` spherical-harmonic
        interpolant of the sampled parametrization ``q``.  Concrete surfaces
        may override this method when an exact evaluation is available.
        With ``full=False`` only ``(q, Jacobian, normal)`` is returned.
        ``full=True`` additionally returns ``Ttheta``, ``Tphi`` and the
        spherical frame ``Erho``, ``Etheta``, ``Ephi``.
        """
        directions = np.asarray(directions, dtype=float)
        if not hasattr(self, "_q_sh_coeff"):
            basis = _complex_spherical_harmonics(self.n, self.xhat)
            self._q_sh_coeff = np.einsum(
                "...j,...,...k->jk",
                np.conj(basis),
                self.quadweights,
                self.q,
            )

        basis, btheta, bphi = _complex_spherical_harmonics(
            self.n, directions, derivatives=True
        )
        q = np.einsum("...j,jk->...k", basis, self._q_sh_coeff).real
        Ttheta = np.einsum(
            "...j,jk->...k", btheta, self._q_sh_coeff
        ).real
        Tphi = np.einsum(
            "...j,jk->...k", bphi, self._q_sh_coeff
        ).real
        _, _, _, Erho, Etheta, Ephi = _angles_and_frame(directions)
        return self._finish_direction_evaluation(
            q, Ttheta, Tphi, Erho, Etheta, Ephi, full=full
        )

    @staticmethod
    def _finish_direction_evaluation(
            q, Ttheta, Tphi, Erho, Etheta, Ephi, full=False):
        normal = np.cross(Ttheta, Tphi)
        Jacobian = np.linalg.norm(normal, axis=-1)
        if np.any(Jacobian <= 0):
            raise ValueError("surface parameterization has a vanishing Jacobian.")
        normal = normal / Jacobian[..., None]
        if full:
            return q, Jacobian, normal, Ttheta, Tphi, Erho, Etheta, Ephi
        return q, Jacobian, normal

    def _complete_evaluation(self)->None:
        """Compute Jacobian and unit normal from the tangent vectors."""
        self._normal = np.cross(self.Ttheta, self.Tphi)
        self._Jacobian = np.linalg.norm(self._normal, axis=2)
        if np.any(self._Jacobian <= 0):
            raise ValueError("surface parameterization has a vanishing Jacobian.")
        self._normal /= self._Jacobian[:,:,None]

    def grid_space(self)->GridFcts:
        """ returns the GridFcts space of the quadrature grid.
        """
        return MeasureSpaceFcts(self.quadweights)

    def area(self)->float:
        """returns area of the surface
        """
        return np.sum(self.Jacobian * self.quadweights)
    
    def center_of_gravity(self)->np.ndarray:
        """returns center of gravity of the surface, a np.ndarray of shape (3,)
        """
        return np.sum(self.q * self.Jacobian[:,:,None] * self.quadweights[:,:,None], axis=(0,1)) / self.area()
    

class StarSurface(S2Surface, ABC):
    """ star-shape S2Surface with parameterization given by
    :math:`q(xhat) = r(xhat)xhat`
    """
    def __init__(self, n:int):
        super().__init__(n)

    @abstractmethod
    def radius_and_derivatives(self)->tuple[np.ndarray,np.ndarray,np.ndarray]:
        """Return r, dr/dθ, dr/dφ"""

    def _evaluate(self)->None:
        r, dr_dtheta, dr_dphi = self.radius_and_derivatives()
        self._q = r[:,:,None] * self.Erho 
        self._Ttheta = dr_dtheta[:,:,None] * self.Erho + r[:,:,None] * self.Etheta
        self._Tphi   = dr_dphi[:,:,None]   * self.Erho + r[:,:,None] * self.Ephi

class Sphere(StarSurface):
    def __init__(self, n:int, R=0.5):
        self.R = R
        super().__init__(n)

    def radius_and_derivatives(self)->tuple[np.ndarray,np.ndarray,np.ndarray]:
        r = self.R * np.ones(self.grid_shape)
        return r, np.zeros(self.grid_shape), np.zeros(self.grid_shape)

    def evaluate_directions(self, directions: np.ndarray, full: bool = False):
        _, _, _, Erho, Etheta, Ephi = _angles_and_frame(directions)
        q = self.R * Erho
        Ttheta = self.R * Etheta
        Tphi = self.R * Ephi
        return self._finish_direction_evaluation(
            q, Ttheta, Tphi, Erho, Etheta, Ephi, full=full
        )


class Ellipsoid(S2Surface):
    def __init__(self,n:int,a: float,b: float,c: float):
        self.axes = np.asarray([a, b, c], dtype=float)
        if np.any(self.axes <= 0):
            raise ValueError("ellipsoid semi-axes must be positive.")
        super().__init__(n)

    def _evaluate(self):
        # Keep the historical ``scaling`` attribute, but rebuild its broadcast
        # view for the current quadrature order.
        self.scaling = np.broadcast_to(
            self.axes[None,None,:], self.grid_shape + (3,)
        )
        self._q = self.scaling * self.Erho
        self._Ttheta = self.scaling * self.Etheta
        self._Tphi = self.scaling * self.Ephi

    def evaluate_directions(self, directions: np.ndarray, full: bool = False):
        _, _, _, Erho, Etheta, Ephi = _angles_and_frame(directions)
        q = self.axes * Erho
        Ttheta = self.axes * Etheta
        Tphi = self.axes * Ephi
        return self._finish_direction_evaluation(
            q, Ttheta, Tphi, Erho, Etheta, Ephi, full=full
        )


class Peanut(S2Surface):
    r"""The ``peanut`` surface from the original MATLAB 3D toolbox.

    The old MATLAB surface is not a radial star surface in Euclidean
    coordinates: its second Cartesian coordinate is stretched by a factor
    two.  The previous Python version omitted this factor and therefore
    represented a different obstacle.
    """

    def __init__(self, n:int, alpha=0.25):
        if alpha <= 0:
            raise ValueError("alpha must be positive.")
        self.alpha = float(alpha)
        super().__init__(n)

    def _evaluate(self):
        (
            self._q, _, _, self._Ttheta, self._Tphi,
            _, _, _
        ) = self.evaluate_directions(self.xhat, full=True)

    def evaluate_directions(self, directions: np.ndarray, full: bool = False):
        theta, _, st, Erho, Etheta, Ephi = _angles_and_frame(directions)
        ct = np.cos(theta)
        s2 = 2*st*ct
        c2 = ct*ct - st*st

        A = self.alpha + 1.0
        sqrtB = np.sqrt(A - s2*s2)
        sqrtC = np.sqrt(c2 + sqrtB)
        denom = 2*np.sqrt(1+np.sqrt(A))
        r = sqrtC / denom

        ds2 = 2*c2
        dc2 = -2*s2
        dB = -2*s2*ds2
        dC = dc2 + dB/(2*sqrtB)
        dr_dtheta = dC/(2*sqrtC*denom)

        # This anisotropic factor is present in the MATLAB ``peanut``.
        scale = np.array([1.0, 2.0, 1.0])
        scaled_Erho = Erho * scale
        q = r[..., None] * scaled_Erho
        Ttheta = (
            dr_dtheta[..., None] * scaled_Erho
            + r[..., None] * (Etheta * scale)
        )
        Tphi = r[..., None] * (Ephi * scale)

        return self._finish_direction_evaluation(
            q, Ttheta, Tphi, Erho, Etheta, Ephi, full=full
        )


class Bean(S2Surface):
    r"""The ``bean`` surface from the original MATLAB ``S2_surface.m``.

    Parameters
    ----------
    n : int
        Quadrature order.
    R : float, optional
        Overall scaling.  MATLAB default: ``0.5``.
    """

    def __init__(self, n: int, R: float = 0.5):
        if R <= 0:
            raise ValueError("R must be positive.")
        self.R = float(R)
        super().__init__(n)

    def _evaluate(self):
        (
            self._q, _, _, self._Ttheta, self._Tphi,
            _, _, _
        ) = self.evaluate_directions(self.xhat, full=True)

    def evaluate_directions(self, directions: np.ndarray, full: bool = False):
        theta, phi, st, Erho, Etheta, Ephi = _angles_and_frame(directions)
        ct = np.cos(theta)
        sp, cp = np.sin(phi), np.cos(phi)

        cos_pct = np.cos(np.pi * ct)
        sin_pct = np.sin(np.pi * ct)

        a = np.sqrt(0.64 * (1.0 - 0.1 * cos_pct))
        b = np.sqrt(0.64 * (1.0 - 0.4 * cos_pct))

        q1 = a * st * cp
        q2 = b * st * sp - 0.3 * cos_pct
        q3 = ct
        q = self.R * np.stack((q1, q2, q3), axis=-1)

        da_dtheta = (
            0.5 / a * 0.64 * 0.1 * sin_pct * np.pi * (-st)
        )
        db_dtheta = (
            0.5 / b * 0.64 * 0.4 * sin_pct * np.pi * (-st)
        )

        dq1_dtheta = a * ct * cp + da_dtheta * st * cp
        dq2_dtheta = (
            b * ct * sp
            + db_dtheta * st * sp
            + 0.3 * sin_pct * np.pi * (-st)
        )
        dq3_dtheta = -st
        Ttheta = self.R * np.stack(
            (dq1_dtheta, dq2_dtheta, dq3_dtheta), axis=-1
        )

        # Derivative in the unit Ephi direction, i.e. (1/sin(theta))*d/dphi.
        Tphi = self.R * np.stack(
            (-a * sp, b * cp, np.zeros_like(st)), axis=-1
        )

        return self._finish_direction_evaluation(
            q, Ttheta, Tphi, Erho, Etheta, Ephi, full=full
        )


class Ogive(S2Surface):
    r"""The ``ogive`` surface from the original MATLAB ``S2_surface.m``.

    This surface has the conical singularities of the MATLAB model.
    """

    def __init__(
            self, n: int,
            angle: float = np.pi * 22.62 / 180.0,
            length: float = 1.0):
        if not 0 < angle < np.pi/2:
            raise ValueError("angle must lie in (0, pi/2).")
        if length <= 0:
            raise ValueError("length must be positive.")
        self.angle = float(angle)
        self.length = float(length)
        super().__init__(n)

    def _evaluate(self):
        (
            self._q, _, _, self._Ttheta, self._Tphi,
            _, _, _
        ) = self.evaluate_directions(self.xhat, full=True)

    def evaluate_directions(self, directions: np.ndarray, full: bool = False):
        theta, phi, st, Erho, Etheta, Ephi = _angles_and_frame(directions)
        sp, cp = np.sin(phi), np.cos(phi)
        length = self.length
        d0 = 1.0 - np.cos(self.angle)

        q1 = length * (-0.5 + theta / np.pi)
        c = 2.0 * np.sin(self.angle) / length
        radicand = np.maximum(0.0, 1.0 - (c * q1)**2)
        root = np.sqrt(radicand)
        f = (root - 1.0 + d0) * length / (10.0 * d0)

        q = np.stack((q1, f * sp, f * cp), axis=-1)

        dq1_dtheta = length / np.pi
        df_dtheta = (
            (-0.1 * length / d0)
            * (c**2 * length / np.pi)
            * q1 / root
        )
        Ttheta = np.stack(
            (
                np.full_like(theta, dq1_dtheta),
                df_dtheta * sp,
                df_dtheta * cp,
            ),
            axis=-1,
        )

        f_over_st = _divide_by_sin_with_pole_limit(
            f, df_dtheta, theta, st
        )
        Tphi = np.stack(
            (
                np.zeros_like(theta),
                f_over_st * cp,
                -f_over_st * sp,
            ),
            axis=-1,
        )

        return self._finish_direction_evaluation(
            q, Ttheta, Tphi, Erho, Etheta, Ephi, full=full
        )


class OgiveCos(S2Surface):
    r"""The ``ogive_cos`` surface from the original MATLAB ``S2_surface.m``."""

    def __init__(
            self, n: int,
            angle: float = np.pi * 22.62 / 180.0,
            length: float = 1.0):
        if not 0 < angle < np.pi/2:
            raise ValueError("angle must lie in (0, pi/2).")
        if length <= 0:
            raise ValueError("length must be positive.")
        self.angle = float(angle)
        self.length = float(length)
        super().__init__(n)

    def _evaluate(self):
        (
            self._q, _, _, self._Ttheta, self._Tphi,
            _, _, _
        ) = self.evaluate_directions(self.xhat, full=True)

    def evaluate_directions(self, directions: np.ndarray, full: bool = False):
        theta, phi, st, Erho, Etheta, Ephi = _angles_and_frame(directions)
        sp, cp = np.sin(phi), np.cos(phi)
        length = self.length
        d0 = 1.0 - np.cos(self.angle)

        q1 = -0.5 * length * np.cos(theta)
        c = 2.0 * np.sin(self.angle) / length
        radicand = np.maximum(0.0, 1.0 - (c * q1)**2)
        root = np.sqrt(radicand)
        f = (root - 1.0 + d0) * length / (10.0 * d0)

        q = np.stack((q1, f * sp, f * cp), axis=-1)

        dq1_dtheta = 0.5 * length * st
        df_dtheta = (
            (-0.1 * length / d0)
            * (c**2 * 0.5 * length)
            * q1 * st / root
        )
        Ttheta = np.stack(
            (dq1_dtheta, df_dtheta * sp, df_dtheta * cp), axis=-1
        )

        f_over_st = _divide_by_sin_with_pole_limit(
            f, df_dtheta, theta, st
        )
        Tphi = np.stack(
            (
                np.zeros_like(theta),
                f_over_st * cp,
                -f_over_st * sp,
            ),
            axis=-1,
        )

        return self._finish_direction_evaluation(
            q, Ttheta, Tphi, Erho, Etheta, Ephi, full=full
        )


class ConeSphere(S2Surface):
    r"""The ``cone_sphere`` surface from the original MATLAB toolbox.

    A cone is glued C1-smoothly to a sphere.  The default unscaled MATLAB
    parameters ``alpha=23.821``, ``beta=3.306`` and ``gamma=2.947`` are
    rescaled so that ``alpha + beta == length``.
    """

    def __init__(
            self, n: int, length: float = 1.0,
            alpha: float = 23.821, beta: float = 3.306,
            gamma: float = 2.947):
        if length <= 0:
            raise ValueError("length must be positive.")
        if alpha <= 0 or beta <= 0 or gamma <= 0:
            raise ValueError("alpha, beta and gamma must be positive.")

        scale = float(length) / (float(alpha) + float(beta))
        self.alpha = scale * float(alpha)
        self.beta = scale * float(beta)
        self.gamma = scale * float(gamma)
        self.length = float(length)
        super().__init__(n)

    def _evaluate(self):
        (
            self._q, _, _, self._Ttheta, self._Tphi,
            _, _, _
        ) = self.evaluate_directions(self.xhat, full=True)

    def evaluate_directions(self, directions: np.ndarray, full: bool = False):
        theta, phi, st, Erho, Etheta, Ephi = _angles_and_frame(directions)
        sp, cp = np.sin(phi), np.cos(phi)

        alpha = self.alpha
        beta = self.beta
        gamma = self.gamma
        beta0 = beta - gamma

        q1 = -alpha + theta * (alpha + beta) / np.pi
        dq1_dtheta = (alpha + beta) / np.pi

        arg1 = np.maximum(
            0.0, 1.0 - ((q1 - beta0) / gamma)**2
        )
        h1 = gamma * np.sqrt(arg1)
        cone_factor = np.sqrt(
            max(0.0, 1.0 - (-beta0 / gamma)**2)
        )
        h2 = gamma * cone_factor * (q1 + alpha) / alpha

        cone = (q1 <= 0.0) & (q1 >= -alpha)
        sphere = (q1 <= beta) & (q1 > 0.0)
        h = np.where(cone, h2, np.where(sphere, h1, 0.0))

        q = np.stack((q1, h * sp, h * cp), axis=-1)

        with np.errstate(divide="ignore", invalid="ignore"):
            dh1_dtheta = -dq1_dtheta * (q1 - beta0) / h1
        dh2_dtheta = gamma * cone_factor * dq1_dtheta / alpha
        dh_dtheta = np.where(
            cone, dh2_dtheta, np.where(sphere, dh1_dtheta, 0.0)
        )

        Ttheta = np.stack(
            (
                np.full_like(theta, dq1_dtheta),
                dh_dtheta * sp,
                dh_dtheta * cp,
            ),
            axis=-1,
        )

        h_over_st = _divide_by_sin_with_pole_limit(
            h, dh_dtheta, theta, st
        )
        Tphi = np.stack(
            (
                np.zeros_like(theta),
                h_over_st * cp,
                -h_over_st * sp,
            ),
            axis=-1,
        )

        return self._finish_direction_evaluation(
            q, Ttheta, Tphi, Erho, Etheta, Ephi, full=full
        )

######################################################################

class ParameterizedSurface():
    r""" This is an abstract class for S2Surfacess parameterized by coefficients in some vector space.

    Typical forward operators are naturally defined on a shape space. In particular, they 
    are invariant under re-parametrizations of the surface. However, such shape spaces are not vector spaces, and we define forward operators on some vector space of coefficients paraemeterizing a certain class of surface (e.g., star-shape surfaces with radial functions given by spherical harminces of a certain degree).  

    One may think of the mapping taking coefficients to surfaces as an operator, and of the parameterized forward operators as compositions of this operator with the actual forward operator defined 
    on a shape space. However, as mentioned above, shape spaces aren't vector spaces, and it is often  
    computationally more convenient, not to implement the discrete Frechet derivative of the parameterized forward operator, but to discretize a continuous characterization of the Frechet derivative. For this reason the forward operators on shape spaces in this toolbox are not implemented as such compositions.

    Due to invariance under reparameterizations, Frechet derivatives of operators on shape spaces only 
    depend on the normal component of a perturbation of the surface, and this quantity is used 
    in our implementations of the derivatives of parameterized forward operators. More precisely, 
    these derivatives are compositions of a discretization of the continuous derivative on shape space 
    and the operator `self.der_normal`.
    """
    from regpy.operators import Operator
    def __init__(self,coeff:np.ndarray,der_normal:Operator,**kwargs):
        if not coeff in der_normal.domain:
            raise ValueError(Errors.not_a_vecsp(coeff,der_normal.domain))
        if not isinstance(der_normal.domain,ParameterizedSurfaceSpc):
            raise ValueError(Errors.not_instance(der_normal.domain,ParameterizedSurfaceSpc))
        if not isinstance(der_normal.codomain,MeasureSpaceFcts):
            raise ValueError(Errors.not_instance(der_normal.codomain,MeasureSpaceFcts))
        n = der_normal.codomain.shape[0]
        if not der_normal.codomain.shape==(n,2*n):
            raise ValueError(Errors.generic_message('codomain must have shape of length 2.'))
        self.coeff = coeff
        """The coefficient vector characterizing the surface.
        """
        self.der_normal = der_normal
        """Linear `regpy.operators.Operator` given by the inner product of the derivative of the coefficient-to-surface "operator" (i.e. the derivative of `ParameterizedCurveSpc.coeff2surface`) and the normal vector of length 1 of the surface. """


class ParameterizedSurfaceSpc(ABC):
    r""" Abstract base class of vector spaces of coefficients of `ParameterizedSurface`s. 
    """
    def __init__(self,**kwargs):
        pass

    @abstractmethod
    def coeff2surface(self, coeff : np.ndarray, nderivs : int =0)->ParameterizedSurface:
        r"""Compute a curve for the given coefficients.
        """
        pass

class StarSHSurfaceSpc(ParameterizedSurfaceSpc,MeasureSpaceFcts):
    def __init__(self, n:int, sh_deg:int=3):
        if not isinstance(n,int) or not isinstance(sh_deg,int):
            raise TypeError(Errors.type_error(f'n and sh_deg must be integers. {n},{sh_deg}'))
        if sh_deg < 0 or sh_deg>n or n<0:
            raise ValueError(Errors.value_error(msg=f'need 0<=sh_deg<=n. {n},{sh_deg}'))

        # The coefficient vector space carries no Sobolev geometry. Inner
        # products and regularization norms belong to regpy.hilbert and can
        # therefore be selected independently in a Setting.
        MeasureSpaceFcts.__init__(
            self,
            measure=np.ones(((sh_deg+1)**2,), dtype=float),
            dtype=float,
        )
        ParameterizedSurfaceSpc.__init__(self)
        self.n = n
        self.sh_deg = sh_deg
        dat = S2Surface.sphere_quad_data(n)
        self.__dict__.update(dat.__dict__)
        self._compute_SH_basis()

    def _compute_SH_basis(self):
        (
            self.basis,
            self.basis_theta,
            self.basis_phi,
        ) = _real_spherical_harmonics(
            self.sh_deg, self._Erho, derivatives=True
        )

    def coeff2surface(self,coeff:np.ndarray)->StarSHSurface:
        if len(coeff.shape)!=1 or len(coeff) != (self.sh_deg+1)**2:
            raise ValueError(f'coefficient vector must have shape ((sh_deg+1)**2,). sh_deg={self.sh_deg}, {coeff.shape}')
        return StarSHSurface(self,coeff)

    def sphere(self,radius:float=1.)->StarSHSurface:
        """ Return a StarSHSurface which is a sphere of a given radius. 
        This can be used as an initial guess.
        """
        coeff = self.zeros()
        coeff[0]=radius *np.sqrt(4*np.pi)
        return StarSHSurface(self,coeff)            


class StarSHSurface(StarSurface,ParameterizedSurface):
    """
    StarSurface with radial function given by spherical harmonics. 

    Parameters:
    -----------

    spc: StarSHSurfaceSpc
        The underlying vector space of of SH coefficients. 
    n: int
        Parameter of quadrature grid 
    coeff: np.ndarray
        coefficient vector
    """

    class der_normal_StarSHSurface(Operator):
        def __init__(self,spc:StarSHSurfaceSpc,surf:StarSHSurface):
                super().__init__(domain= spc,codomain=surf.grid_space(), linear=True)
                self.basis = spc.basis
                self.xhat_dot_normal = np.sum(surf.normal*surf.xhat,axis=2)

        def _eval(self,coeff):
            radial = np.sum(coeff[None,None,:] * self.basis,axis=2)
            radial *= self.xhat_dot_normal
            return radial

        def _adjoint(self, y):
            aux = y*self.xhat_dot_normal
            return np.sum(aux[:,:,None]*self.basis,axis=(0,1))

    def __init__(self, spc:StarSHSurfaceSpc, coeff:np.ndarray):
        """
        StarSurface with radial function r given by a linear combination of spherical harmonics 
        of some maximal degree.

        Parameters
        ----------
        sh_deg: int
            Maximum harmonic degree
        """

        if not isinstance(coeff,np.ndarray):
            raise TypeError(Errors.type_error("coeff must be a np.ndarray",coeff))
        self.spc = spc
        if not len(coeff.shape)==1 or (spc.sh_deg+1)**2 != len(coeff):
            raise ValueError(Errors.value_error(f"length of coeff must be a perfect square. Given shape: {coeff.shape}"))
        
        self.coeff=coeff
        StarSurface.__init__(self,spc.n)
        der_normal = StarSHSurface.der_normal_StarSHSurface(spc,self)
        ParameterizedSurface.__init__(self,coeff,der_normal=der_normal)

    @S2Surface.n.setter
    def n(self,n_new:int):
        if not isinstance(n_new, (int, np.integer)) or n_new < 0:
            raise ValueError("n must be a non-negative integer.")
        n_new = int(n_new)
        if self.spc.n != n_new:
            raise RuntimeError(
                f"StarSHSurface is tied to its StarSHSurfaceSpc grid: "
                f"spc.n={self.spc.n}, requested n={n_new}."
            )
        self._n = n_new
        for k in [
                "_theta", "_phi", "_st", "_ct", "_sp", "_cp",
                "_Ephi", "_Etheta", "_Erho", "_alpha"]:
            setattr(self, k, getattr(self.spc, k))
        self.__dict__.pop("_q_sh_coeff", None)
        if getattr(self, "_surface_initialized", False):
            self._evaluate()
            self._complete_evaluation()

    def radius_and_derivatives(self)->tuple[np.ndarray,np.ndarray,np.ndarray]:
        """
        Evaluate spherical harmonic field and its surface gradient

        Parameters
        ----------
        coeff : ((N+1)^2,) array
            Spherical harmonic coefficients in MATLAB ordering: index = l^2 + l + m
        """

        r      = np.sum(self.coeff[None,None,:] * self.spc.basis,axis=2)
        rtheta = np.sum(self.coeff[None,None,:] * self.spc.basis_theta,axis=2) 
        rphi   = np.sum(self.coeff[None,None,:] * self.spc.basis_phi,axis=2)

        return r, rtheta, rphi

    def evaluate_directions(self, directions: np.ndarray, full: bool = False):
        basis, btheta, bphi = _real_spherical_harmonics(
            self.spc.sh_deg, directions, derivatives=True
        )
        r = np.einsum("...j,j->...", basis, self.coeff)
        rtheta = np.einsum("...j,j->...", btheta, self.coeff)
        rphi = np.einsum("...j,j->...", bphi, self.coeff)
        _, _, _, Erho, Etheta, Ephi = _angles_and_frame(directions)
        q = r[..., None] * Erho
        Ttheta = rtheta[..., None] * Erho + r[..., None] * Etheta
        Tphi = rphi[..., None] * Erho + r[..., None] * Ephi
        return self._finish_direction_evaluation(
            q, Ttheta, Tphi, Erho, Etheta, Ephi, full=full
        )

    def field2SHcoeffs(self,u:np.ndarray)->np.ndarray:
        """ Computes the spherical harmonics coefficients of a real-valued field on the sphere given 
        at the quadrature points.

        Parameters:
        ----------
        u: np.ndarray

        Returns:
        ------------
        np.ndarray
        """
        if not isinstance(u,np.ndarray) or not np.issubdtype(u.dtype,np.floating):
            raise TypeError('u must be a real np.ndarray')
        if u.shape!=self.grid_shape:
            raise ValueError(f'shape of u does not match: {u.shape},{self.grid_shape}')
        return np.sum(u[:,:,None]*self.quadweights[:,:,None]*self.spc.basis,axis =(0,1))
