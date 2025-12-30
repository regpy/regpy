from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
from typing import Optional
from scipy.special import lpmv, factorial

@dataclass(frozen=True)
class SphereQuadrature:
    n_theta: int
    n_phi: int

    def nodes_and_weights(self):
        # Gauss–Legendre in μ = cos(θ)
        mu, w_theta = np.polynomial.legendre.leggauss(self.n_theta)

        theta = np.arccos(mu)               # θ ∈ (0, π)
        phi = np.linspace(0, 2*np.pi, self.n_phi, endpoint=False)

        # Tensor grid
        TH, PH = np.meshgrid(theta, phi, indexing="ij")

        # Quadrature weights
        w_phi = (2*np.pi) / self.n_phi
        W = w_theta[:, None] * w_phi        # shape (nθ, nφ)

        return TH, PH, W



def spherical_basis(x_hat: NDArray[np.float64]):
    """
    Input:
    xhat: np.ndarray
        A 3xn array the columns of which are unit vectors

    Output:
        theta, phi: spherical coordinates of xhat
        st, ct: sin and cos of theta
        sp, cp: sin and cos of phi
        Erho, Etheta, Ephi: standard orthonormal basis in spherical coordinates at each point in xhat
    """
    x, y, z = x_hat

    theta = np.arccos(z)
    phi = np.arctan2(y, x)

    st = np.sin(theta)
    ct = np.cos(theta)
    sp = np.sin(phi)
    cp = np.cos(phi)

    Erho = np.stack((st * cp, st * sp, ct))
    Etheta = np.stack((ct * cp, ct * sp, -st))
    Ephi = np.stack((-sp, cp, np.zeros_like(phi)))

    return theta, phi, st, ct, sp, cp, Erho, Etheta, Ephi


class S2Surface(ABC):
    def __init__(self, n: int):
        self.n = n

    @abstractmethod
    def evaluate(self, x_hat: NDArray[np.float64]):
        """Return q, Jacobian, normal, Ttheta, Tphi"""



class RadialSurface(S2Surface, ABC):
    @abstractmethod
    def radius_and_derivatives(self, theta, phi):
        """Return r, dr/dθ, dr/dφ"""

    def evaluate(self, theta, phi):
        st, ct = np.sin(theta), np.cos(theta)
        sp, cp = np.sin(phi), np.cos(phi)

        Er = np.stack((st*cp, st*sp, ct))
        Et = np.stack((ct*cp, ct*sp, -st))
        Ep = np.stack((-sp, cp, np.zeros_like(phi)))

        r, rt, rp = self.radius_and_derivatives(theta, phi)

        q = r * Er
        Tt = rt * Er + r * Et
        Tp = rp * Er + r * st * Ep

        normal = np.cross(Tt.T, Tp.T).T
        Jacobian = np.linalg.norm(n, axis=0)
        normal /= Jacobian

        return q, Jacobian, normal

    """def evaluate(self, x_hat):
        theta, phi, st, ct, sp, cp, Erho, Etheta, Ephi = spherical_basis(x_hat)

        r = self.radius(theta, phi)
        q = r * Erho

        dr_dtheta = np.gradient(r, theta, axis=0)
        dr_dphi = np.gradient(r, phi, axis=1)

        Ttheta = dr_dtheta * Erho + r * Etheta
        Tphi = dr_dphi * Erho + r * Ephi

        normal = np.cross(Ttheta.T, Tphi.T).T
        Jacob = np.linalg.norm(normal, axis=0)
        normal /= Jacob

        return q, Jacob, normal, Ttheta, Tphi
        """

@dataclass
class SurfaceEvaluation:
    q: NDArray
    normal: NDArray
    Jacobian: NDArray
    weights: NDArray

    def weighted_J(self):
        return self.Jacobian * self.weights

def evaluate_on_sphere(surface: S2Surface, quad):
    theta, phi, W = quad.nodes_and_weights()
    q, J, n = surface.evaluate(theta, phi)
    return q, J, n, W

"""
def unit_vectors(TH, PH):
    st = np.sin(TH)
    return np.stack((
        st * np.cos(PH),
        st * np.sin(PH),
        np.cos(TH)
    ))

def evaluate_on_sphere(surface, quad: SphereQuadrature) -> SurfaceEvaluation:
    TH, PH, W = quad.nodes_and_weights()

    x_hat = unit_vectors(TH, PH)

    q, J, normal, Tt, Tp = surface.evaluate(x_hat)

    return SurfaceEvaluation(
        q=q,
        normal=normal,
        Jacobian=J,
        weights=W
    )
"""

class ParametricSurface(S2Surface, ABC):
    @abstractmethod
    def parametrization(self, theta, phi):
        """Return q, Tθ, Tφ"""

    def evaluate(self, theta, phi):
        q, Tt, Tp = self.parametrization(theta, phi)

        n = np.cross(Tt.T, Tp.T).T
        J = np.linalg.norm(n, axis=0)
        n /= J

        return q, J, n

class DirectionalSurface(S2Surface, ABC):
    @abstractmethod
    def support_function(self, xhat):
        """Return q(x̂)"""

    def evaluate(self, theta, phi):
        st, ct = np.sin(theta), np.cos(theta)
        sp, cp = np.sin(phi), np.cos(phi)

        xhat = np.stack((st*cp, st*sp, ct))
        q = self.support_function(xhat)

        # Finite-difference tangents (localized, controlled)
        eps = 1e-6
        q_t = self.support_function(
            np.stack((np.sin(theta+eps)*cp, np.sin(theta+eps)*sp, np.cos(theta+eps)))
        )
        q_p = self.support_function(
            np.stack((st*np.cos(phi+eps), st*np.sin(phi+eps), ct))
        )

        Tt = (q_t - q) / eps
        Tp = (q_p - q) / eps

        n = np.cross(Tt.T, Tp.T).T
        J = np.linalg.norm(n, axis=0)
        n /= J

        return q, J, n


class Ball(RadialSurface):
    def __init__(self, R=0.5):
        self.R = R

    def radius_and_derivatives(self, theta, phi):
        r = self.R * np.ones_like(theta)
        return r, np.zeros_like(theta), np.zeros_like(theta)


@dataclass
class Ellipsoid(S2Surface):
    A: float
    B: float
    C: float

    def evaluate(self, x_hat):
        _, _, st, ct, sp, cp, Erho, Etheta, Ephi = spherical_basis(x_hat)

        q = np.vstack((
            self.A * Erho[0],
            self.B * Erho[1],
            self.C * Erho[2]
        ))

        normal = np.vstack((
            self.B * self.C * Erho[0],
            self.A * self.C * Erho[1],
            self.A * self.B * Erho[2]
        ))

        Jacob = np.linalg.norm(normal, axis=0)
        normal /= Jacob

        Ttheta = np.vstack((
            self.A * ct * cp,
            self.B * ct * sp,
            -self.C * st
        ))

        Tphi = np.vstack((
            -self.A * sp,
            self.B * cp,
            np.zeros_like(cp)
        ))

        return q, Jacob, normal, Ttheta, Tphi


class Peanut(RadialSurface):
    def __init__(self, alpha=0.25):
        self.alpha = alpha

    def radius_and_derivatives(self, theta, phi):
        st, ct = np.sin(theta), np.cos(theta)
        s2 = 2*st*ct
        c2 = ct*ct - st*st

        A = self.alpha + 1
        B = A - s2*s2
        sqrtB = np.sqrt(B)

        C = c2 + sqrtB
        sqrtC = np.sqrt(C)

        denom = 2*np.sqrt(1+np.sqrt(A))
        r = sqrtC / denom

        ds2 = 2*c2
        dc2 = -2*s2
        dB = -2*s2*ds2
        dC = dc2 + dB/(2*sqrtB)
        dr = dC/(2*sqrtC*denom)

        return r, dr, np.zeros_like(theta)

"""@dataclass(frozen=True)
class Peanut(RadialSurface):
    alpha: float = 0.25

    def radius_and_derivative(self, theta):
        st = np.sin(theta)
        ct = np.cos(theta)

        # sin(2θ), cos(2θ)
        s2 = 2 * st * ct
        c2 = ct * ct - st * st

        # inner expressions
        A = self.alpha + 1.0
        B = A - s2 * s2              # = alpha + 1 - sin²(2θ)
        sqrtB = np.sqrt(B)

        C = c2 + sqrtB
        sqrtC = np.sqrt(C)

        denom = 2.0 * np.sqrt(1.0 + np.sqrt(A))
        r = sqrtC / denom

        # ----- derivative dr/dθ (exact) -----

        # d/dθ sin(2θ) = 2 cos(2θ)
        ds2 = 2 * c2

        # d/dθ cos(2θ) = -2 sin(2θ)
        dc2 = -2 * s2

        dB = -2 * s2 * ds2
        d_sqrtB = dB / (2 * sqrtB)

        dC = dc2 + d_sqrtB
        d_sqrtC = dC / (2 * sqrtC)

        dr = d_sqrtC / denom

        return r, dr

    def evaluate(self, x_hat: NDArray[np.float64]):
        theta, phi, st, ct, sp, cp, Erho, Etheta, Ephi = spherical_basis(x_hat)

        r, dr = self.radius_and_derivative(theta)

        # Surface points
        q = r * Erho

        # Tangent vectors
        Ttheta = dr * Erho + r * Etheta
        Tphi = r * st * Ephi

        # Normal and Jacobian
        normal = np.cross(Ttheta.T, Tphi.T).T
        Jacob = np.linalg.norm(normal, axis=0)
        normal /= Jacob

        return q, Jacob, normal, Ttheta, Tphi
        """

class Ellipsoid(ParametricSurface):
    def __init__(self, a, b, c):
        self.a, self.b, self.c = a, b, c

    def parametrization(self, theta, phi):
        st, ct = np.sin(theta), np.cos(theta)
        sp, cp = np.sin(phi), np.cos(phi)

        q = np.stack((self.a*st*cp, self.b*st*sp, self.c*ct))
        Tt = np.stack((self.a*ct*cp, self.b*ct*sp, -self.c*st))
        Tp = np.stack((-self.a*st*sp, self.b*st*cp, np.zeros_like(phi)))

        return q, Tt, Tp

class SuperEllipsoid(RadialSurface):
    def __init__(self, a, b, c, p):
        self.a, self.b, self.c, self.p = a, b, c, p

    def radius_and_derivatives(self, theta, phi):
        st, ct = np.sin(theta), np.cos(theta)
        sp, cp = np.sin(phi), np.cos(phi)

        x = st*cp/self.a
        y = st*sp/self.b
        z = ct/self.c

        f = (np.abs(x)**self.p +
             np.abs(y)**self.p +
             np.abs(z)**self.p)**(1/self.p)

        r = 1/f
        return r, np.zeros_like(theta), np.zeros_like(theta)

class RoundedPolyhedron(DirectionalSurface):
    def __init__(self, vertices, radius):
        self.V = vertices
        self.radius = radius

    def support_function(self, xhat):
        dots = np.tensordot(self.V, xhat, axes=(1,0))
        idx = np.argmax(dots, axis=0)
        v = self.V[idx].T
        return v + self.radius * xhat

######################################################################

@dataclass
class SphericalHarmonicsCache:
    N: int
    P: list              # P[l][m+l, theta]
    dP: list             # d/dtheta P[l][m+l, theta]
    exp_imphi: np.ndarray  # shape (n_phi, 2N+1)


"""def harmonics_to_surface(
    coeff: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    Etheta: np.ndarray,
    Ephi: np.ndarray,
    *,
    N: int,
    cache: Optional[SphericalHarmonicsCache] = None,
    compute_gradient: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    ###
    Parameters
    ----------
    coeff : ( (N+1)^2, ) complex
        Spherical harmonic coefficients, ordered as in MATLAB:
        index = l^2 + l + m
    theta : (n_theta,)
    phi   : (n_phi,)
    Etheta, Ephi : (3, n_theta * n_phi)
        Tangent vectors on S^2
    N : int
        Harmonic truncation
    cache : optional SphericalHarmonicsCache
    ###

    theta = theta.reshape(-1)
    phi = phi.reshape(-1)

    n_theta = theta.size
    n_phi = phi.size

    sin_theta = np.sin(theta)[:, None]

    # --------------------------------------------------
    # Precomputation (if not supplied)
    # --------------------------------------------------
    if cache is None:
        from scipy.special import lpmv

        P = []
        dP = []

        x = np.cos(theta)

        for l in range(N + 1):
            Pl = np.zeros((2*l + 1, n_theta))
            dPl = np.zeros_like(Pl)

            for m in range(-l, l + 1):
                Pl[m + l] = lpmv(m, l, x)
                # derivative w.r.t. theta
                dPl[m + l] = -np.sin(theta) * np.gradient(Pl[m + l], theta)

            P.append(Pl)
            dP.append(dPl)

        exp_imphi = np.exp(1j * phi[:, None] * np.arange(-N, N + 1))

    else:
        P = cache.P
        dP = cache.dP
        exp_imphi = cache.exp_imphi

    # --------------------------------------------------
    # Field evaluation
    # --------------------------------------------------
    H = np.zeros((n_theta, n_phi), dtype=complex)

    if compute_gradient:
        dtheta_H = np.zeros_like(H)
        dphi_H = np.zeros_like(H)

    for m in range(-N, N + 1):
        res_P = np.zeros(n_theta, dtype=complex)
        res_dP = np.zeros(n_theta, dtype=complex)

        for l in range(abs(m), N + 1):
            idx = l*l + l + m
            res_P += coeff[idx] * P[l][m + l]
            if compute_gradient:
                res_dP += coeff[idx] * dP[l][m + l]

        exp_mphi = exp_imphi[:, m + N]  # (n_phi,)

        H += res_P[:, None] * exp_mphi[None, :]

        if compute_gradient:
            dtheta_H += res_dP[:, None] * exp_mphi[None, :]
            dphi_H += (1j * m) * res_P[:, None] * exp_mphi[None, :]

    # --------------------------------------------------
    # Surface gradient
    # --------------------------------------------------
    gradS2 = None
    if compute_gradient:
        dtheta_H = dtheta_H.reshape(1, -1)
        dphi_H = (dphi_H / sin_theta).reshape(1, -1)

        gradS2 = (
            dtheta_H * Etheta
            +
            dphi_H * Ephi
        )

    return H, gradS2
"""

class SphericalHarmonicField:
    def __init__(self, N: int, theta: np.ndarray, phi: np.ndarray):
        """
        Parameters
        ----------
        N : int
            Maximum harmonic degree
        theta : (n_theta,) array
            Polar angles [0, pi]
        phi : (n_phi,) array
            Azimuthal angles [0, 2*pi]
        """
        self.N = N
        self.theta = theta.reshape(-1)
        self.phi = phi.reshape(-1)

        self.n_theta = self.theta.size
        self.n_phi = self.phi.size

        # Precompute Legendre polynomials and derivatives (MATLAB normalization)
        self.P = []
        self.dP = []

        x = np.cos(self.theta)

        for l in range(N + 1):
            Pl = np.zeros((2*l + 1, self.n_theta))
            dPl = np.zeros_like(Pl)
            for m in range(-l, l + 1):
                # MATLAB normalization: multiply lpmv by (-1)^m * factorial(l-m)/factorial(l+m)
                norm = (-1)**m * np.sqrt(factorial(l - abs(m))/factorial(l + abs(m)))
                Pl[m + l] = norm * lpmv(abs(m), l, x)
                # derivative w.r.t theta
                dPl[m + l] = -np.sin(self.theta) * np.gradient(Pl[m + l], self.theta)
            self.P.append(Pl)
            self.dP.append(dPl)

        # Precompute exp(i m phi)
        self.exp_imphi = np.exp(1j * self.phi[:, None] * np.arange(-N, N + 1))

    def evaluate(
        self,
        coeff: np.ndarray,
        Etheta: Optional[np.ndarray] = None,
        Ephi: Optional[np.ndarray] = None,
        compute_gradient: bool = True
    ):
        """
        Evaluate spherical harmonic field and optionally its surface gradient

        Parameters
        ----------
        coeff : ((N+1)^2,) array
            Spherical harmonic coefficients in MATLAB ordering: index = l^2 + l + m
        Etheta, Ephi : (3, n_theta*n_phi) arrays, optional
            Tangent vectors on the surface (required if compute_gradient=True)
        compute_gradient : bool
            Whether to compute surface gradient

        Returns
        -------
        H : (n_theta, n_phi) array
            Field values
        gradS2 : (3, n_theta*n_phi) array or None
            Surface gradient
        """
        N = self.N
        theta = self.theta
        phi = self.phi
        n_theta, n_phi = self.n_theta, self.n_phi
        sin_theta = np.sin(theta)[:, None]

        H = np.zeros((n_theta, n_phi), dtype=complex)
        if compute_gradient:
            if Etheta is None or Ephi is None:
                raise ValueError("Etheta and Ephi must be provided for gradient computation")
            dtheta_H = np.zeros_like(H)
            dphi_H = np.zeros_like(H)

        for m in range(-N, N + 1):
            res_P = np.zeros(n_theta, dtype=complex)
            res_dP = np.zeros(n_theta, dtype=complex) if compute_gradient else None

            for l in range(abs(m), N + 1):
                idx = l*l + l + m
                res_P += coeff[idx] * self.P[l][m + l]
                if compute_gradient:
                    res_dP += coeff[idx] * self.dP[l][m + l]

            exp_mphi = self.exp_imphi[:, m + N]  # shape (n_phi,)
            H += res_P[:, None] * exp_mphi[None, :]

            if compute_gradient:
                dtheta_H += res_dP[:, None] * exp_mphi[None, :]
                dphi_H += 1j * m * res_P[:, None] * exp_mphi[None, :]

        gradS2 = None
        if compute_gradient:
            dtheta_H_flat = dtheta_H.reshape(1, -1)
            dphi_H_flat = (dphi_H / sin_theta).reshape(1, -1)
            gradS2 = dtheta_H_flat * Etheta + dphi_H_flat * Ephi

        return H, gradS2


"""import pyshtools as pysh
class SphericalHarmonicField:
    ###
    Represents a scalar field on the sphere expanded in spherical harmonics.
    Uses pyshtools internally for fast evaluation and gradient computation.
    ###
    def __init__(self, coeffs=None, lmax=None, normalization='schmidt'):
        ###
        Initialize the field.
        
        Parameters
        ----------
        coeffs : array_like, optional
            MATLAB-style 1D coefficients in order l^2 + l + m, including negative m.
        lmax : int
            Maximum degree of the expansion. Required if coeffs is given.
        normalization : str
            Normalization to use ('4pi', 'schmidt', or 'unnorm').
        ###
        self.normalization = normalization
        if coeffs is not None:
            assert lmax is not None, "Must provide lmax if coeffs are given"
            self.lmax = lmax
            self.coeffs = self._convert_matlab_coeffs(coeffs, lmax)
        else:
            self.lmax = 0
            self.coeffs = None

    def _convert_matlab_coeffs(self, coeffs, lmax):
        """
        Convert MATLAB linear coefficients to pyshtools (C_lm/S_lm) array.
        """
        clm = np.zeros((2, lmax+1, lmax+1))
        for l in range(lmax+1):
            for m in range(0, l+1):
                # MATLAB linear index: l^2 + l + m
                idx_pos = l**2 + l + m
                clm[0, l, m] = coeffs[idx_pos].real  # cosine term
                clm[1, l, m] = coeffs[idx_pos].imag  # sine term
        return clm

def evaluate(self, theta, phi):
        ###
        Evaluate H(theta,phi) and gradS2(theta,phi) in Cartesian coordinates.
        theta, phi in radians
        Returns: H (lt x lp), gradS2 (3 x lt*lp)
        ###
        theta = np.asarray(theta)
        phi = np.asarray(phi)
        lt, lp = theta.size, phi.size

        # Associated Legendre functions and derivatives
        plm = pysh.legendre.Plm(self.lmax, theta)
        dplm = pysh.legendre.PlmDeriv(self.lmax, theta)

        # exp(i*m*phi) matrix
        mvec = np.arange(-self.lmax, self.lmax+1)
        exp_imphi = np.exp(1j * np.outer(phi, mvec))  # lp x (2*lmax+1)

        # Initialize
        H = np.zeros((lt, lp), dtype=complex)
        grad_theta = np.zeros((lt, lp), dtype=complex)
        grad_phi = np.zeros((lt, lp), dtype=complex)

        # Vectorized sum over m
        for idx_m, m in enumerate(mvec):
            res = np.zeros(lt, dtype=complex)
            dres = np.zeros(lt, dtype=complex)
            for l in range(abs(m), self.lmax+1):
                c_idx = l**2 + l + m
                coef = self.coeffs[c_idx]
                res += coef * plm[l, abs(m), :]
                dres += coef * dplm[l, abs(m), :]
            exp_col = exp_imphi[:, idx_m]  # lp
            H += np.outer(res, exp_col)
            grad_theta += np.outer(dres, exp_col)
            grad_phi += np.outer(1j*m*res, exp_col)

        # Compute tangent vectors in Cartesian coordinates
        st = np.sin(theta)[:, None]
        ct = np.cos(theta)[:, None]
        sp = np.sin(phi)[None, :]
        cp = np.cos(phi)[None, :]

        # Etheta and Ephi as 3 x (lt*lp)
        Etheta = np.array([ct, np.zeros_like(ct), -st])  # along theta
        Ephi   = np.array([-sp, cp, np.zeros_like(sp)])  # along phi
        # Repeat to match grid
        Etheta = np.repeat(Etheta, lp, axis=1)
        Ephi = np.repeat(Ephi, lt, axis=1)

        # Vectorized gradS2
        gradS2 = (grad_theta.ravel()[None,:] * Etheta +
                  (grad_phi.ravel()/st.ravel())[None,:] * Ephi)

        return H.real, gradS2.real
def evaluate(self, theta, phi):
        ###
        Evaluate H(theta,phi) and gradS2(theta,phi) using full vectorization.
        theta, phi in radians
        Returns: H (lt x lp), gradS2 (3 x lt*lp)
        ###
        theta = np.asarray(theta)
        phi = np.asarray(phi)
        lt, lp = theta.shape[0], phi.shape[0]

        # Compute associated Legendre functions
        # pyshtools: 'Plm' returns array of shape (lmax+1, lmax+1, theta.size)
        plm = pysh.legendre.Plm(self.lmax, theta)
        dplm = pysh.legendre.PlmDeriv(self.lmax, theta)

        # Build exp(i*m*phi) matrix (lp x 2*lmax+1)
        mvec = np.arange(-self.lmax, self.lmax+1)
        exp_imphi = np.exp(1j * np.outer(phi, mvec))  # shape: lp x (2*lmax+1)

        # Vectorized computation over m
        H = np.zeros((lt, lp), dtype=complex)
        grad_theta = np.zeros((lt, lp), dtype=complex)
        grad_phi = np.zeros((lt, lp), dtype=complex)

        for idx_m, m in enumerate(mvec):
            res = np.zeros(lt, dtype=complex)
            dres = np.zeros(lt, dtype=complex)
            for l in range(abs(m), self.lmax+1):
                c_idx = l**2 + l + m
                coef = self.coeffs[c_idx]
                res += coef * plm[l, abs(m), :]
                dres += coef * dplm[l, abs(m), :]
            exp_col = exp_imphi[:, idx_m]  # size lp
            H += np.outer(res, exp_col)
            grad_theta += np.outer(dres, exp_col)
            grad_phi += np.outer(1j*m*res, exp_col)

        # gradS2: combine Etheta and Ephi
        st = np.sin(theta)[:, None]
        Etheta = np.array([[1],[0],[0]])  # Placeholder; replace with actual 3D vectors
        Ephi = np.array([[0],[1],[0]])    # Placeholder; replace with actual 3D vectors
        gradS2 = (grad_theta.ravel()[None,:] * Etheta +
                  (grad_phi.ravel()/st.ravel())[None,:] * Ephi)

        return H.real, gradS2.real
    def evaluate(self, theta, phi):
        ###
        Evaluate the spherical harmonic field at points (theta, phi).
        Theta and phi are in radians.
        ###
        # Convert to grids suitable for pyshtools
        theta = np.asarray(theta)
        phi = np.asarray(phi)
        # Make regular lat/lon grids for DH sampling
        lat = 90 - np.degrees(theta)  # colatitude -> latitude
        lon = np.degrees(phi)
        grid = pysh.expand.MakeGridDH(self.coeffs, sampling=2, lmax=self.lmax,
                                      norm=self.normalization)
        # Interpolation to arbitrary points
        from scipy.interpolate import RegularGridInterpolator
        lats = np.linspace(90, -90, grid.shape[0])
        lons = np.linspace(0, 360, grid.shape[1], endpoint=False)
        interp = RegularGridInterpolator((lats, lons), grid, bounds_error=False, fill_value=None)
        points = np.stack([lat.ravel(), lon.ravel()], axis=-1)
        values = interp(points).reshape(theta.shape)
        return values

    def gradient(self, theta, phi):
        ###
        Compute the gradient on the sphere (∇_θ, ∇_φ components).
        ###
        theta = np.asarray(theta)
        phi = np.asarray(phi)
        # Compute theta/phi derivatives using pyshtools
        grad_theta, grad_phi = pysh.expand.SHGradDH(self.coeffs, lmax=self.lmax,
                                                    norm=self.normalization)
        # Interpolation to arbitrary points
        lat = 90 - np.degrees(theta)
        lon = np.degrees(phi)
        lats = np.linspace(90, -90, grad_theta.shape[0])
        lons = np.linspace(0, 360, grad_theta.shape[1], endpoint=False)
        interp_theta = RegularGridInterpolator((lats, lons), grad_theta, bounds_error=False, fill_value=None)
        interp_phi = RegularGridInterpolator((lats, lons), grad_phi, bounds_error=False, fill_value=None)
        points = np.stack([lat.ravel(), lon.ravel()], axis=-1)
        grad_theta_val = interp_theta(points).reshape(theta.shape)
        grad_phi_val = interp_phi(points).reshape(theta.shape)
        return grad_theta_val, grad_phi_val

    @classmethod
    def from_random(cls, lmax, seed=None, normalization='schmidt'):
        """
        Create a random spherical harmonic field.
        """
        rng = np.random.default_rng(seed)
        coeffs = rng.standard_normal((2, lmax+1, lmax+1))
        field = cls(lmax=lmax, normalization=normalization)
        field.coeffs = coeffs
        return field
"""