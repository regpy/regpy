Operator in RegPy
=================

This tutorial provides a more detailed explanation than the general guide in :ref:`/usage.rst#forward-operator` on how to define custom operators using the base class :class:`Operator`.

Let us consider a forward problem between Hilbert spaces, defined as:

.. math::
    F\colon \mathbb{X}\to\mathbb{Y}.

In `RegPy`, such an operator is interpreted in its discretized form as:

.. math::
    \underline{F}\colon \underline{\mathbb{X}}\to\underline{\mathbb{Y}}

where :math:`\underline{\mathbb{X}}` and :math:`\underline{\mathbb{Y}}` are finite dimensional subspaces of :math:`\mathbb{X}` and :math:`\mathbb{Y}` respectively. In `Regpy`, we explicitly distinguish between the Hilbert space structure and the underlying vector space structure—see :ref:`/spaces.rst` for further details. 

The Hilbert space structure is only introduced when needed, typically during the regularization of an inverse problem. In the implementation, we treat :math:`\underline{\mathbb{X}}= \mathbb{R}^N` and :math:`\underline{\mathbb{Y}}= \mathbb{R}^M`, both equipped with the standard scalar product. 

For a linear forward operator :math:`T`, its discretization :math:`\underline{T}` is a matrix in :math:`\mathbb{R}^{N\times M}`. If the scalar products on :math:`\underline{\mathbb{X}}` and :math:`\underline{\mathbb{Y}}` are represented by the Gram matrices :math:`G_{\underline{\mathbb{X}}}` and :math:`G_{\underline{\mathbb{Y}}}`, then the discrete adjoint :math:`\underline{T^ast}` with respect to the Hilbert space inner products is given by:

.. math::
    \underline{T^\ast} = G_{\underline{\mathbb{X}}}^{-1} \underline{T}^{T} G_{\underline{\mathbb{Y}}}.

Here, :math:`\underline{T}^T` denotes the transpose of the discrete matrix :math:`\underline{T}`, which is the adjoint with respect to the standard Euclidean scalar products in :math:`\mathbb{R}^M` and :math:`\mathbb{R}^N`. 

This decomposition motivates the design of operator implementations in `RegPy`: the adjoint of a linear operator is computed assuming the standard scalar product. If a different scalar product is required, it can later be incorporated by assigning the appropriate space structure (via Gram matrices) to the domain and codomain.

Using existing operators
~~~~~~~~~~~~~~~~~~~~~~~~

The easiest way to construct new operators is by using the existing operators in the `regpy.operators` module. This module provides many standard operators, such as multiplication, Fourier transform, convolution, and more. You can then combine these operators through direct sums or compositions to create new operators.

.. code-block:: python

    op_1 = Some_Operator(...)
    op_2 = Some_other_Operator(...)

    my_op = op_1 * op_2 # a composition which works if the codomain of op_2 == domain op_1
    my_op =+ op_1.codomain.rand() # shifting in the codomain by a random vector

Recall from :ref:`/usage.rst#operator-operations` that you have the following options to combine operators

- `a * op1 + b * op2` for linear combination
- `op1 * op2` composition
- `op * arr` composition with array as point wise multiplication in domain
- `op + arr` operator shifted in codomain



Own linear operator class
~~~~~~~~~~~~~~~~~~~~~~~~~

An operator requires the definition of the vector space structure, meaning you must specify both the `domain` and `codomain` as subclasses of :class:`regpy.vecsps.VectorSpace`. This can be done by passing these values into the initialization method of the class, or by computing them within the class itself.

For a linear operator, you need to implement two methods:

- `_eval`: This method computes the evaluation of the forward operator.
- `_adjoint`: This method computes the adjoint of the forward operator.

A typical implementation might look like this:
.. code-block:: python

    from regpy.operators import Operator

    class My_OwnOperator(Operator):
        def __init__(self,par_1,par_2, ...):
            # Here you may do some initializing computations depending on your parameter
            # In particular you have to compute the domain and codomain if you do not supply them as parameter
            # At the end you have to call the super initialization by:
            super().__init__(
                domain = my_domain, #The particular discretization of the domain associated to a vector in R^N
                codomain = my_codomain, #The particular discretization of the codomain associated to a vector in R^N
                linear=True # has to be set since the default is False
            )

        def _eval(self,x):
            # Compute with x being in the my_domain what the operator evaluates as y=Tx
            return y

        def _adjoint(self,y):
            # Compute with y being in the my_codomain what the standard adjoint operator evaluates as x=T^Ty
            return x

As an easy example you might want to checkout the Volterra problem in :ref:`/notebooks/volterra_main_example.ipynb` and as a more complicated example look at the `ngsolve` operator in :ref:`/notebooks/tfm.ipynb`.

Own non-linear operator class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For non-linear operators, you similarly need to define the domain and codomain. However, you must also define the adjoint of the linearization, which corresponds to the Fréchet derivative at a specific location.

For non-linear operators, you need to implement the following methods:

- `_eval`: his method computes the evaluation of the forward operator. It must also take two extra arguments, `derivative` and `adjoint_derivative`, which are booleans. These arguments determine whether you want to compute the derivative and/or the composition of the adjoint and the derivative.
- `_derivative`:  This method computes the derivative of the forward operator at a specific location. The location is not an argument of the method. One has to construct a full Frechèt derivative operator using the `linearize` of the operator. The returned derivative then uses this method as evaluation.
- `_adjoint`: This method computes the adjoint of the derivative of the forward operator.

The core idea in `RegPy` is that in typical setups, you don't just need the derivative; rather, you require a linearization. Therefore, you need both the evaluation :math:`F(x)` and the linear operator :math:`F'[x]`. However, in some cases, you may need to precompute the derivative at a specific location.

One of the main reasons to force to have such a connection between evaluation and derivate is to prevent the use of a derivative from a different location. This can be for example be seen that `RegPy` revokes a derivate once you evaluate the operator on a new location. Thus if you seriously want to use a derivate of a different location then you have to make a copy of the derivate.

Thus, when `RegPy` linearizes an operator by calling its `linearize` method at a point :math:`x`, the following steps occur:

1. The operator is evaluated, and the optional argument `derivative=True` is passed, so that the `_eval` method knows a derivative is required.
2. Using the optional parameters, the operator prepares for linearization by precomputing and storing certain parameters as attributes.
3. The linearize method returns both the evaluation (as an object in the codomain) and the derivative, represented as an :class:`Operator` mapping from the domain to the codomain.

.. code-block:: python
    
    y, derive = my_op.linearize(x)

Note that the optional third argument is only necessary, if you wish to also get the composition of the adjoint and derivate :math:`F'[x]^\ast F'[x]`. Then `linearize` returns a third object that is also an operator mapping from the domain to the domain.

.. code-block:: python

    y, derive, adjoint_deriv = my_op.linearize(x,adjoint_derivative=True)

Thus a typical implementation would look like this:

.. code-block:: python

    from regpy.operators import Operator

    class My_OwnOperator(Operator):
        def __init__(self,par_1,par_2, ...):
            # Here you may do some initializing computations depending on your parameter
            # In particular you have to compute the domain and codomain if you do not supply them as parameter
            # At the end you have to call the super initialization by:
            super().__init__(
                domain = my_domain, #The particular discretization of the domain associated to a vector in R^N
                codomain = my_codomain, #The particular discretization of the codomain associated to a vector in R^N
                linear=False # can also be left since the default is False
            )

        def _eval(self,x, derivative = False, adjoint_derivative = False):
            # Compute with x being in the my_domain what the operator evaluates as y=Tx
            if derivate:
                self.x = x # Storing the location at which the linearization takes place
                # make necessary precomutatoins for derivative at x
            if adjoint_derivative:
                # make necessary precomutatoins for the composition of adjoint and derivative
            return y

        def _derivative(self,x):
            # compute for x in the my_domain the derivative y = F'[self.x](x) at the predefined location self.x
            return y

        def _adjoint(self,y):
            # Compute with y being in the my_codomain what the standard adjoint of the derivative x = F'[self.x]*(y) at the predefined location self.x
            return x

If one wishes to additionally use the evaluation of the composition of adjoint and derivative one can redefine the method `_adjoint_derivative` of the operator such that

.. code-block:: python

    def _adjoint_derivative(self,x):
    # compute for x in the my_domain the composition of derivative and its adjoint x = F'[self.x]*F'[self.x](x) at the predefined location self.x
    return x

As an easy example you might want to checkout the Volterra problem in :ref:`/notebooks/volterra_main_example.ipynb` for the exponent not equal one we have a non-linear operator.


`ngsolve` Operators
~~~~~~~~~~~~~~~~~~~

`RegPy` offers an interface for `ngsolve` to allow for implementations of your favourite inverse problem with a nice pde solver and use `RegPy` to regularize. For this the library provides 3 extra modules in the operators `regpy.operators.ngsolve`, the Hilbert spaces `regpy.hilbert.ngsolve` and for functionals `regpy.functionals.ngsolve`.

Note that even for these type of operators `RegPy` requires you to implement the adjoint with respect to the standard scalar product!

If you want to implement your own operator you should use the base class :class:`NgsOperator`. With this you have already some basic methods to deal with the `numpy` interface in `regpy` and the `ngsolve` interface. **This Interface will be changed**

For many problems that are based upon a scalar parameter identification problem using a second order elliptic linear pde we have implemented an operator base class :class:`SecondOrderEllipticCoefficientPDE`. You can use this class to define your own operator and what remains to do is to implement you bilinear from and linear form to define such an operator. For an example checkout diffusion problem in :ref:`/notebooks/diffusion_coefficient.ipynb`.
