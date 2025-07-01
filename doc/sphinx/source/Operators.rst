Operators in RegPy
==================

This tutorial provides a more detailed explanation than the general guide in :ref:`/usage.rst#forward-operator` on how to define custom operators using the base class :class:`Operator`.

Let us consider a forward problem between Hilbert spaces, defined as:

.. math::
    F\colon \mathbb{X}\to\mathbb{Y}.

In `RegPy`, such an operator is interpreted in its discretized form as:

.. math::
    \underline{F}\colon \underline{\mathbb{X}}\to\underline{\mathbb{Y}}

where :math:`\underline{\mathbb{X}}` and :math:`\underline{\mathbb{Y}}` are finite dimensional subspaces of :math:`\mathbb{X}` and :math:`\mathbb{Y}` respectively. In `RegPy`, we explicitly distinguish between the Hilbert space structure and the underlying vector space structure — see :ref:`/spaces.rst` for further details.

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

* `a * op1 + b * op2` for linear combination
* `op1 * op2` composition
* `op * arr` composition with array as point wise multiplication in domain
* `op + arr` operator shifted in codomain


.. _Linear_Operator:

Linear operators
~~~~~~~~~~~~~~~~

An operator requires the definition of the vector space structure, meaning you must specify both the `domain` and `codomain` as subclasses of :class:`regpy.vecsps.VectorSpace`. This can be done by passing these values into the initialization method of the class, or by computing them within the class itself. These domains only define the basic vector structure that you want to use in the operator.

The initialization
------------------

Assuming for example you want to define an operator that maps from a uniformly discretized square domain to some the you could let the operator take tuples `(start,end,number)` for each dimension or some `numpy.linspace` instances to construct the according uniform grid space :class:`regpy.vecsps.UniformGridFcts`. A typical init could look like:

.. code-block:: python

    def __init__(self,d_1,d_2,cd_1,cd_2):
        domain = UniformGridFcts(d_1,d_2)
        codomain = UniformGridFcts(cd_1,cd_2)
        super().__init__(
            domain = UnifromGrid(d_1,d_2),
            codomain = UnifromGrid(cd_1,cd_2),
            linear = True
        )

For a linear operator, you need to implement two methods:

- `_eval`: This method computes the evaluation of the forward operator.
- `_adjoint`: This method computes the adjoint of the forward operator.

Example
^^^^^^^

For example for a two dimensional Fourier transform on a centred square uniform grid. That is the domain is assumed to be :class:`regpy.vecsps.UniformGridFcts` that is two dimension for example :code:`domain=UniformGridFcts(d, d, dtype = np.complex128)` and where `d` defines a centred interval for example by :code:`d = (-1,1,100)`. Moreover, from the domain we can construct the codomain as a uniform grid computing the spacing from the spacing in the domain. Thus we obtain an initialization as follows

.. code-block:: python

    def __init__(self,d):
        domain = UniformGridFcts(d,d,dtype = complex)
        cd = (-1/2/domain.spacing[0],1/2/domain.spacing[0],domain.shape)
        codomain = UniformGridFcts(cd,cd,dtype = complex)
        super().__init__(
            domain = domain,
            codomain = codomain,
            linear = True
        )


The evaluation method
---------------------

The `_eval` method for a linear operator only takes one mandatory input usually named `x`. The method is only called by the super method `eval` which it self receives the input when an instances of the class gets called on a particular values. To be sure that the argument is in the domain the super method `eval` which should not be touched asserts if the argument belongs to the space. So the method that you have to implement can assume that `x` belongs to the domain which you have specified in the initialization. Your method is then required to return a value that belongs to the codomain. This property will be asserted in the evaluation to guarantee that the implementation is returning a valid object which can be treated as an element in the codomain.

Example
^^^^^^^

For the example, of the two dimensional Fourier transform we can use the `numpy` FFT implementation and define the evaluation as

.. code-block:: python

    def _eval(self,x):
        return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x), norm='ortho'))

The adjoint evaluation method
-----------------------------

The `_adjoint` method works similarly to the `_eval` method. The operators adjoint is accessed by `my_op.adjoint(y)` to evaluate the adjoint. Corresponding to the evaluation the `adjoint` method which calls your particular implementation asserts first if the argument belongs to the codomain and then if the computed result form your method belongs to the domain. Thus guaranteeing a minimum constancy when using the methods. Most important `regpy` assumes that the implementation of the adjoint is with respect to the standard real scalar product :math:`\langle x,y\rangle = x^T y`.

In case you are unsure if your implementation of the adjoint works, we provide a utility check for operators in :meth:`regpy.util.operator_tests.test_adjoint`. Which you may use to assert if your adjoint is sufficiently good.

Example
^^^^^^^

Now for the example above we know that the inverse Fourier transform defines our adjoint so that we may implement the adjoint

.. code-block:: python

    def _adjoint(self,y):
        return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(y), norm='ortho'))


Defining the class
------------------

Thus if we combine the methods you can implement your own class for a linear operator has a typical structure as follows:

.. code-block:: python

    from regpy.operators import Operator
    class My_OwnOperator(Operator):
        def __init__(self,par_1,par_2, ...):
            # Here you may do some initializing computations depending on your parameter
            # In particular you have to compute the domain and codomain if you do not
            # supply them as parameter
            # At the end you have to call the super initialization by:
            super().__init__(
                domain = my_domain,
                #The particular discretization of the domain associated to a vector in R^N
                codomain = my_codomain,
                #The particular discretization of the codomain associated to a vector in R^N
                linear=True
                # has to be set since the default is False
            )

        def _eval(self,x):
            # Compute with x being in the my_domain what the operator evaluates as y=Tx
            return y

        def _adjoint(self,y):
            # Compute with y being in the my_codomain what the standard
            # adjoint operator evaluates as x=T^Ty
            return x

As an easy example you might want to checkout the Volterra problem in :ref:`/notebooks/volterra_main_example.ipynb` and as a more complicated example look at the `ngsolve` operator in :ref:`/notebooks/tfm.ipynb`.

Example
^^^^^^^

Returning the example of the Fourier transform we can combine the above code snippets to the following class

.. code-block:: python

    from regpy.operators import Operator
    from regpy.vecsps import UniformGridFcts
    class SimpleFFTOnSquare(Operator):
        def __init__(self,d):
            domain = UniformGridFcts(d,d,dtype = complex)
            cd = (-1/2/domain.spacing[0],1/2/domain.spacing[0],domain.shape[0])
            codomain = UniformGridFcts(cd,cd,dtype = complex)
            super().__init__(
                domain = domain,
                codomain = codomain,
                linear = True
            )

        def _eval(self,x):
            return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x), norm='ortho'))

        def _adjoint(self,y):
            return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(y), norm='ortho'))

Note that the implement of the Fourier transform can be found in the :mod:`regpy.operators` and the simple version above is a stripped version for square uniform domains.

.. _Non-linear_operators:

Non-linear operators
~~~~~~~~~~~~~~~~~~~~

For non-linear operators, you similarly need to define the domain and codomain when initializing. However, the structure of the evaluation has changed. The evaluation of the non-linear forward model remains and is associated with the `_eval` method. However, to solve the problem we need its linearization thus you have to define the Fréchet derivative and its adjoint.

The core idea in `RegPy` is to force a connection between the evaluation and the linearization. That is, typically you don't just need the derivative as an general method; rather, you require a linearization at a specific point. Therefore, you need both the evaluation :math:`F(x)` and the linear operator :math:`F'[x]`. Thus `RegPy` enforces this connection, by two main implementation choices:

* a derivate is accessed by calling the :meth:`linearize` of the operator which evaluates the operator where it might precompute objects needed for the derivate and then returns both the evaluation and a linear operator that is the derivative
* when ever the operator gets reevaluated at a location the derivate currently connected gets revoked and is not accessible any more

One of the main reasons to force to have such a connection between evaluation and derivate is to prevent the use of a derivative from a different location. Thus if you seriously want to use a derivate of a different location then you have to make a copy of the derivate.

The methods for evaluation, derivative and adjoint
--------------------------------------------------

For the explained above structure of a non-linear operator, you need to implement the following methods:

* `_eval`: his method computes the evaluation of the forward operator. It must also take two extra arguments, `derivative` and `adjoint_derivative`, which are booleans. These arguments determine whether you want to compute the derivative and/or the composition of the adjoint and the derivative. More details below in :ref:`eval_nonlinear`
* `_derivative`:  This method computes the derivative of the forward operator at a specific location. The location is not an argument of the method. One has to construct a full Frechèt derivative operator using the `linearize` of the operator. The returned derivative then uses this method as evaluation.
* `_adjoint`: This method computes the adjoint of the derivative of the forward operator.

.. _eval_nonlinear:

The _eval method
^^^^^^^^^^^^^^^^

Now the core principle of the evaluation method has not changed compared to a linear operator. The only additional requirement is to incorporated the optional boolean arguments `derivative` and `adjoint_derivative`. The second argument is only interesting to you if you want to implement a combined evaluation of the adjoint and derivative (More details in :ref:`Adjoint_Derivative`).

As already pointed out above and addressed in more detail later in :ref:`Linearization_Method` we want to associate the derivate with a evaluation to get a full linearization. However, the evaluation of the derivative at a specific location depends on the point at which we evalute and maybe we can precompute certain objects that are later required when evaluation the derivative or its adjoint. These computations might take some time thus we do not want to precompute every time that we evaluate. Hence, we can use the optional argument `derivative` which is only true if we want to compute the derivative. Thus we may put into the evaluation method anything that we need to precompute to by putting it behind an if statement. Thus we have the structure

.. code-block:: python

    def _eval(self,x, derivative = False, adjoint_derivative = False):
            # Compute with x being in the my_domain what the operator evaluates as y=Tx
            if derivate:
                self.x = x # Storing the location at which the linearization takes place
                # make necessary precomutatoins for derivative at x
            return y

In this general structure we store the point at which we computed the derivate as the attribute `x` of the operator. This attributes are then passed to the derivate.

The _derivative and _adjoint method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now assuming you have already made all the precomutatoins such that you would be able to define the linear operator :math:`F'[x]`. Now if you recall (:ref:`Linear_Operator`) a linear operator only need to define what its evaluation and its adjoint are. So now you can think of :meth:`_derivative` as the :meth:`_eval` of the linear operator :math:`F'[x]` and the :meth:`_adjoint` is now the adjoint of this linear operator :math:`F'[x]^\ast`. Moreover, this is exactly how `RegPy` treats these methods. The derivate is itself just a linear operator, which is particularly linked to its full non-linear operator using the methods and attributes that are associated with it. In particular, the full non-linear operator instance can be revoked by this its derivative in case it gets reevaluated at a different location.

.. _Linearization_Method:

What happens when you linearize
-------------------------------

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

Example
-------

Let us discuss as an example the simple point-wise squared modulus operator :math:`f\mapsto |f|^2` as found in :class:`regpy.operators.SquaredModulus`. The operators initialization just takes some domain and defines the codomain the real vector space of that domain.

Now we can recall that the derivate at a point :math:`f` is given by :math:`h\mapsto 2\Re(\overline{f}\cdot h)`. Thus in the precompute of the evaluation we should store the point-wise factor that is the function :math:`x` multiplied with the factor 2. Hence we obtain the evaluation method

.. code-block:: python

    def _eval(self, x, differentiate=False):
        if differentiate:
            self._factor = 2 * x
        return x.real**2 + x.imag**2

The method :meth:`_derivative` is now the method that handles the evaluation of the derivative. That is, given an input :math:`h` in the domain the output is given by :math:`2\Re(\overline{f}\cdot h)`. Since we already stored the factor as an attribute we can simply define this method by

.. code-block:: python

    def _derivative(self, h):
        return (self._factor.conj() * h).real

It remains to define the adjoint of the derivative which in this simple example is defined by :math:`y\mapsto 2x\cdot y`. Thus we can use the precomputed factors conjugate as the point-wise multiplier and define the adjoint by

.. code-block:: python

    def _adjoint(self, y):
        return self._factor.conj() * y

Combining the above methods we obtain the operator for point-wise squared modules as found in the module :mod:`regpy.operators`

.. code-block:: python

    class SquaredModulus(Operator):

        def __init__(self, domain):
            if domain:
                codomain = domain.real_space()
            else:
                codomain = None
            super().__init__(domain, codomain)

        def _eval(self, x, differentiate=False):
            if differentiate:
                self._factor = 2 * x
            return x.real**2 + x.imag**2

        def _derivative(self, h):
            return (self._factor.conj() * h).real

        def _adjoint(self, y):
            return self._factor * y

Further examples
^^^^^^^^^^^^^^^^

As an easy example you might want to checkout the Volterra problem in :ref:`/notebooks/volterra_main_example.ipynb` for the exponent not equal one we have a non-linear operator.


`ngsolve` Operators
~~~~~~~~~~~~~~~~~~~

`RegPy` offers an interface for `ngsolve` to allow for implementations of your favourite inverse problem with a nice pde solver and use `RegPy` to regularize. For this the library provides 3 extra modules in the operators `regpy.operators.ngsolve`, the Hilbert spaces `regpy.hilbert.ngsolve` and for functionals `regpy.functionals.ngsolve`.

Note that even for these type of operators `RegPy` requires you to implement the adjoint with respect to the standard scalar product!

If you want to implement your own operator you should use the base class :class:`NgsOperator`. With this you have already some basic methods to deal with the `numpy` interface in `regpy` and the `ngsolve` interface.

.. caution::
    This Interface will be changed

For many problems that are based upon a scalar parameter identification problem using a second order elliptic linear pde we have implemented an operator base class :class:`SecondOrderEllipticCoefficientPDE`. You can use this class to define your own operator and what remains to do is to implement you bilinear from and linear form to define such an operator. For an example checkout diffusion problem in :ref:`/notebooks/diffusion_coefficient.ipynb`.

.. _Adjoint_Derivative:

Commbined adjoint and derivative
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now in some use cases it might be beneficial to not construct many object in the image space. Thus we may not want to evalute the operator (or derivate) and then apply its adjoint on the output but rather use a simplefied and less memory consuming implementation for this concatenation. In such a case one can use the `_adjoint_derivative` method and then calling linearize with the additional flag `adjoint_derivative` to get a third output which is a linear operator for the concatenation.

.. warning::
    Note that this simplefication is currently under further development and there are currently no solvers that rely on this reduction!
