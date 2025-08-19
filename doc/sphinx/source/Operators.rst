Operators in RegPy
==================

This tutorial provides a more detailed explanation than the general guide in :ref:`/usage.rst#forward-operator` on how to define custom operators using the base class :class:`Operator`.

In `RegPy` an operator represents a (possibly nonlinear) mapping between vector spaces :math:`\mathbb{X}` and :math:`\mathbb{Y}`:

.. math::
    F\colon \mathbb{X}\to\mathbb{Y}.

Here :math:`\mathbb{X}` and :math:`\mathbb{Y}` are instances of the class :class:`VectorSpace`. Examples of such :class:`VectorSpace` s (currently the only examples!) are sets of NumPy arrays of a fixed shape of a floating or complexfloating type with the canonical addition and scalar multiplication. We refer to :ref:`/spaces.rst` for further details. Considering complex vector spaces as real vector space of twice the dimension, we can always think of :math:`\mathbb{X}= \mathbb{R}^N` and :math:`\mathbb{Y}= \mathbb{R}^M`.

A linear operator :math:`T\colon \mathbb{X}\to\mathbb{Y}` can of course always be represented by a matrix in :math:`\underline{T}\in\mathbb{R}^{N\times M}`, but it is often inefficient or impossible to set up this matrix, and all we need is a routine implementing matrix-vector products. Such a routine has to be implemented by in the :code:`_eval` method. In addition we often need matrix-vector products with the transposed matrix  :math:`\underline{T}^{\top}` -- the adjoint with respect to the standard real Euclidean scalar products in :math:`\mathbb{R}^M` and :math:`\mathbb{R}^N`. This should be implemented in the :code:`_adjoint` method.

Alternatively, we can view  :code:`_adjoint` as dual operator :math:`T':\mathbb{Y}'=\mathbb{Y}\to \mathbb{X}'=\mathbb{X}` with the dual pairing given by

.. math::
    \langle u,v\rangle = \text{numpy.vdot(x,y).real},\qquad u\in \mathbb{X}'=\mathbb{X}, \; v\in \mathbb{X}.

Hence the :code:`_eval` and :code:`_adjoint` methods (called by :code:`eval` and :code:`adjoint`) should be implemented such that the following identity is always satisfied:

.. code-block:: python

    numpy.vdot(T.eval(x),y).real == numpy.vdot(x,T.adjoint(y)).real

We point out that if :math:`T` is :math:`\mathbb{C}`-linear, i.e. represented by a matrix :math:`\underline{T}\in\mathbb{C}^{N\times M}`,
then this identity is satisfied if and only if :code:`T.adjoint(.)` is represented by the transposed conjugate matrix of :math:`\underline{T}`. In this case the above identity also holds true without the :code:`.real` parts.

Often an additional Hilbert space structure is introduced in regularization methods for inverse problems. If scalar products on :math:`\mathbb{X}` and :math:`\mathbb{Y}` are represented by the Gram matrices :math:`G_{\mathbb{X}}` and :math:`G_{\mathbb{Y}}`, then the adjoint :math:`T^{\ast}` with respect to these Hilbert space inner products is given by:

.. math::
    T^{\ast} = G_{\mathbb{X}}^{-1} T^{\top} G_{\mathbb{Y}}.

This decomposition motivates the design of operator implementations in `RegPy`: the adjoint of a linear operator is computed assuming the standard scalar product. If a different scalar product is required, it can later be incorporated by assigning the appropriate space structure (via Gram matrices) to the domain and codomain. This separates the choice and implementation of operator representations from the choice and implementation of data fidelity and penalty terms.

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

An operator requires the definition of the vector space structure, meaning you must specify both the `domain` and `codomain` as subclasses of :class:`regpy.vecsps.VectorSpace` (e.g., a space of NumPy arrays of a certain shape). This can be done by passing these values into the initialization method of the class, or by computing them within the class itself. These domains only define the basic vector structure that you want to use in the operator.

The initialization
------------------

Assuming for example you want to define an operator that maps from a uniformly discretized square domain to a uniformly discretized square codomain. Then you could let the operator take tuples `(start,end,number)` for each dimension or some `numpy.linspace` instances to construct the according uniform grid space :class:`regpy.vecsps.UniformGridFcts`. A typical init could look like:

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

For example, for a two dimensional Fourier transform on a centred square uniform grid. That is the domain is assumed to be :class:`regpy.vecsps.UniformGridFcts` that is two dimension for example :code:`domain=UniformGridFcts(d, d, dtype = np.complex128)` and where `d` defines a centred interval for example by :code:`d = (-1,1,100)`. Moreover, from the domain we can construct the codomain as a uniform grid computing the spacing from the spacing in the domain. Thus we obtain an initialization as follows

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

The `_eval` method for a linear operator only takes one mandatory input usually named `x`. The method is only called by the super
method `eval` which it self receives the input when an instances of the class gets called on a particular values. To be sure that
the argument is in the domain the super method `eval` which should not be touched asserts if the argument belongs to the space.
So the method that you have to implement can assume that `x` belongs to the domain which you have specified in the initialization.
Your method is then required to return a value that belongs to the codomain. This property will be asserted in the evaluation to
guarantee that the implementation is returning a valid object which can be treated as an element in the codomain.

Example
^^^^^^^

For the example of the two dimensional Fourier transform we can use the `numpy` FFT implementation and define the evaluation as

.. code-block:: python

    def _eval(self,x):
        return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x), norm='ortho'))

The adjoint evaluation method
-----------------------------

The `_adjoint` method works similarly to the `_eval` method. The operators adjoint is accessed by `my_op.adjoint(y)` to evaluate the adjoint. Corresponding to the evaluation the `adjoint` method which calls your particular implementation asserts first if the argument belongs to the codomain and then if the computed result form your method belongs to the domain. Thus guaranteeing a minimum constancy when using the methods. Most important `RegPy` assumes that the implementation of the adjoint is with respect to the standard real scalar product :math:`\langle x,y\rangle = x^{\top} y`.

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
            # Here you may do some initializing computations depending on your parameters.
            # In particular you have to compute the domain and codomain if they do not
            # have to be supplied as parameters.
            # At the end you have to call the super initialization by:
            super().__init__(
                domain = my_domain,
                #The preimage space (domain of definition) of the operator
                codomain = my_codomain,
                #The image space of the operator
                linear=True
                # has to be set since the default is False
            )

        def _eval(self,x):
            # Compute with x being in the my_domain the image y=Tx of x under the operator T.
            return y

        def _adjoint(self,y):
            # Compute with y being in the my_codomain what the standard
            # adjoint operator evaluates as x=T^{\top}y
            return x

As an easy example you might want to look at the Volterra problem in :ref:`/notebooks/volterra_main_example.ipynb`.
For a more complicated example you may study the `NGSolve` operator in :ref:`/notebooks/tfm.ipynb`.

Example
^^^^^^^

Returning the example of the Fourier transform we can combine the above code snippets to the following class

.. code-block:: python

    import numpy as np
    from regpy.operators import Operator
    from regpy.vecsps import UniformGridFcts

    class SimpleFFTOnSquare(Operator):
        def __init__(self,d):
            domain = UniformGridFcts(d,d,dtype = complex)
            # Compute dual grid arising if FFT is used as an approximation of the continuous Fourier transform
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

This example is a simplified version of the Fourier transform implementated in the :mod:`regpy.operators`.

.. _Non-linear_operators:

Non-linear operators
~~~~~~~~~~~~~~~~~~~~

For non-linear operators, you similarly need to define the domain and codomain when initializing. However, the structure of the evaluation has changed. The evaluation of the non-linear forward model remains and is associated with the `_eval` method. However, to approximately solve the associated operator equation, we typically need its linearization, i.e., its Fréchet derivative and the adjoint of the Fréchet derivative.

A core idea in the design of nonlinear operators in `RegPy` is to enforce a connection between the evaluation and the linearization. The reason is that you typically need evaluations of the operator and its derivative at the same points, and often evaluations of directional derivatives at the same point in many directions. The operations often share many computations that need to be done only once. Therefore, the linear operator :math:`F'[x]` is optionally provided together with the evaluation function :math:`F(x)`. More specifically, `RegPy` enforces this connection, by two main implementation choices:

* The derivate is accessed by calling the :meth:`linearize` of the operator which evaluates the operator where it might precompute objects needed for the derivate and then returns both the evaluation and the derivative as a linear operator.
* Whenever the operator gets re-evaluated at another point, the derivate at the old evalution point is revoked and is then not accessible any more.

The main reason to enforce such a connection between evaluation and derivate is to prevent simultaneous use of derivatives at different points. This would require simultaneous storage of the precomputations associated to the evaluations at these different points, which is not need in most cases. If you really need to use derivates at different points simultaneously,  then you have to make a copy of the derivate.

The methods for evaluation, derivative and adjoint
--------------------------------------------------

For the structure of a non-linear operator explained above, you need to implement the following methods:

* `_eval`: Given :math:`x`, this method computes :math:`F(x)`, i.e. it evaluates the forward operator. It must also accept two extra optional boolean arguments, `derivative` and `adjoint_derivative`. These arguments determine whether you want to compute the derivative and/or the composition of the adjoint and the derivative. More details below in :ref:`eval_nonlinear`
* `_derivative`:  This method computes  :math:`F'[x]h` given :math:`h`, i.e.  the derivative of the forward operator in direction :math:`h` . The point :math:`x` is not an argument of the method, and users should not call this method directly. They rather first call  the `linearize` method of the operator with argument `x`, which in turn calls :code:`_eval` with argument `x` and `linearize=true` to obtain a (virtual) Jacobian :math:`F'[x]`. If this virtual Jacobian is evaluated, it will call this method.
* `_adjoint`: This method computes the adjoint of the derivative of the forward operator, i.e., :math:`F'[x]^{\top}y`. Again, :math:`x` is not an argument of this method, but it will be called by the virtual Jacobian :math:`F'[x]` if the adjoint of the Jacobian is called by the user.

.. _eval_nonlinear:

The _eval method
^^^^^^^^^^^^^^^^

Now the core principle of the evaluation method has not changed compared to a linear operator. The only additional requirement is to incorporated the optional boolean arguments `derivative` and `adjoint_derivative`. The second argument is only interesting to you if you want to implement a combined evaluation of the adjoint and derivative (More details in :ref:`Adjoint_Derivative`).

As already pointed out above and addressed in more detail later in :ref:`Linearization_Method` we want to associate the derivate with a evaluation to get a full linearization. However, the evaluation of the derivative at a specific location depends on the point at which we evalute and maybe we can precompute certain objects that are later required when evaluation the derivative or its adjoint. These computations might take some time thus we do not want to precompute every time that we evaluate. Hence, we can use the optional argument `derivative` which is only true if we want to compute the derivative. Thus we may put into the evaluation method anything that we need to precompute to by putting it behind an if statement. Thus we have the structure

.. code-block:: python

    def _eval(self,x, derivative = False, adjoint_derivative = False):
            # Compute with x being in the my_domain what the operator evaluates as y=Tx
            if derivative:
                self.x = x # Storing the location at which the linearization takes place
                # make necessary precomputations for derivative at x
            return y

In this general structure we store the point at which we computed the derivate as the attribute `x` of the operator. This attributes are then passed to the derivate.

The _derivative and _adjoint method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now assuming you have already made all the precomputations such that you would be able to define the linear operator :math:`F'[x]`. Now if you recall (:ref:`Linear_Operator`) a linear operator only need to define what its evaluation and its adjoint are. So now you can think of :meth:`_derivative` as the :meth:`_eval` of the linear operator :math:`F'[x]` and the :meth:`_adjoint` is now the adjoint of this linear operator :math:`F'[x]^\ast`. Moreover, this is exactly how `RegPy` treats these methods. The derivate is itself just a linear operator, which is particularly linked to its full non-linear operator using the methods and attributes that are associated with it. In particular, the full non-linear operator instance can be revoked by this its derivative in case it gets reevaluated at a different location.

.. _Linearization_Method:

What happens when you linearize
-------------------------------

Thus, when `RegPy` linearizes an operator by calling its `linearize` method at a point :math:`x`, the following steps occur:

1. The operator is evaluated, and the optional argument `derivative=True` is passed, so that the `_eval` method knows a derivative is required.
2. In this case the operator prepares the linearization by storing as attributes any intermediate quantities which arise in the evaluation of the operator and which are needed again for the evaluation of the derivative.
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
                domain = my_domain, #The preimage space (or domain) X of the operator
                codomain = my_codomain, #The the image space of the operator
                linear=False # can also be left since the default is False
            )

        def _eval(self,x, derivative = False, adjoint_derivative = False):
            # Compute with x being in the my_domain the image y=F(x) of x under F
            if derivative:
                self.x = x # Storing the point at which the operator is linearized
                # make necessary precomputations for derivative at x
            if adjoint_derivative:
                # make necessary precomputations for the composition of adjoint and derivative
            return y

        def _derivative(self,h):
            # compute for h in the my_domain the derivative y = F'[self.x](h) at the point self.x saved by _eval
            return y

        def _adjoint(self,y):
            # Compute with y being in the my_codomain what the standard adjoint of the derivative x = F'[self.x]*(y) at the predefined location self.x
            return x


Example
-------

Let us discuss as an example the simple observation operator associated with phase retrieval given by the composition of the point-wise squared modulus operator and the Fourier transform :math:`x\mapsto |\mathcal{F}(x)|^2`. The operators initialization just takes some uniform centred domain and defines the codomain the real vector space of the corresponding Fourier space.

.. code-block:: python

    def __init__(self,domain):
        # Compute dual grid arising if FFT is used as an approximation of the continuous Fourier transform
        self.cd = (-1/2/domain.spacing[0],1/2/domain.spacing[0],domain.shape[0])
        codomain = UniformGridFcts(cd,cd)
        super().__init__(
            domain = domain,
            codomain = codomain,
            linear = True
        )

Recall that the derivate of the squared modulus :math:`Sq\colon x\mapsto |x|^2` at a point :math:`f` is given by :math:`h\mapsto 2\Re(\overline{f}\cdot h)`. Moreover, as the Fourier transform is linear the derivative is given by itself that is :math:`\mathcal{F}'[x](h)=\mathcal{F}(h)`. Hence using the chain rule for the Fréchet derivative that is :math:`(G\circ F)'[f]h=G'[F(f)]F'[f]h` we obtain

.. math::
    (Sq\circ\mathcal{F})'[f]h=Sq'[\mathcal{F}(f)]\mathcal{F}'[f]h = Sq'[\mathcal{F}(f)]\mathcal{F}(h).

Moreover, for the adjoint we obtain

.. math::
    (Sq\circ\mathcal{F})'[f]^\ast y= \mathcal{F}^\ast Sq'[\mathcal{F}(f)]^\ast y.


Thus both for the derivative and its adjoint we need :math:`\mathcal{F}(f)` the Fourier transform of the point :math:`f` at which we linearize. Hence, when we want to linearize and execute :code:`_eval` with the option code:`differentiate=True`, indicating that we want to differentiate at :math:`x`, we should store :math:`\mathcal{F}(f)`. Hence we obtain the evaluation method

.. code-block:: python

    def _eval(self, x, differentiate=False):
        y = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x), norm='ortho'))
        if differentiate:
            self._factor = y
        return y.real**2 + y.imag**2

The method :meth:`_derivative` implements the :math:`(Sq\circ\mathcal{F})'[f]h` for any :math:`h` as a method that handles the evaluation of the derivative. That is, given an input :math:`h` in the domain the output is given by :math:`2\Re(\overline{\mathcal{F}(f)}\cdot h)`. Since we already stored the factor :math:`\mathcal{F}(f)` as an attribute we can simply define this method by

.. code-block:: python

    def _derivative(self, h):
        return 2*(self._factor.conj() * np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(h), norm='ortho'))).real

It remains to define the adjoint of the derivative which as stated above is defined by :math:`y\mapsto \mathcal{F}^\ast(2\mathcal{F}(f)y)`. Thus we can use the precomputed factor as the point-wise multiplier and define the adjoint by

.. code-block:: python

    def _adjoint(self, y):
        return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(2*self._factor * y), norm='ortho'))

Combining the above methods we obtain the observation operator appearing in Phase retrieval problems by

.. code-block:: python

    import numpy as np
    from regpy.vecsps import UniformGridFcts
    from regpy.operators import Operator

    class Observation(Operator):

        def __init__(self,domain):
            # Compute dual grid arising if FFT is used as an approximation of the continuous Fourier transform
            self.cd = (-1/2/domain.spacing[0],1/2/domain.spacing[0],domain.shape[0])
            codomain = UniformGridFcts(cd,cd)
            super().__init__(
                domain = domain,
                codomain = codomain,
                linear = False
            )

        def _eval(self, x, differentiate=False):
            y = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x), norm='ortho'))
            if differentiate:
                self._factor = y
            return y.real**2 + y.imag**2

        def _derivative(self, h):
            return 2*(self._factor.conj() * np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(h), norm='ortho'))).real

        def _adjoint(self, y):
            return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(2*self._factor * y), norm='ortho'))

The great benefit of RegPy is that you are not require to implement the composition and compute the derivate your self. Rather you can implement each operator by its own and then combine them by composing them. Hence for this example the same operator is implemented with

.. code-block:: python

    from regpy.operators import SquaredModulus, FourierTransform
    from regpy.vecsps import UniformGridFcts

    domain = UniformGridFcts((-1,1,100),(-1,1,100), dtype = complex)

    ft = FourierTransform(domain,centered=True)
    sqm = SquaredModulus(ft.codomain)

    observe = sqm * ft


Further examples
^^^^^^^^^^^^^^^^

As an easy example you might want to have a look at the Volterra problem in :ref:`/notebooks/volterra_main_example.ipynb`.For the exponent not equal one we have a non-linear operator.


`ngsolve` Operators
~~~~~~~~~~~~~~~~~~~

`RegPy` offers an interface for `ngsolve` to allow for implementations of your favourite inverse problem with a nice pde solver and use `RegPy` to regularize. For this the library provides 3 extra modules in the operators `regpy.operators.ngsolve`, the Hilbert spaces `regpy.hilbert.ngsolve` and for functionals `regpy.functionals.ngsolve`.

Note that even for these type of operators `RegPy` requires you to implement the adjoint with respect to the standard scalar product!

If you want to implement your own operator you should use the base class :class:`NgsOperator`. With this you have already some basic methods to deal with the `numpy` interface in `RegPy` and the `ngsolve` interface.

.. caution::
    This Interface will be changed

For many problems that are based upon a scalar parameter identification problem using a second order elliptic linear pde we have implemented an operator base class :class:`SecondOrderEllipticCoefficientPDE`. You can use this class to define your own operator and what remains to do is to implement you bilinear from and linear form to define such an operator. For an example checkout diffusion problem in :ref:`/notebooks/diffusion_coefficient.ipynb`.

.. _Adjoint_Derivative:

Combined adjoint and derivative
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For some operators more efficient implementations of the composition of adjoint and derivative than the straightforward one exist.
E.g., for inverse problems with correlation data the codomain of the operator is often so large that elements of this space would not fit into memory.
In this case one can redefine the method `_adjoint_derivative` of the operator as follows:

.. code-block:: python

    def _adjoint_derivative(self,x):
    # compute for x in the my_domain the composition of derivative and its adjoint x = F'[self.x]*F'[self.x](x) at the predefined location self.x
    return x

.. warning::
    Note that this simplification is currently under further development and the released branch currently contains no solvers that rely on this reduction!
