===================
Features of `regpy`
===================

`regpy` is a library to solve inverse and ill-posed problems using regularisation methods. That is for a given forward operator

.. math::
    F\colon X\to Y

the goal is to find solution for the problem 

.. math::
    F(f) = g

given some obervation data :math:`g^{obs}=g+\eta`
This would reauire the inverse :math:`F^{-1}`, which is for most cases not continuous. Hence small pertubartions by noise :math:`\eta` can make the inverse unstable. Such problems occure in many applications in imaging mathods in physics, biology, medecine and more. For examples checkout the :ref:`examples`.

`regpy` can be divided into 3 parts:
 * modelling the forward operator
 * modelling space structure, data-fidelity and regularisation functional
 * regularisation solvers


----------------
Forward Operator
----------------

For modelling forward operator you require two subpackges `regpy.operators` for the operators and `regpy.vecsps` for the vector space structure. 

"""""""""""""
vector spaces
"""""""""""""
The spaces :math:`X` and :math:`Y` have to be discretised as discrete vector spaces, which are provided in this subpackges. This discretisation provides the base for the spaces that are used by `regpy.operators.Operator`. The base class is `VectorSpace`,
which represents plain numpy arrays of some shape and dtype. Currently the vectors have to be `numpy.ndarrays` which for example for the `ngsolve` extesion causes a lot of conversion. 

At the moment whis will be amended by some generalized duck-typing, so that vectors can be any thing as lang as you can somehow define addition and scalar multiplication.

`VectorSpaces` serve the following main purposes:

 * Derived classes can contain additional data like grid coordinates or mesures, bundling metadata in one place instead of having every operator generate linspaces / basis functions / whatever on their own.
 * Providing methods for generating elements of the proper shape and dtype, like zero arrays, random arrays or iterators over a basis.
 * Checking whether a given array is an element of the vector space. This is used for consistency checks, e.g. when evaluating operators. 
 * Checking whether two vector spaces are considered equal. This is used in consistency checks e.g. for operator compositions.

All vector spaces are considered as **real vector spaces**, even if the dtype is complex. This affects iteration over a basis as well as functions returning the dimension or flattening arrays.

More complicated vector spaces can be constructed from others by 

 * adding two sapces `s_1 + s_2` which will be the direct sum :math:`S_1 \oplus S_2`
 * multiplication `s_1 * s_2` which will be the tensor product :math:`S_1 \otimes S_2`
 * powers `s**3` is the direct sum :math:`S\oplus S \oplus S`
  
"""""""""
operators
"""""""""
`regpy.operators` provides the basis for defining forward operators, and implements some simple auxiliary operators. The base class is `Operator`. Further submoduls provide specific operators for different use cases. 

The base class `Operator` for forward operators covers both linear and non-linear operators are handled. Operator instances are callable, calling them with an array argument evaluates the operator. If you wish to implement your own operator their are the following methods that you have to implement:

 * `_eval(self, x, differentiate=False)` for :math:`F(x)`
 * `_derivative(self, x)` for :math:`F'[x_0]x`
 * `_adjoint(self, y)` for :math:`F'[x_0]^\ast y`

These methods are not intended for external use. The idea is that whenever you have an operator non-linear :math:`F\colon X\to Y` you need to be able to 

 * evaluate :math:`F(x)` for :math:`x\in X` and
 * linearize :math:`F(x+h) = F(x)+F'[x]h`.

Since the linearization is always bound to some point :math:`x` and its value :math:`y=F(x)` it is clear that linearizing by `Operator.linearize` will call the operator with the flag `differentiate=True` (used for precomputations for the derivative) and return a linear `Operator` i.e. :math:`F'[x]` that will call `_deivative` for evaluation and `_adjoint` for :math:`F'[x]^\ast`. **Attention** whenever you evaluate an opertor after a linearization it will revoke the derivative automatically! 

Note that for Linear operators only 

```
    _eval(self, x)
    _adjoint(self, y)
```

for its evaluation and adjoint.

**Important** The adjoint should be computed with respect to the standard real inner product

.. math::
    \langle x,y\rangle = \mathrm{Re}(\sum_i x_i * \overline{y_i}).

That is you can think of the implemented adjoint as the conjugate transpose of the the matrix it defines on the choosen discretization. The main idea is, other inner products can be inplemented in `regpy.hilbert` module by their Gram matrices. That makes them completely independent of both vector spaces and operators and makes it possible to switch between them without recomputing and reimplementing the derivative and adjoint. 

One of the main features of `Operator` are the basic operator algebra that is supported:
 * `a * op1 + b * op2` for linear combination :math:`aF + bG` for :math:`F\colon X\to Y`, :math:`G\colon X\to Y` and scalars :math:`a,b`
 * `op1 * op2` composition, i.e. :math:`G\circ F` for :math:`F\colon X\to Y` and :math:`G\colon Y\to Z`
 * `op * arr` composition with array as point wise multiplication in domain, i.e. for :math:`F\colon X\to Y` and :math:`arr\in X` this is :math:`F(arr\cdot x)`
 * `op + arr` operator shifted in codomain, i.e. for :math:`F\colon X\to Y` and :math:`arr\in Y` this is :math:`F(x) + arr`


space structure, data-fidelity and regularisation functional
------------------------------------------------------------

""""""""""""""
Hilbert Spaces
""""""""""""""
Since to modell different Hilbert space structures on vector spaces from `regpy.vecsps` there is the module `regpy.hilbert` implementing different Hilbert space structures. The only thing that is required is the Gram operator as a linear `Operator`. In combination with the way the adjoint is required to be defined this gives the possibility to define an adjoint with respect to any of the spaces since 

.. math::
    F^\ast = G^{-1}_X \underline{F}^\ast G_Y

where :math:`\underline{F}^\ast` is conjugate transpose of the forward operator matrix.

There are so called `AbstractSpaces` which are only wrappers that when called on a specific space choose the correct implementation. So that you dont have to know the exact class name but rather can call 

```
    L2(domain)
```

for example to dreate an `L2` sapce on whatever the domain was.

Constructing Hilbert spaces on direct sums can be made easily by defining each Hilbert sapce and then adding them. 

"""""""""""
Functionals
"""""""""""
As a second part of structures exist functionals in `regpy.functionals` for `regpy.vecsps`. `Functional` is the base class for implementation of convex functionals. The evaluation of a specific functional on some element of the `domain` can be done by
simply calling the functional on that element. That is the minal requirement for a functional to work. For a given convex functional :math:`F\colon X \to \mathbb{R}\cup\{\infty\}` the following things might be defined:
 * sugradient :math:`\partial F` - requires `_subgradient`
 * linearization :math:`(F(x),F'[x])` - requires either `_subgradient` or `linearize`
 * hessian :math:`H_F` - requires `_hessian`
 * proximal :math:`\mathrm{prox}_{\tau F}(f)` - requires `_proximal`
 * conjugate :math:`F^\ast(a^\ast) := \mathrm{sup}_x [\langle x^\ast,x\rangle - F(x)]` - requires `_conj`
 * subgradient, linearization, hessian and proximal for canjugate
Note that the proximal is should be defined with respect to the Hilbert space that is associated to that functional by construction. 

The important feature is that you can construct new functionals from others
 * `a * f + b * g` for linear combination :math:`aF + bG` for :math:`F,G\colon X\to \to \mathbb{R}\cup\{\infty\}` and scalars :math:`a,b`
 * `f * op` composition with operators, i.e. :math:`F\circ Op` for :math:`F\colon X\to \to \mathbb{R}\cup\{\infty\}` and :math:`Op\colon Y\to X`
 * `f * a` inner multiplication :math:`F(a \cdot)` for :math:`F\colon X\to \to \mathbb{R}\cup\{\infty\}` and scalars :math:`a` or :math:`a \in X`


regularisation solvers
----------------------

With the structure of operators, Hilbert spaces and functionals the problem of finding a solution to :math:`F(x) = y` given data :math:`y^{obs}`. Then a regularisation method describes a family :math:`R_\alpha\colon Y \to X` and a parameter choice :math:`\hat{\alpha}(g^{obs},\delta)`. Here :math:`\delta` decribes the noise level corresponding with respect to some data fidelity functional :math:`S_{g^{obs}}`. It is a regularization method if for the regularized solution :math:`\hat{x}^{\alpha}:=R_{\hat{\alpha}}(g^{obs})` holds

.. math::
    \Vert x - \hat{x}^{\alpha} \Vert_X \to 0 if \delta \to 0.

Such regularization methods usually iunclude also penalty functional :math:`R\colon X \to \mathbb{R} \cup \{\infty\}`.

To make anything depending on that structure more accessable exists the `regpy.solvers.RegularizationSetting` the bindes the operator with both the data fidelity and penalty functional. In the case that the index is a regularization parameter 

Solvers
"""""""
`regpy.solvers.Solver` is an abstract base class for solvers. Solvers do not implement loops themselves, but are driven by repeatedly calling the `next` method. They expose the current iterate stored in and value as attributes `x` and `y`, and can be iterated over, yielding the `(x, y)` tuple on every iteration (which may or may not be the same arrays as before, modified in-place).

To stop a solver exist methods to run the solver with in `regpy.stoprules.StopRule`. If the solver converged, `converge` should be called, afterwards `_next` will never be called again. Most solvers will probably never converge on their own, but rely on the caller or a `regpy.stoprules.StopRule` for termination.

Stop Rules
""""""""""
Stop rules are used to stop the iteration of an `Solver`. These stoprules are suppossed to choos the :math:`\hat{\alpha}` so that the reconstruction stays stable. 