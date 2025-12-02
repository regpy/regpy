# Usage of `RegPy`

`RegPy` is a library to solve inverse and ill-posed problems using regularisation methods. That is for a given forward operator

$$
    F\colon X\to Y
$$

the goal is to find the solution for the problem

$$
    F(f) = g
$$

given some observation data $g^{obs}=g+\eta$. This would require the inverse $F^{-1}$, which in most cases is not continuous. Hence small perturbations by noise $\eta$ can make the inverse unstable. Such problems occur in many applications in imaging methods in physics, biology, medicine and more. For examples checkout the [examples](https://num.math.uni-goettingen.de/regpy/examples.html).

`RegPy` can be divided into three parts:

* modelling the forward operator: with `regpy.operators` and `regpy.vecsps`
* modelling space structure, data-fidelity and regularisation functional: with `regpy.hilbert` and `regpy.functionals`
* regularisation solvers: with `regpy.solvers` and `regpy.stoprules`

---

## Forward operator

For modelling forward operator we need two subpackges `regpy.operators` for the operator structure and `regpy.vecsps` for the vector space structure.

### Vector Spaces

The spaces $X$ and $Y$ have to be given as discrete vector spaces as provided in `regpy.vecsps`. The base class for any vector space is `VectorSpace`, which represents any vector as plain numpy array of some shape and dtype. The vectors have to be `numpy.ndarrays`, which for example in the `ngsolve` extesion is done by conversion.

`VectorSpaces` serve the following main purposes:

* Providing methods for generating elements of the proper shape and dtype, like:
  * zero arrays `VectorSpaces.zeros()`
  * random arrays `VectorSpaces.rand()`
  * iterators over the basis
* consistency checks
  * providing a control routine whether a given element is an element of the vector space: simply used by `x in VectorSpaces`
  * test of equality of two vector spaces
* Derived classes can contain additional data like grid coordinates or measures, bundling metadata in one place.

All vector spaces are considered as **real vector spaces**, even if the dtype is complex.

More complicated vector spaces can be constructed from others by

* adding two spaces `s_1 + s_2` which will be the direct sum $S_1 \oplus S_2$
* multiplication `s_1 * s_2` which will be the tensor product $S_1 \otimes S_2$
* powers `s**3` is the direct sum $S\oplus S \oplus S$
  
### Operators

`regpy.operators` provides the basis for defining forward operators, and implements some simple auxiliary operators. The base class for any operator is `Operator`. Furthermore, `regpy.operators` provides many basic operators which can be combined by operator operations combining basic operators.

The base class `Operator` for forward operators provides a frame work for both linear and non-linear operators. Operator instances are callables, calling them with an array argument evaluates the operator at this position. If you wish to implement your own operator their are the following methods that you have to implement:

* `_eval(self, x, differentiate=False)` for :math:`F(x)`
* `_derivative(self, x)` for $F'[x]h$
* `_adjoint(self, y)` for $F'[x]^\ast y$

These methods are not intended for external use. The idea is that whenever you have an operator $F\colon X\to Y$ you need to be able to

* evaluate $F(x)$ for $x\in X$ and
* linearize $F(x+h) = F(x)+F'[x]h$.

Since the linearization is always bound to some point $x$ and its value $y=F(x)$ the implementation binds the linearization to an evaluation. That is linearizing by `Operator.linearize` will call the operator and evaluate `_eval` with the flag `differentiate=True` (to separate precomputations for the derivative) and returns a linear `Operator` i.e. $F'[x]$ that will call `_derivative` for evaluation and `_adjoint` for $F'[x]^\ast$. **Attention:** Since the linearization is bound to the location the implementation prevents the use of the derivate after a reevaluation of the operator. That is whenever you evaluate an operator after a linearization it will revoke the derivative automatically!

#### Minimal example of non-linear operator

```python
from regpy.operators import Operator
class op_name(Operator):
    def __init__(self,arg_1,arg_2):
        .....
        super.__init__(domain=dom,codomain=codom,linear=False)
    
    def _eval(self,x,differentiate=false):
        # computations for y=F(x)
        if differentiate:
            # precomputations for derivative at x 
            # (e.g., store x as an attribute)
        return y
    
    def _derivative(self,h):
        # evaluation of the y=F[x](h) using precomputations by _eval
        return y
    
    def _adjoint(self,y):
        # evaluation of the v=F[x]*(y) using precomputations by _eval
        return v
```

#### Minimal example of linear operator

```python
from regpy.operators import Operator
class op_name(Operator):
    def __init__(self,arg_1,arg_2):
        # ...
        super.__init__(domain=dom,codomain=codom,linear=True)
    
    def _eval(self,x,differentiate=false):
        # computations for y=F(x) 
        if differentiate:
            # precomputations for derivative
        return y
        
    def _adjoint(self,y):
        # evaluation of the x=F*(y) 
        return x
```

Note that the linear operators only requires

```python
    _eval(self, x)
    _adjoint(self, y)
```

for its evaluation and its adjoint.

#### Adjoint of an operator

The adjoint should be computed with respect to the standard real inner product

$$
    \langle x,y\rangle = \mathrm{Re}(\sum_i x_i \overline{y_i}).
$$

That is you can think of the implemented adjoint as the conjugate transpose of the the matrix representing the linear mapping.
The motivation of this implementation is that other inner products can be added later by applying specific Gram matrices implemented in `regpy.hilbert` module. Thus the operators (derivatives) adjoint implementation is independent of the inner product structure on the vector spaces and makes it possible to switch between them without recomputing and reimplementing the derivative and adjoint.

#### Operator operations

One of the main features of `Operator` are the basic operator algebra that is supported:

* `a * op1 + b * op2` for linear combination $aF + bG$ for $F\colon X\to Y$, $G\colon X\to Y$ and scalars $a,b$
* `op1 * op2` composition, i.e. $G\circ F$ for $F\colon X\to Y$ and $G\colon Y\to Z$
* `op * arr` composition with array as point wise multiplication in domain, i.e. for $F\colon X\to Y$ and $arr\in X$ this is $F(arr\cdot x)$
* `op + arr` operator shifted in codomain, i.e. for $F\colon X\to Y$ and $arr\in Y$ this is $F(x) + arr$

---

## Space structure, data-fidelity and regularisation functional

### Hilbert spaces

The module `regpy.hilbert` models different Hilbert space structures on vector spaces from `regpy.vecsps.VectorSpaces`. This is done by introducing the Gram operator as a linear `regpy.operators.Operator`. Recall that any operator defined on some `VectorSpaces` defines the adjoint with respect to the standard scalar product. Thus, in combination with the Gram operator it is possible to define an adjoint with respect to any inner product as

$$
    F^\ast = G^{-1}_X \underline{F}^\ast G_Y
$$

where $\underline{F}^\ast$ is conjugate transpose of the forward operator matrix.

As an example let us construct an $ L^2$ inner product space on a uniform grid:

```python
from regpy.vecsps import UniformGridFcts
from regpy.hilbert import L2UniformGridFcts 

domain = UniformGridFcts((-1,1,100))
h_domain = L2UniformGridFcts(domain)
```

#### Abstract class

The `AbstractSpaces` give implicit structure without the explicit implementation. They are callable objects which when called on a specific domain type choose the correct implementation. Thus one does not have to know the exact class name but rather can call on the domain to get a specific implementation:

```python
import numpy as np
from regpy.vecsps import UniformGridFcts,MeasureSpaceFcts
from regpy.hilbert import L2

uniform_domain = UniformGridFcts((-1,1,100))
h_uniform_domain = L2(uniform_domain) #uses regpy.hilbert.L2UniformGridFcts 
measure_domain = MeasureSpaceFcts(np.random.rand(100))
h_measure_domain = L2(measure_domain) #uses regpy.hilbert.L2MeasureSpaceFcts 
```

In this example we only needed to introduce `L2` as the structure we wanted and can call it on both domains to get the explicit implementation.

#### Combinations of Hilbert Spaces

Constructing Hilbert spaces for example on direct sums can be done easily by defining each Hilbert space and then adding them.

```python
import numpy as np
from regpy.vecsps import UniformGridFcts,MeasureSpaceFcts
from regpy.hilbert import L2,H1

uniform_domain = UniformGridFcts((-1,1,100))
measure_domain = MeasureSpaceFcts(np.random.rand(100))
h_sum = L2(uniform_domain)+H1(measure_domain)
```

### Functionals

As a second part of structures ond `VectorSpaces` one can define functionals from `regpy.functionals`. `Functional` is the base class for the explicit implementation of convex functionals. Every functional provides the possibility for evaluation on some element of its `Functional.domain` that is a `VectorSpace`. Any functional has to provide at least such an evaluation. For a given convex functional $F\colon X \to \mathbb{R}\cup\{\infty\}$ the following methods can be defined:

* sugradient $\partial F$ - implemented as `Functional._subgradient` and called using `Functional.subgradient`
* linearization $(F(x),F'[x])$ - requires either `Functional._subgradient` or `Functional._linearize`
* hessian $H_F$ - requires `_hessian`
* proximal $\mathrm{prox}_{\tau F}(f)$ - requires `_proximal`
* conjugate $F^\ast(a^\ast) := \mathrm{sup}_x [\langle x^\ast,x\rangle - F(x)]$ - requires `_conj`
* subgradient, linearization, hessian and proximal for canjugate

Note that the proximal has to be defined with respect to the Hilbert space that is associated to that functional by construction. That is each functional has an assigned Hilbert space.

One of the important feature is that new functionals can be constructed by natively defined combinations from others

* `a * f + b * g` for linear combination $aF + bG$ for $F,G\colon X\to \to \mathbb{R}\cup\{\infty\}$ and scalars $a,b$
* `f * op` composition with operators, i.e. $F\circ Op$ for $F\colon X\to \to \mathbb{R}\cup\{\infty\}$ and $Op\colon Y\to X$
* `f * a` inner multiplication $F(a \cdot)$ for $F\colon X\to \to \mathbb{R}\cup\{\infty\}$ and scalar $a$ or $a \in X$

## Regularisation and Solvers

Using the structure of operators, Hilbert spaces and functionals the problem of finding a solution to $F(x) = y$ given data $y^{obs}$ it remains to introduce how to regularize using solvers. Recall a regularisation method describes a family $R_\alpha\colon Y \to X$ and a parameter choice $\hat{\alpha}(g^{obs},\delta)$. Here $\delta$ describes the noise level corresponding with respect to some data fidelity functional $S_{g^{obs}}$. It is a regularization method if for the regularized solution $\hat{x}^{\alpha}:=R_{\hat{\alpha}}(g^{obs})$ holds

$$
    \Vert x - \hat{x}^{\alpha} \Vert_X \to 0 \quad if\quad  \delta \to 0.
$$

Such regularization methods usually include also penalty functional $R\colon X \to \mathbb{R} \cup \{\infty\}$ which models the domain structure.

To bind all of the structure of your specific regularization setting together the library uses `regpy.solvers.Setting`. Objects of this class provide an easy access for any solver to access the operator and both the data fidelity and penalty functional.

```python
from regpy.vecsps import UniformGridFcts, GridGcts

from my_op import My_Op # implementation of my own operator

domain = UniformGridFcts((-1,1,100))
codomain = UniformGridFcts((-10,10,1000))+GridFcts(200)

op = My_Op(domain,codomain)

from regpy.hilbert import L2,H1
from regpy.functionals import L1

from regpy.solvers import Setting

setting = Setting(
    op = op, # the operator
    penalty = L1, # Using the abstract Functional L1 as penalty in the domain
    data_fid = L2+H1, # Using the abstract direct sum of Hilbert space L2 and H1 on the direct sum codomain of UniformGridFcts and GridFcts
)
```

The setting in particular provides some test like testing the adjoint with `setting.test_adjoint` that throws an assertion if the implementation of the adjoint is not of an accuracy of `1e-10`. Note that the adjoint is not tested with respect to the inner products but only of the operator with respect to the standard inner product. As a second test one can test the implementation of the adjoint using `setting.test_adjoint` that tests for a random decreasing vector $x$ and $v$ in the domain if $||\frac{F(x+tv)-F(x)}{t}-F'(x)v||$ is a decreasing sequence for a decreasing sequence of $t$. Note that the norm here is also the standard squared modulus.

### Solvers

We separate solvers in two submodules `regpy.solvers.linear` linear solvers and `regpy.solvers.nonlinear` non-linear solvers. Solvers in in the non-linear submodule will work for linear forward models as well, however might not be as fast as the solvers suited for the linear models.

```python
from regpy.solvers.linear import TikhonovCG

solver = TikhonovCG(
    setting, # A Regularization setting 
    data = y_obs, # The data, optional if not given the library tries to extract from the data fidelity
    regpar = 0.1, # The regularization parameter, if the setting is a Tikhonov setting then the regularisation parameter defined there is used
)
```

Solvers are implemented as subclasses of the abstract base class `regpy.solvers.Solver` or its subclass`regpy.solvers.RegSolver` which requires a regularization setting. Solvers do not implement loops themselves, but are driven by repeatedly calling the `next` method. They expose the current iterates stored as attributes `x` and `y`, and can be iterated over, yielding the `(x, y)` tuple on every iteration (which may or may not be the same arrays as before, modified in-place).

```python
for x,y in solver:
    ...
    # do something with the iteration
```

This runs the solver until it converges. Note, that it has no stopping criterion. To stop with depending on the iteration number or the current iterates `RegPy` supplies stopping rules in the module `regpy.stoprules`. A stopping rule can be used in different ways in connection with a solver. Note that once a solvers converged or a stopping rule triggered it has to be reinitiated to restart.

```python
from regpy.stoprule import CountIterations

for x,y in solver.while(CountIterations(500)):
    ...
    # do something with the iteration while the iteration counter is below 500
```

```python
from regpy.stoprule import Discrepancy,CountIterations

stoprule = CountIterations(500) + Discrepancy(
    norm = setting.hcodomain.norm, 
    data = y_obs, 
    noiselevel = noise,
    tau = 2.5
)

for x,y in solver.until(stoprule):
    ...
    # do something with the iteration until the iteration counter is 500 or stop early if relative discrepancy is below 2.5
```

The previous methods assumed one need the iterates. However, to simply run the solver until it stops and get the final iterate one can use the `Solver.run` method:

```python
from regpy.stoprule import Discrepancy,CountIterations

stoprule = CountIterations(500) + Discrepancy(
    norm = setting.hcodomain.norm, 
    data = y_obs, 
    noiselevel = noise,
    tau = 2.5
)

x,y = solver.run(stoprule)
```

For subclasses of `RegSolver` like the `TikhonovCG` there exists a convenience method to run the solver with the disrcrepancy principle by:

```python
x,y = solver.runWithDP(
    data = y_obs,
    delta = noise,
    tau = 2.5
)
```

### Stopping Rules

As seen above stopping rules are used to stop the iteration of an `Solver`. These stoprules are suppossed to choose the $\hat{\alpha}$ so that the reconstruction stays stable. Stopping rules can be combined with standard summation of two stoping prules as illustrated above. Note that a stopping rule has be reinitated for any other running solver.
