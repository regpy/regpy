from functools import wraps
from logging import getLogger, INFO, StreamHandler, Formatter, WARNING

import numpy as np


_default_rng = np.random.default_rng()
"""The default Random Generator to be used throughout RegPy
    """

def get_rng():
    """Return the global Random Generator of RegPy.
    
    Returns
    -------
    Generator
        The random generator.
    """
    return _default_rng

def set_rng_seed(seed : None | int | np.random.SeedSequence | np.random.BitGenerator | np.random.Generator | np.random.RandomState = None):
    """
    Reset the global RNG seed using the provided seed for RegPy.

    Parameters
    ----------
    seed : {None, int, array_like[ints], SeedSequence, BitGenerator, Generator, RandomState}, optional
        The random seed to be used by the `numpy.random.default_rng` to construct the random generator used 
        to generate pseudo random vectors. For possible details how the argument is handled we refer to the 
        numpy documentation.
    """
    global _default_rng
    _default_rng = np.random.default_rng(seed)



class Errors:

    @staticmethod
    def _compose_message(title : str, content : str):
        return f"""
-------------------------------------------------------
    RegPy Error - {title}
{content}
-------------------------------------------------------"""

    @staticmethod
    def generic_message(msg : str):
        return Errors._compose_message(title = "Generic Error", content = msg)
    
    @staticmethod
    def failed_test(msg : str, obj : object | None = None, meth : str | None = None):
        if obj is None:
            if meth is None:
                return Errors._compose_message(title = "Test Failed", content = msg)
            else:
                return Errors._compose_message(title = f"Test of method {meth} failed", content = msg)
        else:
            if meth is None:
                return Errors._compose_message(title = f"Test of {obj} failed", content = msg)
            else:
                return Errors._compose_message(title = f"Test of method {meth} of {obj} failed", content = msg)
    
    @staticmethod
    def value_error(msg : str, obj : object | None = None, meth : str | None = None):
        if obj is None:
            if meth is None:
                return Errors._compose_message(title = "Value Error", content = msg)
            else:
                return Errors._compose_message(title = f"Value Error in method {meth}", content = msg)
        else:
            if meth is None:
                return Errors._compose_message(title = f"Value Error in {obj}", content = msg)
            else:
                return Errors._compose_message(title = f"Value Error in method {meth} of {obj}", content = msg)
    
    @staticmethod
    def type_error(msg : str, obj : object | None = None, meth : str | None = None):
        if obj is None:
            if meth is None:
                return Errors._compose_message(title = "Type Error", content = msg)
            else:
                return Errors._compose_message(title = f"Type Error in method {meth}", content = msg)
        else:
            if meth is None:
                return Errors._compose_message(title = f"Type Error in {obj}", content = msg)
            else:
                return Errors._compose_message(title = f"Type Error in method {meth} of {obj}", content = msg)
    
    @staticmethod
    def runtime_error(msg : str, obj : object | None = None, meth : str | None = None):
        if obj is None:
            if meth is None:
                return Errors._compose_message(title = "Runtime Error", content = msg)
            else:
                return Errors._compose_message(title = f"Runtime Error in method {meth}", content = msg)
        else:
            if meth is None:
                return Errors._compose_message(title = f"Runtime Error in {obj}", content = msg)
            else:
                return Errors._compose_message(title = f"Runtime Error in method {meth} of {obj}", content = msg)


    @staticmethod
    def not_in_vecsp(vec: any, vecsp: object, vec_name:str = "vector", space_name:str = "vector space", add_info:str = "") -> str:
        return Errors._compose_message(
            "VECTOR NOT IN VECTOR SPACE",
            f"""
        The given {vec_name} does not belong to the {space_name}.
        {add_info}
            vec = {vec}
            vecsp = {vecsp}""")

    @staticmethod
    def not_a_vecsp(vecsp: object, cls: type, add_info:str = "") -> str:
        return Errors._compose_message(
            "NOT VECTOR SPACE of CERTAIN TYPE",
            f"""
        The given vector space {vecsp} is not of type {cls}.
        {add_info}"""
        )

    @staticmethod
    def not_equal(first: any, second: any, first_type:any = None, second_type:any = None, add_info:str = ""):
        if first_type == None:
            first_type = type(first)
        if second_type == None:
            second_type = type(second)
        return Errors._compose_message(
            "OBJECTS NOT EQUAL",
            f"""
            Comparing an object of type {first_type} 
            with another of type {second_type} failed.
            {add_info}
            The objects:
                first = {first}
                second = {second}
            """
        )

    @staticmethod
    def not_linear_op(operator: object, add_info:str = "") -> str:
        return Errors._compose_message(
            "OPERATOR NOT LINEAR",
            f"""
            The given operator {operator} is of type {type(operator)} is not linear.
            {add_info}
            """
        )

    @staticmethod
    def not_instance(obj: object, cls:type, add_info:str = "") -> str:
        return Errors._compose_message(
            "NOT CORRECT INSTANCE",
            f"""
            The given object {obj} is not an instance of {cls}.
            {add_info}
            """
        )
    
    @staticmethod
    def indexation(index: any, obj: object, add_info:str = "") -> str:
        return Errors._compose_message(
            "INDEXATION ERROR FOR {type(obj)}",
            f"""
            The given index {index} is not valid for {obj}.
            {add_info}"""
        )


class ClassLogger:
    """The [`logging.Logger`][1] instance. Every subclass has a separate instance, named by its
    fully qualified name. Subclasses should use it instead of `print` for any kind of status
    information to allow users to control output formatting, verbosity and persistence.

    [1]: https://docs.python.org/3/library/logging.html#logging.Logger

    Descriptor that provides a per-class `logging.Logger`.

    - Default name: "<module>.<qualname>"
    - Can be overridden by assigning a logger to `MyClass.log`.
    """

    def __get__(self,instance, owner):
        logger = getattr(owner, "_log", None)
        if logger is None:
            # Otherwise build a default logger for the class
            logger = getLogger(f"{owner.__qualname__}")
            logger.setLevel(WARNING)

            if logger.handlers:
                logger.handlers = []
            handler = StreamHandler()
            handler.setFormatter(
                Formatter(
                    "%(asctime)s %(levelname)-8s %(name)-20s :: %(message)s"
                )
            )
            logger.addHandler(handler)
            owner._log = logger
        return logger

    def __set__(self, instance, value):
        # Allow replacing the class logger
        type(instance)._log = value

class memoized_property:
    def __init__(self, func):
        wraps(func)(self)
        self.func = func
        self.attr = '__memoized_' + func.__qualname__

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        if not hasattr(obj, self.attr):
            setattr(obj, self.attr, self.func(obj))
        return getattr(obj, self.attr)

    def __set__(self, obj, value):
        # allow manual override if you want
        setattr(obj, self.attr, value)

    def __delete__(self, obj):
        # allow `del obj.prop` as the reset syntax
        if hasattr(obj, self.attr):
            delattr(obj, self.attr)

def set_defaults(params, **defaults):
    if params is not None:
        defaults.update(params)
    return defaults


def complex2real(z, axis=-1):
    if not is_complex_dtype(z.dtype):
        raise TypeError(Errors.type_error("complex2real is only defined for complex dtypes!"))
    if z.flags.c_contiguous:
        x = z.view(dtype=z.real.dtype).reshape(z.shape + (2,))
    else:
        x = np.lib.stride_tricks.as_strided(
            z.real, shape=z.shape + (2,),
            strides=z.strides + (z.real.dtype.itemsize,))
    return np.moveaxis(x, -1, axis)


def real2complex(x, axis=-1):
    if not is_real_dtype(x.dtype):
        raise TypeError(Errors.type_error("real2complex is only defined for real dtypes!"))
    if x.shape[axis] != 2:
        raise ValueError(Errors.value_error(f"real2complex needs the complex axis {axis} to be of size 2  but it is {x.shape[axis]}!"))
    x = np.moveaxis(x, axis, -1)
    if np.issubdtype(x.dtype, np.floating) and x.flags.c_contiguous:
        return x.view(dtype=np.result_type(1j, x))[..., 0]
    else:
        z = np.array(x[..., 0], dtype=np.result_type(1j, x))
        z.imag = x[..., 1]
        return z


def is_real_dtype(obj):
    if np.isscalar(obj):
        obj = np.asarray(obj)
    try:
        dtype = obj.dtype
    except AttributeError:
        dtype = np.dtype(obj)
    return (
        np.issubdtype(dtype, np.number) and not np.issubdtype(dtype, np.complexfloating)
    )


def is_complex_dtype(obj):
    if np.isscalar(obj):
        obj = np.asarray(obj)
    try:
        dtype = obj.dtype
    except AttributeError:
        dtype = np.dtype(obj)
    return np.issubdtype(dtype, np.complexfloating)


def is_uniform(x):
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(Errors.value_error("To determine if a vector is uniform the vector need to be one dimensional was given vector of dimension {x.ndim}"))
    if(x.shape[0]==1):
        return True
    diffs = x[1:] - x[:-1]
    return np.allclose(diffs, diffs[0])


def linspace_circle(num, *, start=0, stop=None, endpoint=False):
    if not stop:
        stop = start + 2 * np.pi
    angles = np.linspace(start, stop, num, endpoint)
    return np.stack((np.cos(angles), np.sin(angles)), axis=1)


def make_repr(self, *args, **kwargs):
    try:
        arglist = []
        for arg in args:
            if isinstance(arg, str):
                arglist.append(arg)
            else:
                arglist.append(repr(arg))
        for k, v in sorted(kwargs.items()):
            arglist.append("{}={}".format(repr(k), repr(v)))
        return '{}({})'.format(type(self).__qualname__, ', '.join(arglist))
    except Exception as e:
        return f'ERROR in make_repr of {type(self).__qualname__}: {e}'