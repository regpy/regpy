from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any, Generator
    from numpy.typing import ArrayLike
    from regpy.vecsps.base import ReadOnlyVector, Scalar
    from regpy.functionals.base import Functional
from functools import wraps
from logging import Logger, getLogger, INFO, StreamHandler, Formatter, WARNING

import numpy as np


_default_rng = np.random.default_rng()
"""The default Random Generator to be used throughout RegPy
    """

def get_rng() -> np.random.Generator:
    r"""Return the global Random Generator of RegPy.
    
    Returns
    -------
    np.random.Generator
        The random generator.
    """
    return _default_rng

def set_rng_seed(seed : None | int | np.random.SeedSequence | np.random.BitGenerator | np.random.Generator | np.random.RandomState = None) -> None:
    r"""Reset the global RNG seed using the provided seed for RegPy.
    
    Parameters
    ----------
    seed: None | int | np.random.SeedSequence | np.random.BitGenerator | np.random.Generator | np.random.RandomState [default: None]
        The random seed to be used by the `numpy.random.default_rng` to construct the random generator used 
        to generate pseudo random vectors. For possible details how the argument is handled we refer to the 
        numpy documentation.
    """
    global _default_rng
    _default_rng = np.random.default_rng(seed)



class Errors:

    @staticmethod
    def _compose_message(title : str, content : str) -> str:
        return f"""
-------------------------------------------------------
    RegPy Error - {title}
{content}
-------------------------------------------------------"""

    @staticmethod
    def generic_message(msg : str) -> str:
        r"""Generic message.
        
        Parameters
        ----------
        msg: str
            Msg.
        
        
        Returns
        -------
        str
            Result described above.
        """
        return Errors._compose_message(title = "Generic Error", content = msg)
    
    @staticmethod
    def failed_test(msg : str, 
                    obj : object | None = None, 
                    meth : str | None = None) -> str:
        r"""Failed test.
        
        Parameters
        ----------
        msg: str
            Msg.
        obj: object | None [default: None]
            Obj.
        meth: str | None [default: None]
            Meth.
        
        
        Returns
        -------
        str
            Result described above.
        """
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
    def value_error(msg : str, obj : object | None = None, meth : str | None = None) -> str:
        r"""Value error.
        
        Parameters
        ----------
        msg: str
            Msg.
        obj: object | None [default: None]
            Obj.
        meth: str | None [default: None]
            Meth.
        
        
        Returns
        -------
        str
            Result described above.
        """
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
    def type_error(msg : str, obj : object | None = None, meth : str | None = None) -> str:
        r"""Type error.
        
        Parameters
        ----------
        msg: str
            Msg.
        obj: object | None [default: None]
            Obj.
        meth: str | None [default: None]
            Meth.
        
        
        Returns
        -------
        str
            Result described above.
        """
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
    def runtime_error(msg : str, obj : object | None = None, meth : str | None = None) -> str:
        r"""Runtime error.
        
        Parameters
        ----------
        msg: str
            Msg.
        obj: object | None [default: None]
            Obj.
        meth: str | None [default: None]
            Meth.
        
        
        Returns
        -------
        str
            Result described above.
        """
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
    def not_in_vecsp(vec: Any, vecsp: object, vec_name:str = "vector", space_name:str = "vector space", add_info:str = "") -> str:
        r"""Not in vecsp.
        
        Parameters
        ----------
        vec: Any
            Vec.
        vecsp: object
            Underlying vector space.
        vec_name: str [default: 'vector']
            Vec name.
        space_name: str [default: 'vector space']
            Space name.
        add_info: str [default: '']
            Add info.
        
        
        Returns
        -------
        str
            Result described above.
        """
        return Errors._compose_message(
            "VECTOR NOT IN VECTOR SPACE",
            f"""
        The given {vec_name} does not belong to the {space_name}.
        {add_info}
            vec = {vec}
            vecsp = {vecsp}""")

    @staticmethod
    def not_a_vecsp(vecsp: object, cls: type, add_info:str = "") -> str:
        r"""Not a vecsp.
        
        Parameters
        ----------
        vecsp: object
            Underlying vector space.
        add_info: str [default: '']
            Add info.
        
        
        Returns
        -------
        str
            Result described above.
        """
        return Errors._compose_message(
            "NOT VECTOR SPACE of CERTAIN TYPE",
            f"""
        The given vector space {vecsp} is not of type {cls}.
        {add_info}"""
        )

    @staticmethod
    def not_equal(first: Any, second: Any, first_type:Any = None, second_type:Any = None, add_info:str = "") -> str:
        r"""Not equal.
        
        Parameters
        ----------
        first: Any
            First.
        second: Any
            Second.
        first_type: Any [default: None]
            First type.
        second_type: Any [default: None]
            Second type.
        add_info: str [default: '']
            Add info.
        
        
        Returns
        -------
        str
            Result described above.
        """
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
        r"""Not linear op.
        
        Parameters
        ----------
        operator: object
            Operator to inspect or evaluate.
        add_info: str [default: '']
            Add info.
        
        
        Returns
        -------
        str
            Result described above.
        """
        return Errors._compose_message(
            "OPERATOR NOT LINEAR",
            f"""
            The given operator {operator} is of type {type(operator)} is not linear.
            {add_info}
            """
        )

    @staticmethod
    def not_instance(obj: object, cls:type, add_info:str = "") -> str:
        r"""Not instance.
        
        Parameters
        ----------
        obj: object
            Obj.
        add_info: str [default: '']
            Add info.
        
        
        Returns
        -------
        str
            Result described above.
        """
        return Errors._compose_message(
            "NOT CORRECT INSTANCE",
            f"""
            The given object {obj} is not an instance of {cls}.
            {add_info}
            """
        )
    
    @staticmethod
    def indexation(index: Any, obj: object, add_info:str = "") -> str:
        r"""Indexation.
        
        Parameters
        ----------
        index: Any
            Index.
        obj: object
            Obj.
        add_info: str [default: '']
            Add info.
        
        
        Returns
        -------
        str
            Result described above.
        """
        return Errors._compose_message(
            "INDEXATION ERROR FOR {type(obj)}",
            f"""
            The given index {index} is not valid for {obj}.
            {add_info}"""
        )


class ClassLogger:
    """Provide a default class logger with optional instance overrides.

    The default logger is named by the concrete class's ``__qualname__``.
    Its cache is local to that class rather than inherited from a base class.
    Instances of the same class share this default logger, including its
    level and handlers.

    Assigning ``obj.log = logger`` overrides the logger for that object only.
    ``del obj.log`` removes this override and restores the class default.
    This allows an inner stopping rule to use a solver's child logger without
    changing the loggers used by unrelated rules.

    Assigning ``MyClass.log = logger`` replaces the descriptor on that class
    through normal Python class assignment; it does not call ``__set__``.
    Use instance assignment when retaining descriptor behavior is desired.
    """

    def __get__(
        self, instance: object | None, owner: type | None = None
    ) -> Logger:
        if owner is None:
            if instance is None:
                raise TypeError("ClassLogger requires an instance or owner.")
            owner = type(instance)
        if instance is not None:
            logger = vars(instance).get("_log")
            if logger is not None:
                return logger

        # Do not inherit a logger cached on a base class.
        logger = vars(owner).get("_log")
        if logger is None:
            logger = getLogger(owner.__qualname__)
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

    def __set__(self, instance: object, value: Logger) -> None:
        r"""Override the logger for one instance without modifying its class.
        
        Parameters
        ----------
        instance: object
            Instance.
        value: Logger
            Value.
        """
        if not isinstance(value, Logger):
            raise TypeError("log must be a logging.Logger instance.")
        vars(instance)["_log"] = value

    def __delete__(self, instance: object) -> None:
        r"""Remove an instance override and restore the default class logger.
        
        Parameters
        ----------
        instance: object
            Instance.
        """
        vars(instance).pop("_log", None)

class memoized_property:
    r"""memoized_property interface.
    
    Parameters
    ----------
    func: Functional
        Functional to evaluate or transform.
    """
    def __init__(self, func: Functional) -> None:
        wraps(func)(self)
        self.func = func
        self.attr = '__memoized_' + func.__qualname__

    def __get__(self, obj: object, objtype: type | None=None) -> Any:
        if obj is None:
            return self
        if not hasattr(obj, self.attr):
            setattr(obj, self.attr, self.func(obj))
        return getattr(obj, self.attr)

    def __set__(self, obj: object, value: Scalar | ArrayLike) -> None:
        # allow manual override if you want
        setattr(obj, self.attr, value)

    def __delete__(self, obj: object) -> None:
        # allow `del obj.prop` as the reset syntax
        if hasattr(obj, self.attr):
            delattr(obj, self.attr)

def set_defaults(params: dict[str, Any], **defaults: Any) -> Any:
    r"""Set defaults.
    
    Parameters
    ----------
    params: dict[str, Any]
        Options for params.
    **defaults: Any
        Defaults.
    
    
    Returns
    -------
    Any
        Result described above.
    """
    if params is not None:
        defaults.update(params)
    return defaults


def complex2real(z: ReadOnlyVector, axis: int=-1) -> np.ndarray:
    r"""Complex2real.
    
    Parameters
    ----------
    z: ReadOnlyVector
        Input vector; its coefficients are only read.
    axis: int [default: -1]
        Axis.
    
    
    Returns
    -------
    np.ndarray
        Computed array.
    """
    if not is_complex_dtype(z.dtype):
        raise TypeError(Errors.type_error("complex2real is only defined for complex dtypes!"))
    if z.flags.c_contiguous:
        x = z.view(dtype=z.real.dtype).reshape(z.shape + (2,))
    else:
        x = np.lib.stride_tricks.as_strided(
            z.real, shape=z.shape + (2,),
            strides=z.strides + (z.real.dtype.itemsize,))
    return np.moveaxis(x, -1, axis)


def real2complex(x: ReadOnlyVector, axis: int=-1) -> np.ndarray:
    r"""Real2complex.
    
    Parameters
    ----------
    x: ReadOnlyVector
        Input vector; its coefficients are only read.
    axis: int [default: -1]
        Axis.
    
    
    Returns
    -------
    np.ndarray
        Computed array.
    """
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


def is_real_dtype(obj: object) -> bool:
    r"""Is real dtype.
    
    Parameters
    ----------
    obj: object
        Obj.
    
    
    Returns
    -------
    bool
        Whether the stated condition is satisfied.
    """
    if np.isscalar(obj):
        obj = np.asarray(obj)
    try:
        dtype = obj.dtype
    except AttributeError:
        dtype = np.dtype(obj)
    return (
        np.issubdtype(dtype, np.number) and not np.issubdtype(dtype, np.complexfloating)
    )


def is_complex_dtype(obj: object) -> bool:
    r"""Is complex dtype.
    
    Parameters
    ----------
    obj: object
        Obj.
    
    
    Returns
    -------
    bool
        Whether the stated condition is satisfied.
    """
    if np.isscalar(obj):
        obj = np.asarray(obj)
    try:
        dtype = obj.dtype
    except AttributeError:
        dtype = np.dtype(obj)
    return np.issubdtype(dtype, np.complexfloating)


def is_uniform(x: ReadOnlyVector) -> bool:
    r"""Is uniform.
    
    Parameters
    ----------
    x: ReadOnlyVector
        Input vector; its coefficients are only read.
    
    
    Returns
    -------
    bool
        Whether the stated condition is satisfied.
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(Errors.value_error("To determine if a vector is uniform the vector need to be one dimensional was given vector of dimension {x.ndim}"))
    if(x.shape[0]==1):
        return True
    diffs = x[1:] - x[:-1]
    return np.allclose(diffs, diffs[0])


def linspace_circle(num: int, *, start: int=0, stop: int | None=None, endpoint: bool=False) -> np.ndarray:
    r"""Linspace circle.
    
    Parameters
    ----------
    num: int
        Num.
    start: int [default: 0]
        Start.
    stop: int | None [default: None]
        Stop.
    endpoint: bool [default: False]
        Endpoint.
    
    
    Returns
    -------
    np.ndarray
        Computed array.
    """
    if not stop:
        stop = start + 2 * np.pi
    angles = np.linspace(start, stop, num, endpoint)
    return np.stack((np.cos(angles), np.sin(angles)), axis=1)


def make_repr(self, *args: Any, **kwargs: Any) -> str:
    r"""Make repr.
    
    Parameters
    ----------
    *args: Any
        Positional arguments forwarded to the underlying implementation.
    **kwargs: Any
        Additional keyword arguments forwarded to the underlying implementation.
    
    
    Returns
    -------
    str
        Result described above.
    """
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