"""
Copperhead: Bringing Rust's safety patterns to Python
"""

from typing import TypeVar, Generic, Callable, Optional, Union, Iterator
from dataclasses import dataclass
from functools import wraps
import traceback
import sys

__all__ = [
    'Result',
    'Option',
    'rrange',
    'result',
    'option',
]

T = TypeVar('T')
E = TypeVar('E')
U = TypeVar('U')

class Result(Generic[T, E]):
    """
    A Result type for explicit error handling.
    Similar to Rust's Result<T, E>.
    Used to handle operations that might fail without using exceptions.
    """
    def __init__(self, value: Union[T, E], is_ok: bool):
        self._value = value
        self._is_ok = is_ok

    @classmethod
    def Ok(cls, value: T) -> 'Result[T, E]':
        """Create a successful Result containing a value."""
        return cls(value, True)

    @classmethod
    def Err(cls, error: E) -> 'Result[T, E]':
        """Create a failed Result containing an error."""
        return cls(error, False)

    def is_ok(self) -> bool:
        """Check if the Result is Ok."""
        return self._is_ok

    def is_err(self) -> bool:
        """Check if the Result is Err."""
        return not self._is_ok

    def unwrap(self) -> T:
        """
        Return the contained Ok value.
        Raises the contained error if the Result is Err.
        """
        if self._is_ok:
            return self._value
        else:
            if isinstance(self._value, BaseException):
                raise self._value
            else:
                raise Exception(f"Unwrapped an Err value: {self._value}")

    def unwrap_or(self, default: T) -> T:
        """Return the contained Ok value or a default if Err."""
        return self._value if self._is_ok else default

    def unwrap_or_else(self, op: Callable[[E], T]) -> T:
        """Return the contained Ok value or compute a default using a function."""
        return self._value if self._is_ok else op(self._value)

    def map(self, op: Callable[[T], U]) -> 'Result[U, E]':
        """Apply a function to the contained Ok value."""
        if self._is_ok:
            try:
                return Result.Ok(op(self._value))
            except Exception as e:
                return Result.Err(e)
        else:
            return Result.Err(self._value)

    def map_err(self, op: Callable[[E], U]) -> 'Result[T, U]':
        """Apply a function to the contained Err value."""
        if self._is_ok:
            return Result.Ok(self._value)
        else:
            return Result.Err(op(self._value))

    def and_then(self, op: Callable[[T], 'Result[U, E]']) -> 'Result[U, E]':
        """Chain operations that might fail."""
        if self._is_ok:
            return op(self._value)
        else:
            return Result.Err(self._value)

    def __repr__(self) -> str:
        if self._is_ok:
            return f"Ok({self._value})"
        else:
            return f"Err({self._value})"

class Option(Generic[T]):
    """
    An Option type for handling optional values explicitly.
    Similar to Rust's Option<T>.
    """
    def __init__(self, value: Optional[T]):
        self._value = value

    @classmethod
    def Some(cls, value: T) -> 'Option[T]':
        """Create an Option containing a value (including None)."""
        return cls(value)

    @classmethod
    def None_(cls) -> 'Option[T]':
        """Create an empty Option."""
        return cls(None)

    def is_some(self) -> bool:
        """Check if the Option contains a value (even if it's None)."""
        return self._value is not None

    def is_none(self) -> bool:
        """Check if the Option is empty (value is None)."""
        return self._value is None

    def unwrap(self) -> T:
        """
        Return the contained value.
        Raises ValueError if the Option is None.
        """
        if self._value is not None:
            return self._value
        else:
            raise ValueError("Called unwrap on a None Option")

    def unwrap_or(self, default: T) -> T:
        """Return the contained value or a default if None."""
        return self._value if self._value is not None else default

    def unwrap_or_else(self, op: Callable[[], T]) -> T:
        """Return the contained value or compute a default using a function."""
        return self._value if self._value is not None else op()

    def map(self, op: Callable[[T], U]) -> 'Option[U]':
        """Apply a function to the contained value if it's not None."""
        if self._value is not None:
            return Option.Some(op(self._value))
        else:
            return Option.None_()

    def and_then(self, op: Callable[[T], 'Option[U]']) -> 'Option[U]':
        """Chain operations that return Options."""
        if self._value is not None:
            return op(self._value)
        else:
            return Option.None_()

    def __repr__(self) -> str:
        if self._value is not None:
            return f"Some({self._value})"
        else:
            return "None_"

class rrange:
    """
    Enhanced range with Rust-like syntax:

    rrange['..5']    -> 0,1,2,3,4       (exclusive)
    rrange['1..5']   -> 1,2,3,4         (exclusive)
    rrange['1..=5']  -> 1,2,3,4,5       (inclusive)
    rrange['3..']    -> 3,4,5,...       (infinite)
    rrange['..']     -> 0,1,2,...       (infinite from 0)
    """
    def __init__(self, start: Optional[int] = None, stop: Optional[int] = None):
        self.start = start
        self.stop = stop

    def __iter__(self) -> Iterator[int]:
        current = self.start or 0
        if self.stop is None:
            while True:
                yield current
                current += 1
        else:
            while current < self.stop:
                yield current
                current += 1

    @classmethod
    def __class_getitem__(cls, key: str) -> 'rrange':
        if not isinstance(key, str):
            raise TypeError("Range requires a string notation (e.g., '1..5', '1..=5')")

        if key == "..":
            return cls(0, None)

        inclusive = '..=' in key
        parts = key.replace('..=', '..').split('..')

        if len(parts) != 2:
            raise ValueError(f"Invalid range syntax: '{key}'")

        start_str, end_str = parts
        start = int(start_str) if start_str else None
        end = int(end_str) if end_str else None

        if inclusive and end is not None:
            end += 1

        return cls(start, end)

    def __repr__(self) -> str:
        start = '' if self.start is None else self.start
        stop = '' if self.stop is None else self.stop
        return f"rrange[{start}..{stop}]"

def result(f: Callable) -> Callable:
    """
    Decorator to convert a function that may raise exceptions into one that returns a Result.
    """
    @wraps(f)
    def wrapper(*args, **kwargs) -> Result:
        try:
            return Result.Ok(f(*args, **kwargs))
        except Exception as e:
            return Result.Err(e)
    return wrapper

def option(f: Callable) -> Callable:
    """
    Decorator to convert a function that may return None into one that returns an Option.
    """
    @wraps(f)
    def wrapper(*args, **kwargs) -> Option:
        result = f(*args, **kwargs)
        return Option.Some(result)
    return wrapper
