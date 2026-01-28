from collections import defaultdict
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Literal,
    NamedTuple,
    Optional,
    Protocol,
    TypeVar,
    overload,
)

import numpy as np
import sympy as sp
from numpy.typing import NBitBase
from sympy import Expr
from typing_extensions import Self

NpFlt = np.dtype[np.floating]
Float1D = np.ndarray[tuple[int], NpFlt]
Float2D = np.ndarray[tuple[int, int], NpFlt]
Shape = TypeVar("Shape", bound=tuple[int, ...])
FloatND = np.ndarray[Shape, np.dtype[np.floating[NBitBase]]]


TrajectoryType = TypeVar("TrajectoryType", list[np.ndarray], np.ndarray)


class _BaseSINDy(Protocol):
    optimizer: Any
    feature_library: Any
    feature_names: list[str]

    def fit(self, x: TrajectoryType, t: TrajectoryType, *args, **kwargs) -> Self: ...

    def simulate(self, x0: np.ndarray, t: np.ndarray, **kwargs) -> np.ndarray: ...

    def score(
        self,
        x: TrajectoryType,
        t: TrajectoryType,
        x_dot: TrajectoryType,
        metric: Callable,
    ) -> float: ...

    def predict(self, x: np.ndarray, u: None | np.ndarray = None) -> np.ndarray: ...

    def coefficients(self): ...

    @overload
    def equations(self) -> list[str]: ...

    @overload
    def equations(self, precision: int) -> list[str]: ...

    @overload
    def equations(self, precision: int, fmt: Literal["str"] | None) -> list[str]: ...

    @overload
    def equations(
        self, precision: int, fmt: Literal["sympy"]
    ) -> list[dict[Expr, float]]: ...

    def print(self, precision: int, **kwargs) -> None: ...

    def get_feature_names(self) -> list[str]: ...


class ProbData(NamedTuple):
    """Data bundle for a single trajectory.

    Represents a trajectory's training data and associated metadata.
    """

    dt: float
    t_train: Float1D
    x_train: Float2D
    x_train_true: Float2D
    x_train_true_dot: Float2D
    input_features: list[str]
    integrator: Optional[Any] = None  # diffrax.Solution


class NestedDict(defaultdict):
    """A dictionary that splits all keys by ".", creating a sub-dict.

    Args: see superclass

    Example:

        >>> foo = NestedDict("a.b"=1)
        >>> foo["a.c"] = 2
        >>> foo["a"]["b"]
        1
    """

    def __missing__(self, key):
        try:
            prefix, subkey = key.split(".", 1)
        except ValueError:
            raise KeyError(key)
        return self[prefix][subkey]

    def __setitem__(self, key, value):
        if "." in key:
            prefix, suffix = key.split(".", 1)
            if self.get(prefix) is None:
                self[prefix] = NestedDict()
            return self[prefix].__setitem__(suffix, value)
        else:
            return super().__setitem__(key, value)

    def update(self, other: dict):  # type: ignore
        try:
            for k, v in other.items():
                self.__setitem__(k, v)
        except:  # noqa: E722
            super().update(other)

    def flatten(self):
        """Flattens a nested dictionary without mutating.  Returns new dict"""

        def _flatten(nested_d: dict) -> dict:
            new = {}
            for key, value in nested_d.items():
                if not isinstance(key, str):
                    raise TypeError("Only string keys allowed in flattening")
                if not isinstance(value, dict):
                    new[key] = value
                    continue
                for sub_key, sub_value in _flatten(value).items():
                    new[key + "." + sub_key] = sub_value
            return new

        return _flatten(self)


@dataclass
class DynamicsTrialData:
    trajectories: list[ProbData]
    true_equations: list[dict[sp.Expr, float]]
    sindy_equations: list[dict[sp.Expr, float]]
    model: _BaseSINDy
    input_features: list[str]
    smooth_train: list[np.ndarray]


@dataclass
class SINDyTrialUpdate:
    t_sim: Float1D
    t_test: Float1D
    x_sim: FloatND


@dataclass
class FullDynamicsTrialData(DynamicsTrialData):
    sims: list[SINDyTrialUpdate]
