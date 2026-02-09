from numpy import ndarray
from typing import Union, Any

ArrayLike = Union[ndarray, Any]

def cosine_similarity(
    X: ArrayLike, Y: ArrayLike | None = ..., dense_output: bool = ...
) -> ndarray: ...
