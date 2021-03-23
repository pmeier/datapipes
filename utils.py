import io
from typing import Any, Callable, Dict, Iterable, List, Iterator, Union, Tuple, TypeVar

from torch.utils.data import IterDataPipe

__all__ = ["mathandler", "Drop", "DependentGroupByKey"]

D = TypeVar("D")
D2 = TypeVar("D2")
K = TypeVar("K")


class MatHandler:
    def __init__(self, **loadmat_kwargs: Any) -> None:
        try:
            import scipy.io as sio
        except ImportError as error:
            raise ModuleNotFoundError from error

        self.sio = sio
        self.loadmat_kwargs = loadmat_kwargs

    def __call__(self, key: str, data: bytes) -> Dict[str, Any]:
        with io.BytesIO(data) as stream:
            return self.sio.loadmat(stream, **self.loadmat_kwargs)


def mathandler(**loadmat_kwargs: Any) -> MatHandler:
    return MatHandler(**loadmat_kwargs)


class Drop(IterDataPipe):
    def __init__(self, datapipe: Iterable[D], condition: Callable[[D], bool]) -> None:
        super().__init__()
        self.datapipe = datapipe
        self.condition = condition

    def __iter__(self) -> Iterator[D]:
        for data in self.datapipe:
            if self.condition(data):
                continue

            yield data


class DependentGroupByKey(IterDataPipe):
    def __init__(
        self,
        data_pipe: Iterable[D],
        group_key_fn: Callable[[D], K],
        *dependent_data_pipes: Tuple[Iterable[D2], Callable[[D2], K]],
    ):
        super().__init__()
        self.data_pipe = data_pipe
        self.group_key_fn = group_key_fn
        self.dependent_data_pipes = dependent_data_pipes
        self._buffers: Tuple[Dict[K, D2], ...] = (dict(),) * len(dependent_data_pipes)

    def __iter__(self) -> Iterator[List[Union[D, D2]]]:
        for data in self.data_pipe:
            key = self.group_key_fn(data)
            res: List[Union[D, D2]] = [
                self._next_until_key(
                    dependent_datapipe, key_fn=key_fn, key=key, buffer=buffer
                )
                for (dependent_datapipe, key_fn), buffer in zip(
                    self.dependent_data_pipes, self._buffers
                )
            ]
            res.insert(0, data)
            yield res

    @staticmethod
    def _next_until_key(
        datapipe: Iterable[D], *, key_fn: Callable[[D], K], key: K, buffer: Dict[K, D]
    ) -> D:
        if key in buffer:
            return buffer.pop(key)

        for data in datapipe:
            key_ = key_fn(data)
            if key_ == key:
                return data
            else:
                buffer[key_] = data
        else:
            raise RuntimeError(f"Key {key} was never found")
