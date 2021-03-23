import csv
import io
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Iterator,
    Optional,
    Union,
    Tuple,
    TypeVar,
)

from torch.utils.data import IterDataPipe

__all__ = [
    "mathandler",
    "Drop",
    "next_until_key",
    "DependentDrop",
    "DependentGroupByKey",
    "ReadRowsFromCsv",
]

D = TypeVar("D")
DD = TypeVar("DD")
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


def next_until_key(
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


class DependentDrop(Drop):
    def __init__(
        self,
        datapipe: Iterable[D],
        condition_datapipe: Iterable[Tuple[K, bool]],
        *,
        key_fn: Callable[[D], K],
    ) -> None:
        self._buffer = dict()
        super().__init__(
            datapipe,
            condition=lambda data: next_until_key(
                condition_datapipe,
                key_fn=lambda condition_data: condition_data[0],
                key=key_fn(data),
                buffer=self._buffer,
            )[1],
        )


class DependentGroupByKey(IterDataPipe):
    def __init__(
        self,
        datapipe: Iterable[D],
        *dependent_data_pipes: Iterable[Tuple[K, DD]],
        key_fn: Callable[[D], K],
    ):
        super().__init__()
        self.datapipe = datapipe
        self.key_fn = key_fn
        self.dependent_datapipes = dependent_data_pipes
        self._buffers: Tuple[Dict[K, DD], ...] = tuple(
            dict() for _ in range(len(dependent_data_pipes))
        )

    def __iter__(self) -> Iterator[List[Union[D, DD]]]:
        for data in self.datapipe:
            key = self.key_fn(data)
            res: List[Union[D, DD]] = [
                next_until_key(
                    dependent_datapipe,
                    key_fn=lambda dependent_data: dependent_data[0],
                    key=key,
                    buffer=buffer,
                )
                for dependent_datapipe, buffer in zip(
                    self.dependent_datapipes, self._buffers
                )
            ]
            res.insert(0, data)
            yield res


class ReadRowsFromCsv(IterDataPipe):
    def __init__(
        self,
        datapipe: Iterable[Tuple[str, io.BufferedIOBase]],
        length: int = -1,
        fieldnames: Optional[str] = None,
        skip_rows: Optional[int] = None,
    ):
        super().__init__()
        self.datapipe = datapipe
        self.length = length

        self.fieldnames = fieldnames
        self.skip_rows = skip_rows or (1 if fieldnames is not None else 0)

    def __iter__(self) -> Iterator[Tuple[str, List[str]]]:
        for path, fh in self.datapipe:
            for _ in range(self.skip_rows):
                next(fh)

            for row in csv.reader((line.decode() for line in fh), delimiter=" "):
                yield path, row
