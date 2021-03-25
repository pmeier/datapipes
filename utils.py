import collections
import csv
import io
import itertools
import pathlib
import queue
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Iterator,
    Optional,
    Union,
    Tuple,
    TypeVar,
)

from torch.utils.data import IterDataPipe
from torch.utils.data.datapipes.utils.common import validate_pathname_binary_tuple

__all__ = [
    "mathandler",
    "Drop",
    "next_until_key",
    "DependentDrop",
    "DependentGroupByKey",
    "ReadRowsFromCsv",
    "SplitByKey",
    "ReadLineFromFile",
    "collate_sample",
    "find",
    "ReadFilesFromRar",
]

D = TypeVar("D")
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
        *dependent_data_pipes: Iterable[Tuple[K, Any]],
        key_fn: Callable[[D], K],
    ):
        super().__init__()
        self.datapipe = datapipe
        self.key_fn = key_fn
        self.dependent_datapipes = dependent_data_pipes
        self._buffers: Tuple[Dict[K, Any], ...] = tuple(
            dict() for _ in range(len(dependent_data_pipes))
        )

    def __iter__(self) -> Iterator[List[Union[D, Any]]]:
        for data in self.datapipe:
            key = self.key_fn(data)
            res: List[Union[D, Any]] = [
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


class SplitByKey(Generic[D]):
    def __init__(self, datapipe: Iterable[D], *, key_fn: Callable[[D], Any]):
        self.datapipe = datapipe
        self._datapipe_iterator = iter(datapipe)
        self.key_fn = key_fn
        self.splits = collections.defaultdict(lambda: _SplittedIterDataPipe(self))

    def __getitem__(self, key: Any) -> "_SplittedIterDataPipe":
        return self.splits[key]

    def next(self) -> None:
        data = next(self._datapipe_iterator)
        key = self.key_fn(data)
        self.splits[key].put(data)


class _SplittedIterDataPipe(IterDataPipe):
    def __init__(self, splitter: SplitByKey[D]) -> None:
        self._splitter = splitter
        self._queue = queue.Queue()

    def __iter__(self):
        while True:
            while self._queue.empty():
                self._splitter.next()

            yield self._queue.get()

    def put(self, data) -> None:
        self._queue.put(data)


class ReadLineFromFile(IterDataPipe):
    def __init__(
        self,
        datapipe: Iterable[Tuple[str, io.BufferedIOBase]],
        decode: bool = True,
        encoding: str = "utf-8",
        strip: bool = True,
    ) -> None:
        self.datapipe = datapipe
        self.decode = decode
        self.encoding = encoding
        self.strip = strip

    def __iter__(self) -> Iterator[Tuple[str, Union[bytes, str]]]:
        for path, buffer in self.datapipe:
            for line in buffer:
                if self.decode:
                    line = line.decode(self.encoding)
                if self.strip:
                    line = line.strip()
                yield path, line


def collate_sample(data: List[Tuple[Any, Any]]) -> Dict[str, Any]:
    sample: Dict[str, Any] = {}
    for _, partial_data in data:
        if isinstance(partial_data, dict):
            sample.update(partial_data)
    return sample


def find(
    datapipe: Iterable[D], key: K, key_fn: Callable[[D], K]
) -> Tuple[D, Iterable[D]]:
    iterator = iter(datapipe)
    buffer: List[D] = []
    for data in iterator:
        key_ = key_fn(data)
        if key_ == key:
            return data, itertools.chain(buffer, iterator)
        else:
            buffer.append(data)
    else:
        raise RuntimeError(f"Datapipe is exhausted, but key {key} was never found")


class ReadFilesFromRar(IterDataPipe):
    def __init__(self, datapipe: Iterable[Tuple[str, io.BufferedIOBase]]):
        self._rarfile = self._verify_dependencies()

        super().__init__()
        self.datapipe: Iterable[Tuple[str, io.BufferedIOBase]] = datapipe

    @staticmethod
    def _verify_dependencies():
        try:
            import rarfile
        except ImportError as error:
            raise ModuleNotFoundError from error

        try:
            # For me bsdtar was not working
            rarfile.tool_setup(bsdtar=False)
        except rarfile.RarCannotExec as error:
            raise RuntimeError from error

        return rarfile

    def __iter__(self) -> Iterator[Tuple[str, io.BufferedIOBase]]:
        for data in self.datapipe:
            validate_pathname_binary_tuple(data)
            path, stream = data
            rar = self._rarfile.RarFile(stream)
            for info in rar.infolist():
                if info.filename.endswith("/"):
                    continue

                inner_path = str(pathlib.Path(path) / info.filename)

                file_obj = rar.open(info)
                file_obj.source_rar = rar

                yield inner_path, file_obj
