import pathlib
from io import BufferedIOBase
from typing import Any, Dict, Tuple, Union, Iterable, Iterator, Optional, List
import csv

import PIL.Image

import torch.utils.data.datapipes as dp
from torch.utils.data import IterDataPipe
from torch.utils.data.datapipes.utils.decoder import imagehandler


class ReadRowsFromCsv(IterDataPipe):
    def __init__(
        self,
        datapipe: Iterable[Tuple[str, BufferedIOBase]],
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


# dp1 = dp.iter.LoadFilesFromDisk(("list_attr_celeba.txt",))
# dp2 = ReadRowsFromCsv(dp1, skip_rows=2)


class CelebA(IterDataPipe):
    def __init__(self, root: Union[str, pathlib.Path]) -> None:
        self.root = pathlib.Path(root).resolve()

    def __iter__(self) -> Iterator[Tuple[PIL.Image.Image, Dict[str, Any]]]:
        dp1 = dp.iter.LoadFilesFromDisk((str(self.root / "img_align_celeba.zip"),))
        dp2 = dp.iter.ReadFilesFromZip(dp1)
        dp3 = dp.iter.RoutedDecoder(dp2, handlers=[imagehandler("pil")])

        for path, image in dp3:
            # pull the corresponding line based on this key out of all the text files
            key = pathlib.Path(path).name

            yield image, dict(path=path)


for _, features in CelebA("."):
    print(features)
