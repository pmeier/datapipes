import pathlib
import sys
import warnings
from typing import Any, Dict, Iterator, Union

import torch.utils.data.datapipes as dp
from torch.utils.data.datapipes.utils.decoder import torch_video


sys.path.insert(0, str(pathlib.Path(__file__).parent))
from utils import ReadFilesFromRar


class HMDB51:
    def __init__(self, root: Union[str, pathlib.Path], *, decode: bool = True) -> None:
        self.root = pathlib.Path(root)

        datapipe = ((self.root / "hmdb51_org.rar").resolve(),)
        datapipe = dp.iter.LoadFilesFromDisk(str(file) for file in datapipe)
        datapipe = ReadFilesFromRar(datapipe)
        datapipe = ReadFilesFromRar(datapipe)
        if decode:
            datapipe = dp.iter.RoutedDecoder(datapipe, handlers=[torch_video])
        self.datapipe = datapipe

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for path, video in self.datapipe:
            cls = pathlib.Path(path).parent.name
            yield dict(video_path=path, video=video, cls=cls)


if __name__ == "__main__":
    # TODO: Without this a warning is emitted for every decoded video
    warnings.simplefilter("ignore", UserWarning)

    for sample in HMDB51("."):
        pass
