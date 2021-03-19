import pathlib
from typing import Any, Dict, Tuple, Union, Iterator

import PIL.Image

import torch.utils.data.datapipes as dp
from torch.utils.data import IterDataPipe
from torch.utils.data.datapipes.utils.decoder import imagehandler


class Caltech256(IterDataPipe):
    def __init__(self, root: Union[str, pathlib.Path]) -> None:
        self.root = pathlib.Path(root).resolve()

    def __iter__(self) -> Iterator[Tuple[PIL.Image.Image, Dict[str, Any]]]:
        dp1 = dp.iter.LoadFilesFromDisk((str(self.root / "256_ObjectCategories.tar"),))
        dp2 = dp.iter.ReadFilesFromTar(dp1)
        dp3 = dp.iter.RoutedDecoder(dp2, handlers=[imagehandler("pil")])

        for path, image in dp3:
            label_, cls = str(pathlib.Path(path).parent.name).split(".")
            label = int(label_)

            yield image, dict(path=path, label=label, cls=cls)


for image, features in Caltech256("."):
    assert isinstance(image, PIL.Image.Image)
    print(features)

