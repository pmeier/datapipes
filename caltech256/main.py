import pathlib
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import PIL.Image

import torch.utils.data.datapipes as dp
from torch.utils.data.datapipes.utils.decoder import imagehandler


def _caltech256_sample_map(sample: Tuple[str, Any]) -> Dict[str, Any]:
    path, image = sample
    label_, cls = str(pathlib.Path(path).parent.name).split(".")
    label = int(label_)
    return dict(image=image, path=path, label=label, cls=cls)


def caltech256(
    root: Union[str, pathlib.Path],
    handler: Optional[str] = "pil",
) -> Iterable[Dict[str, Any]]:
    root = pathlib.Path(root).resolve()
    datapipe: Iterable = (str(root / "256_ObjectCategories.tar"),)
    datapipe = dp.iter.LoadFilesFromDisk(datapipe)
    datapipe = dp.iter.ReadFilesFromTar(datapipe)
    if handler:
        datapipe = dp.iter.RoutedDecoder(datapipe, handlers=[imagehandler(handler)])
    datapipe = dp.iter.Map(datapipe, fn=_caltech256_sample_map)

    return datapipe


for sample in caltech256("."):
    assert isinstance(sample["image"], PIL.Image.Image)
    assert isinstance(sample["label"], int)
