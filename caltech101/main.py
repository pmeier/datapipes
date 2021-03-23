import pathlib
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.utils.data.datapipes as dp

from torch.utils.data.datapipes.utils.decoder import imagehandler

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from utils import Drop, mathandler, DependentGroupByKey


def _images_drop_condition(data: Tuple[str, Any]) -> bool:
    return pathlib.Path(data[0]).parent.name == "BACKGROUND_Google"


def _images_key_fn(data: Tuple[str, Any]) -> Tuple[str, int]:
    path = pathlib.Path(data[0])

    cls = path.parent.name
    idx = int(path.stem.split("_")[1])

    return cls, idx


CLASS_MAP = {
    "Faces_2": "Faces",
    "Faces_3": "Faces_easy",
    "Motorbikes": "Motorbikes_16",
    "airplanes": "Airplanes_Side_2",
}


def _anns_key_fn(data: Tuple[str, Any]) -> Tuple[str, int]:
    cls, idx = _images_key_fn(data)
    if cls in CLASS_MAP:
        cls = CLASS_MAP[cls]
    return cls, idx


def _map_samples(data: List[Tuple[str, Any]]):
    image_path, image = image_data = data[0]
    ann_path, ann = data[1]

    cls, _ = _images_key_fn(image_data)
    obj_contour = torch.from_numpy(ann["obj_contour"])
    return dict(
        image_path=image_path,
        image=image,
        ann_path=ann_path,
        cls=cls,
        obj_contour=obj_contour,
    )


def caltech101(
    root: Union[str, pathlib.Path], image_decoder: Optional[str] = "pil"
) -> Iterable[Dict[str, Any]]:
    root = pathlib.Path(root).resolve()

    images_datapipe: Iterable = (str(root / "101_ObjectCategories.tar.gz"),)
    images_datapipe = dp.iter.LoadFilesFromDisk(images_datapipe)
    images_datapipe = dp.iter.ReadFilesFromTar(images_datapipe)
    images_datapipe = Drop(images_datapipe, _images_drop_condition)
    if image_decoder:
        images_datapipe = dp.iter.RoutedDecoder(
            images_datapipe, handlers=[imagehandler(image_decoder)]
        )

    anns_datapipe: Iterable = (str(root / "101_Annotations.tar"),)
    anns_datapipe = dp.iter.LoadFilesFromDisk(anns_datapipe)
    anns_datapipe = dp.iter.ReadFilesFromTar(anns_datapipe)
    anns_datapipe = dp.iter.RoutedDecoder(anns_datapipe, handlers=[mathandler()])

    datapipe = DependentGroupByKey(
        images_datapipe, _images_key_fn, (anns_datapipe, _anns_key_fn)
    )
    datapipe = dp.iter.Map(datapipe, fn=_map_samples)

    return datapipe


for sample in caltech101("."):
    image_path = sample["image_path"]
    ann_path = sample["ann_path"]
    assert _images_key_fn((image_path, None)) == _anns_key_fn((ann_path, None))
