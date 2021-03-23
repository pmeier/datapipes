import pathlib
import sys
from collections import defaultdict
from typing import Any, Dict, Tuple, Union, Iterable, Iterator, Optional, List

import torch.utils.data.datapipes as dp
from torch.utils.data import IterDataPipe
from torch.utils.data.datapipes.utils.decoder import imagehandler


sys.path.insert(0, str(pathlib.Path(__file__).parent))
from utils import DependentGroupByKey


class IterateOverAnnotations(IterDataPipe):
    def __init__(self, datapipe: Iterable[Tuple[str, Dict[str, Any]]]):
        super().__init__()
        self.datapipe = datapipe

    def __iter__(self) -> Iterator[Tuple[str, Dict[str, Any]]]:
        for path, content in self.datapipe:
            images = content["images"]
            # We index the annotations to enable O(1) access through the image id
            anns = self._index_anns(content["annotations"])
            for image in images:
                key = image["file_name"]
                data = dict(image_id=image["id"], annotations=anns[image["id"]])
                yield key, data

    @staticmethod
    def _index_anns(anns: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        indexed_anns: Dict[str, List[Dict[str, Any]]] = defaultdict(lambda: [])
        for ann in anns:
            image_id = ann.pop("image_id")
            ann["ann_id"] = ann.pop("id")
            indexed_anns[image_id].append(ann)

        return indexed_anns


def _collate_image(data: Tuple[str, Any]) -> Tuple[str, Dict[str, Any]]:
    path, image = data
    return str(pathlib.Path(path).name), dict(image_path=path, image=image)


def _collate_sample(data: List[Tuple[Any, Dict[str, Any]]]) -> Dict[str, Any]:
    sample: Dict[str, Any] = {}
    for _, partial_data in data:
        sample.update(partial_data)
    return sample


def coco(
    image_archive: Union[str, pathlib.Path],
    annotation_archive: Union[str, pathlib.Path],
    decoder: Optional[str] = "pil",
):
    annotation_datapipe: Iterable = (str(pathlib.Path(annotation_archive).resolve()),)
    annotation_datapipe = dp.iter.LoadFilesFromDisk(annotation_datapipe)
    annotation_datapipe = dp.iter.ReadFilesFromZip(annotation_datapipe)
    annotation_datapipe = dp.iter.RoutedDecoder(annotation_datapipe)
    annotation_datapipe = IterateOverAnnotations(annotation_datapipe)

    image_datapipe: Iterable = (str(pathlib.Path(image_archive).resolve()),)
    image_datapipe = dp.iter.LoadFilesFromDisk(image_datapipe)
    image_datapipe = dp.iter.ReadFilesFromZip(image_datapipe)
    if decoder:
        image_datapipe = dp.iter.RoutedDecoder(
            image_datapipe, handlers=[imagehandler(decoder)]
        )
    image_datapipe = dp.iter.Map(image_datapipe, _collate_image)

    datapipe = DependentGroupByKey(
        annotation_datapipe, image_datapipe, key_fn=lambda data: data[0]
    )
    datapipe = dp.iter.Map(datapipe, _collate_sample)

    return datapipe


for sample in coco("train2014.zip", "annotations.zip"):
    pass
