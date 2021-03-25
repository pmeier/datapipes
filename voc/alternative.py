import collections
import io
import pathlib
import sys
from typing import Any, Dict, Tuple, Union, Optional
import xml.etree.ElementTree as ET

import torch.utils.data.datapipes as dp
from torch.utils.data.datapipes.utils.decoder import imagehandler, Decoder


sys.path.insert(0, str(pathlib.Path(__file__).parent))
from utils import ReadLineFromFile

SPLIT_FOLDER = dict(detection="Main", segmentation="Segmentation")
TARGET_TYPE_FOLDER = dict(detection="Annotations", segmentation="SegmentationClass")


class VOC:
    def __init__(
        self,
        root: Union[str, pathlib.Path],
        *,
        year: str = "2012",
        split: str = "train",
        target_type: str = "detection",  # segmentation
        decoder: Optional[str] = "pil",
    ):
        self.target_type = target_type
        self.decoder = Decoder([imagehandler(decoder)]) if decoder else None

        root = pathlib.Path(root).resolve()
        # TODO: make this variable based on the input
        archive = "VOCtrainval_11-May-2012.tar"

        split_folder = SPLIT_FOLDER[target_type]
        target_type_folder = TARGET_TYPE_FOLDER[target_type]

        datapipe = (str(root / archive),)
        datapipe = dp.iter.LoadFilesFromDisk(datapipe)
        datapipe = dp.iter.ReadFilesFromTar(datapipe)

        split_files: Dict[str, Tuple[str, io.BufferedIOBase]] = {}
        self.images: Dict[str, Tuple[str, io.BufferedIOBase]] = {}
        self.targets: Dict[str, Tuple[str, io.BufferedIOBase]] = {}

        for data in datapipe:
            parent_ = pathlib.Path(data[0]).parent
            parent = parent_.name
            grand_parent = parent_.parent.name

            if grand_parent == "ImageSets" and parent == split_folder:
                dct = split_files
            elif parent == "JPEGImages":
                dct = self.images
            elif parent == target_type_folder:
                dct = self.targets
            else:
                continue

            dct[self._data_to_key(data)] = data

        self.keys = ReadLineFromFile((split_files[split],))

    @staticmethod
    def _data_to_key(data: Tuple[str, Any]) -> str:
        return pathlib.Path(data[0]).stem

    def __iter__(self):
        for _, key in self.keys:
            image_data = self.images[key]
            image_path = image_data[0]
            image = (
                self.decoder(image_data)[image_path] if self.decoder else image_data[1]
            )

            target_data = self.targets[key]
            target_path = target_data[0]
            if self.target_type == "detection":
                target_ = self._parse_voc_xml(ET.parse(target_data[1]).getroot())
                target = target_["annotation"]["object"]
            else:  # self.target_type == "segmentation":
                target = (
                    self.decoder(target_data)[target_path]
                    if self.decoder
                    else target_data[1]
                )

            yield dict(
                image_path=image_path,
                image=image,
                target_path=target_path,
                target=target,
            )

    # straight outta torchvision.datasets
    def _parse_voc_xml(self, node: ET.Element) -> Dict[str, Any]:
        voc_dict: Dict[str, Any] = {}
        children = list(node)
        if children:
            def_dic: Dict[str, Any] = collections.defaultdict(list)
            for dc in map(self._parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == "annotation":
                def_dic["object"] = [def_dic["object"]]
            voc_dict = {
                node.tag: {
                    ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()
                }
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict


if __name__ == "__main__":
    for sample in VOC("."):
        pass
