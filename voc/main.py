import collections
import functools
import pathlib
import sys
from typing import Any, Dict, Tuple, Union, Iterable, Optional
import xml.etree.ElementTree as ET

import torch.utils.data.datapipes as dp
from torch.utils.data.datapipes.utils.decoder import imagehandler


sys.path.insert(0, str(pathlib.Path(__file__).parent))
from utils import DependentGroupByKey, SplitByKey, ReadLineFromFile, collate_sample


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
        archive_datapipe = _make_archive_datapipe(
            root, year=year, split=split, target_type=target_type
        )

        split_datapipe = _make_split_datapipe(
            archive_datapipe["split"], target_type=target_type, split=split
        )

        image_datapipe = _make_image_datapipe(
            archive_datapipe["image"], decoder=decoder
        )
        target_datapipe = _make_target_datapipe(
            archive_datapipe["target"], target_type=target_type, decoder=decoder
        )

        datapipe = DependentGroupByKey(
            split_datapipe, image_datapipe, target_datapipe, key_fn=_group_key_fn
        )
        self.datapipe = dp.iter.Map(datapipe, collate_sample)

    def __iter__(self):
        yield from self.datapipe


def _make_archive_datapipe(
    root: Union[str, pathlib.Path], *, year: str, split: str, target_type: str
) -> SplitByKey:
    root = pathlib.Path(root).resolve()
    # TODO: make this variable based on the input
    archive = "VOCtrainval_11-May-2012.tar"

    datapipe = (str(root / archive),)
    datapipe = dp.iter.LoadFilesFromDisk(datapipe)
    datapipe = dp.iter.ReadFilesFromTar(datapipe)
    datapipe = SplitByKey(
        datapipe, key_fn=functools.partial(_split_key_fn, target_type=target_type)
    )
    return datapipe


def _make_split_datapipe(
    datapipe, *, target_type: str, split: str
) -> Iterable[Tuple[str, str]]:
    for data in datapipe:
        path = pathlib.Path(data[0])
        if path.parent.name == SPLIT_FOLDER[target_type] and path.stem == split:
            return ReadLineFromFile((data,))
    else:
        raise RuntimeError


def _make_image_datapipe(
    datapipe: Iterable, *, decoder: Optional[str]
) -> Iterable[Tuple[str, Dict[str, Any]]]:
    if decoder:
        datapipe = dp.iter.RoutedDecoder(datapipe, handlers=[imagehandler(decoder)])
    datapipe = dp.iter.Map(datapipe, _collate_image)
    return datapipe


def _make_target_datapipe(
    datapipe: Iterable, *, target_type: str, decoder: Optional[str]
) -> Iterable[Tuple[str, Dict[str, Any]]]:
    if target_type == "detection":
        # TODO
        collate = _collate_target_detection
    else:  # target_type == "segmentation":
        if decoder:
            datapipe = dp.iter.RoutedDecoder(datapipe, handlers=[imagehandler(decoder)])

        collate = _collate_target_segmentation
    datapipe = dp.iter.Map(datapipe, collate)
    return datapipe


def _split_key_fn(data: Tuple[str, Any], *, target_type: str) -> Optional[str]:
    path = pathlib.Path(data[0])

    if path.parent.parent.name == "ImageSets":
        return "split"
    elif path.parent.name == "JPEGImages":
        return "image"
    elif path.parent.name == TARGET_TYPE_FOLDER[target_type]:
        return "target"


def _group_key_fn(data: Tuple[str, str]):
    return data[1]


def _path_to_key(path: str) -> str:
    return pathlib.Path(path).stem


def _collate_image(data: Tuple[str, Any]) -> Tuple[str, Dict[str, Any]]:
    path, image = data
    return _path_to_key(path), dict(image_path=path, image=image)


def _collate_target_detection(data: Tuple) -> Tuple:
    path, xml = data
    target = parse_voc_xml(ET.parse(xml).getroot())["annotation"]["object"]
    return _path_to_key(path), dict(target_path=path, target=target)


# straight outta torchvision.datasets
def parse_voc_xml(node: ET.Element) -> Dict[str, Any]:
    voc_dict: Dict[str, Any] = {}
    children = list(node)
    if children:
        def_dic: Dict[str, Any] = collections.defaultdict(list)
        for dc in map(parse_voc_xml, children):
            for ind, v in dc.items():
                def_dic[ind].append(v)
        if node.tag == "annotation":
            def_dic["object"] = [def_dic["object"]]
        voc_dict = {
            node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}
        }
    if node.text:
        text = node.text.strip()
        if not children:
            voc_dict[node.tag] = text
    return voc_dict


def _collate_target_segmentation(data: Tuple) -> Tuple:
    path, seg = data
    return _path_to_key(path), dict(seg_path=path, seg=seg)


for sample in VOC("."):
    pass
