import pathlib
from typing import Any, Dict, Tuple, Union, Iterable, Optional, List

import torch
import torch.utils.data.datapipes as dp
from torch.utils.data.datapipes.utils.decoder import imagehandler

import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from utils import DependentGroupByKey, DependentDrop, ReadRowsFromCsv


SPLIT_MAP = {
    "train": 0,
    "valid": 1,
    "test": 2,
}


def _not_in_split(data: Tuple[str, str], *, split: str) -> Tuple[str, bool]:
    _, (image_id, split_idx) = data
    in_split = False if split == "all" else int(split_idx) != SPLIT_MAP[split]
    return image_id, in_split


def _splits_datapipe(root: pathlib.Path, *, split: str) -> Iterable[Tuple[str, bool]]:
    datapipe = (str(root / "list_eval_partition.txt"),)
    datapipe = dp.iter.LoadFilesFromDisk(datapipe)
    datapipe = ReadRowsFromCsv(datapipe)
    datapipe = dp.iter.Map(datapipe, _not_in_split, fn_kwargs=dict(split=split))

    return datapipe


def _key_fn(data: Tuple[str, Any]) -> str:
    return pathlib.Path(data[0]).name


def _images_datapipe(
    root: pathlib.Path,
    split_datapipe: Iterable[Tuple[str, bool]],
    *,
    decoder: Optional[str]
) -> Iterable[Tuple[str, Any]]:
    images_datapipe = (str(root / "img_align_celeba.zip"),)
    images_datapipe = dp.iter.LoadFilesFromDisk(images_datapipe)
    images_datapipe = dp.iter.ReadFilesFromZip(images_datapipe)
    images_datapipe = DependentDrop(images_datapipe, split_datapipe, key_fn=_key_fn)
    if decoder:
        images_datapipe = dp.iter.RoutedDecoder(
            images_datapipe, handlers=[imagehandler(decoder)]
        )

    return images_datapipe


def _collate_ann(data: Tuple[str, List[str]]) -> Tuple[str, List[str]]:
    _, (key, *ann) = data
    return key, ann


def _ann_datapipes(root: pathlib.Path) -> List[Iterable[Tuple[str, List[str]]]]:
    ann_datapipes = []
    for file, csv_kwargs in (
        ("identity_CelebA.txt", dict()),
        ("list_attr_celeba.txt", dict(skip_rows=2)),
        ("list_bbox_celeba.txt", dict(skip_rows=2)),
        ("list_landmarks_align_celeba.txt", dict(skip_rows=2)),
    ):
        ann_datapipe = (str(root / file),)
        ann_datapipe = dp.iter.LoadFilesFromDisk(ann_datapipe)
        ann_datapipe = ReadRowsFromCsv(ann_datapipe, **csv_kwargs)
        ann_datapipe = dp.iter.Map(ann_datapipe, _collate_ann)

        ann_datapipes.append(ann_datapipe)

    return ann_datapipes


def _collate_sample(data: List[Tuple[str, Any]]) -> Dict[str, Any]:
    sample: Dict[str, Any] = {}

    image_path, image = data[0]
    sample["image_path"] = image_path
    sample["image"] = image

    _, identity_ = data[1]
    identity = int(identity_[0])
    sample["identity"] = identity

    _, attr_ = data[2]
    attr = torch.tensor([int(x) for x in attr_]).add(1).bool()
    sample["attr"] = attr

    _, bbox_ = data[3]
    bbox = torch.tensor([int(x) for x in bbox_])
    sample["bbox"] = bbox

    _, landmarks_ = data[4]
    landmarks = torch.tensor([int(x) for x in landmarks_])
    sample["landmarks"] = landmarks

    return sample


def celeba(
    root: Union[str, pathlib.Path],
    split: str = "train",
    decoder: Optional[str] = "pil",
) -> Iterable[Dict[str, Any]]:
    root = pathlib.Path(root).resolve()

    split_datapipe = _splits_datapipe(root, split=split)
    images_datapipe = _images_datapipe(root, split_datapipe, decoder=decoder)
    ann_datapipes = _ann_datapipes(root)

    datapipe = DependentGroupByKey(images_datapipe, _key_fn, *ann_datapipes)
    datapipe = dp.iter.Map(datapipe, _collate_sample)

    return datapipe


for sample in celeba("."):
    pass
