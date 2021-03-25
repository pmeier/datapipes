import pathlib
import sys
from typing import Any, Dict, Union, Iterator, Optional

import torch.utils.data.datapipes as dp
from torch.utils.data.datapipes.utils.decoder import imagehandler

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from utils import find


class _ImageNetMeta:
    def __init__(self, root: pathlib.Path, *, split: str):
        self.split = split
        self.available = False

        devkit = root / f"ILSVRC2012_devkit_t12.tar.gz"
        if not devkit.exists():
            return

        try:
            import scipy.io
        except ImportError:
            return

        self.available = True

        datapipe = (str(devkit),)
        datapipe = dp.iter.LoadFilesFromDisk(datapipe)
        datapipe = dp.iter.ReadFilesFromTar(datapipe)

        (_, stream), datapipe = find(
            datapipe,
            "ILSVRC2012_validation_ground_truth.txt",
            lambda data: pathlib.Path(data[0]).name,
        )
        self._val_labels = {
            f"ILSVRC2012_val_{idx:08d}.JPEG": label
            for idx, label in enumerate(
                (int(line.decode().strip()) for line in stream), 1
            )
        }

        (_, stream), datapipe = find(
            datapipe, "meta.mat", lambda data: pathlib.Path(data[0]).name
        )
        synsets = scipy.io.loadmat(stream, squeeze_me=True)["synsets"]
        labels, wnids, clss = zip(
            *(
                (label, wnid, tuple(classes.split(", ")))
                for label, wnid, classes, _, num_children, *_ in synsets
                if num_children == 0
            )
        )
        self._wnid_to_label = dict(zip(wnids, labels))
        self._label_to_wnid = dict(zip(labels, wnids))
        self._label_to_cls = dict(zip(labels, clss))

    def __call__(self, path: Union[str, pathlib.Path]) -> Dict[str, Any]:
        path = pathlib.Path(path)
        if self.split == "train":
            wnid = path.stem.split("_")[0]
            label = None
        else:  # self.split == "val"
            label = self._val_labels[path.name] if self.available else None
            wnid = None

        if self.available:
            if label:
                wnid = self._label_to_wnid[label]
            elif wnid:
                label = self._wnid_to_label[wnid]

        cls = self._label_to_cls[label] if label else None

        return dict(label=label, wnid=wnid, cls=cls)


class ImageNet:
    def __init__(
        self,
        root: Union[str, pathlib.Path],
        *,
        split: str = "train",
        decoder: Optional[str] = "pil",
    ):
        self.root = pathlib.Path(root)
        self.split = split
        self._meta = _ImageNetMeta(self.root, split=self.split)

        datapipe = (str((self.root / f"ILSVRC2012_img_{split}.tar").resolve()),)
        datapipe = dp.iter.LoadFilesFromDisk(datapipe)
        datapipe = dp.iter.ReadFilesFromTar(datapipe)
        if split == "train":
            # the train archive is a tar of tars
            datapipe = dp.iter.ReadFilesFromTar(datapipe)
        if decoder:
            datapipe = dp.iter.RoutedDecoder(datapipe, handlers=[imagehandler(decoder)])
        self.datapipe = datapipe

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for path, image in self.datapipe:
            sample = dict(image_path=path, image=image)
            sample.update(self._meta(path))
            yield sample


if __name__ == "__main__":
    for sample in ImageNet(".", split="train"):
        pass
