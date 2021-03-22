import pathlib
import pickle
from io import BufferedIOBase
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple, Union

import PIL.Image

import torch.utils.data.datapipes as dp
from torch.utils.data import IterDataPipe


class _CIFAR(IterDataPipe):
    ARCHIVE = Tuple[str, str]

    TRAIN_FILES: Iterable[Tuple[str, str]]
    TEST_FILES: Iterable[Tuple[str, str]]
    META_FILE: Tuple[str, str]

    LABELS_KEY: str
    CLASSES_KEY: str

    HEIGHT = WIDTH = 32

    def __init__(self, root: Union[str, pathlib.Path], *, train: bool = True) -> None:
        self.root = pathlib.Path(root).resolve()
        self.train = train
        self._label_to_class: Optional[Dict[int, str]] = None

    def __iter__(self) -> Iterator[Tuple[PIL.Image.Image, Dict[str, Any]]]:
        names_to_read, _ = zip(*(self.TRAIN_FILES if self.train else self.TEST_FILES))

        dp1 = dp.iter.LoadFilesFromDisk(
            (str(self.root / pathlib.Path(self.ARCHIVE[0]).name),)
        )
        dp2 = dp.iter.ReadFilesFromTar(dp1)

        for path, data in dp2:
            name = pathlib.Path(path).name
            if name not in names_to_read:
                if name == self.META_FILE[0]:
                    self._load_meta(data)
                continue

            content = pickle.load(data)
            images = content["data"].view(-1, 3, self.HEIGHT, self.WIDTH)
            labels = content[self.LABELS_KEY]

            for image_, label in zip(images, labels):
                image = PIL.Image.fromarray(image_.permute(2, 1, 0).numpy())
                yield image, label

    def _load_meta(self, data: BufferedIOBase) -> None:
        content = pickle.load(data)
        self._label_to_class = dict(enumerate(content[self.CLASSES_KEY]))

    @property
    def label_to_class(self) -> Dict[int, str]:
        if self._label_to_class is None:
            raise RuntimeError(
                "The file containing the meta information was not loaded yet."
            )

        return self._label_to_class


class CIFAR10(_CIFAR):
    ARCHIVE = (
        "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
        "c58f30108f718f92721af3b95e74349a",
    )

    TRAIN_FILES = (
        ("data_batch_1", "c99cafc152244af753f735de768cd75f"),
        ("data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"),
        ("data_batch_3", "54ebc095f3ab1f0389bbae665268c751"),
        ("data_batch_4", "634d18415352ddfa80567beed471001a"),
        ("data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"),
    )
    TEST_FILES = (("test_batch", "40351d587109b95175f43aff81a1287e"),)
    META_FILE = ("batches.meta", "5ff9c542aee3614f3951f8cda6e48888")

    LABELS_KEY = "labels"
    CLASSES_KEY = "label_names"


class CIFAR100(_CIFAR):
    ARCHIVE = (
        "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
        "eb9058c3a382ffc7106e4002c42a8d85",
    )

    TRAIN_FILES = (("train", "16019d7e3df5f24257cddd939b257f8d"),)
    TEST_FILES = (("test", "f0ef6b0ae62326f3e7ffdfab6717acfc"),)
    META_FILE = ("meta", "7973b15100ade9c7d40fb424638fde48")

    LABELS_KEY = "fine_labels"
    CLASSES_KEY = "fine_label_names"


for image, label in CIFAR10("."):
    assert isinstance(image, PIL.Image.Image)
    assert isinstance(label, int)

for image, label in CIFAR100("."):
    assert isinstance(image, PIL.Image.Image)
    assert isinstance(label, int)
