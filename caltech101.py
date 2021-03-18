import contextlib
import io
import pathlib
from typing import Any, Dict, Tuple

import torch.utils.data.datapipes as dp
from torch.utils.data.datapipes.utils.decoder import imagehandler


data_dir = pathlib.Path("data/caltech101")
# images_archive = data_dir / "101_ObjectCategories.tar.gz"
# uncomment the line below to use an aligned image archive instead
images_archive = data_dir / "101_ObjectCategories_aligned.tar.gz"
annotations_archive = data_dir / "101_Annotations.tar"

images_dp1 = dp.iter.LoadFilesFromDisk((str(images_archive),))
images_dp2 = dp.iter.ReadFilesFromTar(images_dp1)
images_dp3 = dp.iter.RoutedDecoder(images_dp2, handlers=[imagehandler("pil")])
images_dp = images_dp3


class MatHandler:
    def __init__(self, **loadmat_kwargs: Any) -> None:
        try:
            import scipy.io as sio
        except ImportError as error:
            raise ModuleNotFoundError from error

        self.sio = sio
        self.loadmat_kwargs = loadmat_kwargs

    def __call__(self, key: str, data: bytes) -> Dict[str, Any]:
        with io.BytesIO(data) as stream:
            return self.sio.loadmat(stream, **self.loadmat_kwargs)


def mathandler(**loadmat_kwargs: Any) -> MatHandler:
    return MatHandler(**loadmat_kwargs)


annotations_dp1 = dp.iter.LoadFilesFromDisk((str(annotations_archive),))
annotations_dp2 = dp.iter.ReadFilesFromTar(annotations_dp1)
annotations_dp3 = dp.iter.RoutedDecoder(annotations_dp2, handlers=[mathandler()])
annotations_dp = annotations_dp3


dataset_dp1 = dp.iter.Concat(images_dp, annotations_dp)

CLASS_MAP = {
    "Faces": "Faces_2",
    "Faces_easy": "Faces_3",
    "Motorbikes": "Motorbikes_16",
    "airplanes": "Airplanes_Side_2"
}


def group_key_fn(item: Tuple[str, Any]) -> Tuple[str, int]:
    path = pathlib.Path(item[0])

    cls = path.parent.name
    with contextlib.suppress(KeyError):
        cls = CLASS_MAP[cls]

    idx = int(path.stem.split("_")[1])

    return cls, idx


dataset_dp2 = dp.iter.GroupByKey(dataset_dp1, group_size=2, group_key_fn=group_key_fn)
dataset_dp = dataset_dp2

for items in dataset_dp:
    item1, item2 = items

    assert group_key_fn(item1) == group_key_fn(item2)
