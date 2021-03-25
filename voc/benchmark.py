import os
import pathlib
import sys
from distutils.dir_util import remove_tree

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from benchmark_utils import benchmark

from main import VOC as VOCFullDP
from alternative import VOC as VOCSemiDP
from torchvision.datasets import (
    VOCDetection as VOCNoDP_Detection,
    VOCSegmentation as VOCNoDP_Segmentation,
)


benchmark(lambda: VOCFullDP(".", target_type="detection"), "VOCFullDP", "detection")

benchmark(lambda: VOCSemiDP(".", target_type="detection"), "VOCSemiDP", "detection")
benchmark(
    lambda: VOCSemiDP(".", target_type="detection", decoder=None),
    "VOCSemiDP",
    "detection",
    "no image decoding",
)
benchmark(
    lambda: VOCSemiDP(".", target_type="segmentation"),
    "VOCSemiDP",
    "segmentation",
)
benchmark(
    lambda: VOCSemiDP(".", target_type="segmentation", decoder=None),
    "VOCSemiDP",
    "segmentation",
    "no image decoding",
)

if os.path.exists("VOCdevkit"):
    remove_tree("VOCdevkit")
benchmark(
    lambda: VOCNoDP_Detection(".", download=True), "VOCNoDP", "cold start", "detection"
)
benchmark(
    lambda: VOCNoDP_Detection(".", download=False), "VOCNoDP", "warm start", "detection"
)
benchmark(
    lambda: VOCNoDP_Segmentation(".", download=False),
    "VOCNoDP",
    "warm start",
    "segmentation",
)
