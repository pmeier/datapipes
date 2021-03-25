import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from benchmark_utils import benchmark

from main import ImageNet


benchmark(lambda: ImageNet(".", decoder=None), "ImageNet", "no image decoding", n=None)
benchmark(lambda: ImageNet("."), "ImageNet", n=None)
