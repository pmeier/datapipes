import contextlib
import datetime
import os
from distutils.dir_util import remove_tree

from main import VOC as VOCFullDP
from alternative import VOC as VOCSemiDP
from torchvision.datasets import (
    VOCDetection as VOCNoDP_Detection,
    VOCSegmentation as VOCNoDP_Segmentation,
)


@contextlib.contextmanager
def disable_console_output():
    with contextlib.ExitStack() as stack, open(os.devnull, "w") as devnull:
        stack.enter_context(contextlib.redirect_stdout(devnull))
        stack.enter_context(contextlib.redirect_stderr(devnull))
        yield


@contextlib.contextmanager
def _benchmark(name, *descriptors):
    tic = datetime.datetime.now()
    with disable_console_output():
        yield
    toc = datetime.datetime.now()
    delta = (toc - tic).total_seconds()
    seconds, milliseconds = divmod(delta, 1)
    print(
        f"{name}[{', '.join(descriptors)}]: {seconds:.0f}s {milliseconds * 1e3:.0f}ms"
    )


def benchmark(constructor, name, *descriptors):
    with _benchmark(name, *descriptors, "construction"):
        dataset = constructor()

    with _benchmark(name, *descriptors, "iteration"):
        iterator = iter(dataset)
        for _ in range(1000):
            next(iterator)

    print("=" * 80)


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
