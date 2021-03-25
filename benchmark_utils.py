import contextlib
import datetime
import os

import more_itertools


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


def benchmark(constructor, name, *descriptors, n=1000):
    with _benchmark(name, *descriptors, "construction"):
        dataset = constructor()

    with _benchmark(name, *descriptors, "iteration"):
        more_itertools.consume(iter(dataset), n=n)

    print("=" * 80)
