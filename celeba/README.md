# [`torchvision.datasets.CelebA`](https://pytorch.org/vision/stable/datasets.html#celeba)

## Setup

- One archive containing class folders with images

## Problems

- `dp.iter.LoadFilesFromDisk` has accessing mode `"rb"` hard-coded. To use it to read plain text files, a workaround 
  is needed. For this a simple wrapper `fh = (line.decode() for line in bfh)` is enough.
- Similar to the problem in [`Caltech101`](../caltech101/README.md) we need to pull items from different files based on 
  a key pulled from a primary archive.
