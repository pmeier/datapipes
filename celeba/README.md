# [`torchvision.datasets.CelebA`](https://pytorch.org/vision/stable/datasets.html#celeba)

## Setup

- One archive containing class folders with images
- One text file grouping the images into three splits
- Multiple text files with annotations for every image

## Problems

- With the current implementation we lose access to the header lines of the annotation 
  text files. Reading them is straight forward, but I'm unsure of how to pass them 
  around. One option would be to read directly into a dictionary rather than a list.
- Since the archive is read bottom to top, the text files need to be read completely 
  into memory.
- Since each image is loaded by `dp.iter.ReadFilesFromZip` we can only drop images 
  afterwards. It would be better to drop them before we actually load their data.
