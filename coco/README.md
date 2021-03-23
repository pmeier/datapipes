# [`torchvision.datasets.Coco(Detection|Captions)`](https://pytorch.org/vision/stable/datasets.html#coco)

## Setup

- One image archive
- One annotations archive containing `.json` files for a product of different annotation types (`captions`, `instances`, `person_keypoints`), splits (`train`, `val`), and years (`2014`).

## Assumptions

For this proof of concept we assume that the annotation archive only contains a single file which we are going to use. 