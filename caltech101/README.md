# `torchvision.datasets.Caltech101`

## Setup

- Two `.tar` archives for images and annotations
- Image archive contains image files in class folders
- Annotations archive contains `.mat` files in class folders
- Correspondence between images and annotation in general is given by class folder and an index in the file name:
  - `helicopter/image_0001.jpg` does correspond to `helicopter/annotation_0001.mat`
  - `helicopter/image_0001.jpg` does **not** correspond to `ying_yang/annotation_0001.mat`
- Some class folders are named differently in the archives:
  - For example, tThe image class folder `Faces` corresponds to the `Faces_2` annotations class folder
- The image archive has an extra folder (`BACKGROUND_Google`) that has no corresponding folder in the annotations 
  archive
  - It contains files with the same naming scheme as the other images as well as others