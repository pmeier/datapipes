# datapipes

This a proof-of-concept repository on how `torch.utils.data.datapipes` can be used as basis for `torchvision.datasets`.

## `Caltech101`

### Setup

- Two `.tar` archives for images and annotations
- Image archive contains image files in class folders
- Annotations archive contains `.mat` files in class folders
- Correspondence between images and annotation in general is given by class folder and an index in the file name:
  - `helicopter/image_0001.jpg` does correspond to `helicopter/annotation_0001.mat`
  - `helicopter/image_0001.jpg` does **not** correspond to `ying_yang/annotation_0001.mat`
- Some class folders are named differently in the archives:
  - The image class folder `Faces` corresponds to the `Faces_2` annotations class folder
- The image archive has an extra folder (`BACKGROUND_Google`) that has no corresponding folder in the annotations 
  archive
  - It contains files with the same naming scheme as the other images as well as others

### Problems

1. I wasn't able to pull an item for the image pipe and pull from the annotations pipe until I found the matching file. 
   A crude solution to this would be to `dp.iter.Concat` the `images_dp` and `annotations_dp` and then use 
   `dp.iter.GroupByKey` to pull matching pairs. Unfortunately, this would completely exhaust the `images_dp` before a 
   single annotation is pulled and overflow the buffer or memory for a normal dataset.
2. The approach described in 1. breaks down if the archives are not aligned, i.e. not all files have a 1-to-1 
   correspondence.
   
### Possible solution

To overcome this I propose a `dp.iter.DependenceGroupByKey`. It should take one primary datapipe as well as an 
arbitrary number of secondary datapipes. In every step an item is pulled from the primary datapipe and this is used to 
pull a corresponding item from all secondary datapipes. Of course the secondary datapipes need to store non-matching 
items in a buffer until they are requested.
