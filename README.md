# datapipes

This a proof-of-concept repository on how `torch.utils.data.datapipes` can be used as basis for `torchvision.datasets`.

## General observations

- `pathlib.Path` should be a first-class citizen for paths.
- `dp.iter.LoadFilesFromDisk` should have a `mode` parameter. Forcing `rb` makes it cumbersome to read from plain text 
  files. Maybe even an `opener` parameter would be better that defaults to `open` and respects `mode`.
- Files loaded with `get_file_binaries_from_pathnames` used in `dp.iter.LoadFilesFromDisk` are never closed.
- `dp.Iter.RoutedDecoder` only accepts `(path, buffer)` inputs, which is not usable for us. Our datasets return a 
  buffer as well as some additional information.
- It feels weird to call `dp.iter.LoadFilesFromDisk` for a single file, which is usually the case for our datasets.
- I'm aware that this is not possible if we are streaming archives, but if that is not the case, we should be able to 
  read specific files from an archive. Some datasets contain metadata in a separate file that should be available as 
  soon as we create the dataset rather than based on luck when it is stream with the other files.
- `dp.iter.Map` expects an `IterDataPipe` rather than a more general `Iterable` as the other datapipes.
- Instead of `ReadFilesFrom(Tar|Zip)` there should be `ReadFilesFromArchive` that automatically detect the underlying archive type.
- `dp.iter.ReadFilesFrom(Tar|Zip)` should be split in `ListFilesIn(Tar|Zip)` and `LoadFilesFrom(Tar|Zip)`. Most datasets define some splits of the data so that only a part of the data has to be loaded at all. It would be a good idea to drop unused files before we load them.
- For some reason `dp.iter.ReadFilesFrom(Tar|Zip)` returns the files in reversed alphabetical order. This makes it weird to align this with corresponding text files, which are usually read from top to bottom.

## Datasets

Legend:

- :heavy_check_mark: : Fully working
- :o: : Working, but with a significant performance hit
- :x: Not working.

For :o: and :x:, please check out the `README.md` in the corresponding folder for details.

| `torchvision.datasets.`                    | Status             |
|:-------------------------------------------|--------------------|
| [`Caltech101`](caltech101/)                | :heavy_check_mark: |
| [`Caltech256`](caltech256/)                | :heavy_check_mark: |
| [`CelebA`](celeba/)                        | :heavy_check_mark: |
| [`CIFAR10` / `CIFAR100`](cifar/)           | :heavy_check_mark: |
| [`CocoDetection` / `CocoCaptions`](coco/)  | :heavy_check_mark: |
| [`VOCDetection` / `VOCSegmentation`](voc/) | :heavy_check_mark: |
| [`LSUN`](lsun/)                            | :x:                |
| [`ImageNet`](imagenet/)                    | :heavy_check_mark: |

## Notes

- So far, I think the best approach for datasets with related files is to have each individual datapipe to yield a key for the datapoint as well as the data.