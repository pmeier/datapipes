# datapipes

This a proof-of-concept repository on how `torch.utils.data.datapipes` can be used as basis for `torchvision.datasets`.

## General observations

- `pathlib.Path` should be a first-class citizen for paths.
- `dp.iter.LoadFilesFromDisk` should have a `mode` parameter. Forcing `rb` makes it cumbersome to read from plain text 
  files. Maybe even an `opener` parameter would be better that defaults to `open` and respects `mode`.
  
## Datasets

Legend:

- :heavy_check_mark: : Fully working
- :o: : Working, but with a significant performance hit
- :x: Not working.

For :o: and :x:, please check out the `README.md` in the corresponding folder for details.

| `torchvision.datasets.`     | Status             |
|:----------------------------|--------------------|
| [`Caltech101`](caltech101/) | :x:                |
| [`Caltech256`](caltech256/) | :heavy_check_mark: |
| [`CelebA`](celeba/)         | :x:                |
