# datapipes

This a proof-of-concept repository on how `torch.utils.data.datapipes` can be used as basis for `torchvision.datasets`.

Legend:

- :heavy_check_mark: : Fully working
- :o: : Working, but with a significant performance hit
- :x: Not working.

| `torchvision.datasets.`     | Status             |
|:----------------------------|--------------------|
| [`Caltech101`](caltech101/) | :x:                |
| [`Caltech256`](caltech256/) | :heavy_check_mark: |
