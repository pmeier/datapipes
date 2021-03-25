# [`torchvision.datasets.LSUN`](https://pytorch.org/vision/stable/datasets.html#lsun)

## Setup

- Multiple `.zip` archives named `{category}_{split}_lmdb.zip`
- Each archive contains a single `lmdb` database
    ```
    $ unzip bedroom_train_lmdb.zip
    Archive:  bedroom_train_lmdb.zip
       creating: bedroom_train_lmdb/
      inflating: bedroom_train_lmdb/data.mdb  
      inflating: bedroom_train_lmdb/lock.mdb  
    ```

## Problems

- Hopefully I'm wrong here, but I didn't find a way to open an `lmdb` database by file handle