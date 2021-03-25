# Benchmark

The following results were obtained by running `python benchmark.py` against the original [2012 training archive](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar):

```
VOCFullDP[detection, construction]: 0s 813ms
VOCFullDP[detection, iteration]: 7s 423ms
================================================================================
VOCSemiDP[detection, construction]: 1s 973ms
VOCSemiDP[detection, iteration]: 2s 506ms
================================================================================
VOCSemiDP[detection, no image decoding, construction]: 1s 923ms
VOCSemiDP[detection, no image decoding, iteration]: 0s 70ms
================================================================================
VOCSemiDP[segmentation, construction]: 1s 774ms
VOCSemiDP[segmentation, iteration]: 2s 844ms
================================================================================
VOCSemiDP[segmentation, no image decoding, construction]: 1s 780ms
VOCSemiDP[segmentation, no image decoding, iteration]: 0s 1ms
================================================================================
VOCNoDP[cold start, detection, construction]: 5s 539ms
VOCNoDP[cold start, detection, iteration]: 2s 520ms
================================================================================
VOCNoDP[warm start, detection, construction]: 0s 9ms
VOCNoDP[warm start, detection, iteration]: 2s 524ms
================================================================================
VOCNoDP[warm start, segmentation, construction]: 0s 2ms
VOCNoDP[warm start, segmentation, iteration]: 2s 465ms
================================================================================
```

Legend:

- [`VOCFullDP`](main.py): The datapipe is only consumed until the correct split file is found during construction
- [`VOCSemiDP`](alternative.py): The datapipe is fully consumed during construction and the files are indexed. Decoding still happens only at runtime.
- [`VOCNoDP`](https://pytorch.org/vision/stable/datasets.html#voc): The archive is extracted (cold start) and the files are indexed (warm start). Decoding happens at runtime.

## Observations:

- Regardless of the approach the construction time is insignificant compared to the iteration by keeping in mind that some computation will happen during the latter.
- The performance of `VOCSemiDP` and `VOCNoDP` is comparable.
- `VOCFullDP` is quite inefficient in its current state taking ~3x iteration time compared to the other approaches. If we decide this is the way to go, I'll have a look to find and hopefully fix the bottleneck.
