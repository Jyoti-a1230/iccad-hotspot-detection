
## Folder structure

5 benchmarks: iccad1 to iccad5, each with train/ and test/ subfolders of PNG files.

## Label encoding (inconsistent across benchmarks)

- iccad1: `HS93.png` / `NHS110.png`
- iccad2-5: `HSCAD2101.png` / `NHSCAD2999.png`
- Test sets additionally show a doubled `N` prefix on non-hotspots plus a `.pngN.png`-style augmentation suffix (e.g. `NNHS544.png7.png`)
- Fix used: check for `"NHS"` substring anywhere in the filename, not just at the start

## Benchmark stats

| benchmark   |   train_hs |   train_nhs |   train_total |   test_hs |   test_nhs |   test_total |
|:------------|-----------:|------------:|--------------:|----------:|-----------:|-------------:|
| iccad1      |         99 |         340 |           439 |       226 |       4679 |         4905 |
| iccad2      |        174 |        5285 |          5459 |       498 |      41298 |        41796 |
| iccad3      |        909 |        4643 |          5552 |      1808 |      46333 |        48141 |
| iccad4      |         95 |        4452 |          4547 |       177 |      31890 |        32067 |
| iccad5      |         26 |        2716 |          2742 |        41 |      19327 |        19368 |

## Known risk: augmentation and train/test base-pattern overlap

Each benchmark's files are augmented copies (~4.5x) of a smaller set of unique base patterns.

| benchmark   |   train_files |   train_unique_bases |   test_files |   test_unique_bases |   overlap_bases |   overlap_pct_of_train |   clean_test_files |   clean_test_pct_of_test |
|:------------|--------------:|---------------------:|-------------:|--------------------:|----------------:|-----------------------:|-------------------:|-------------------------:|
| iccad1      |           439 |                  439 |         4905 |                1090 |             312 |                   71.1 |               4593 |                     93.6 |
| iccad2      |          5459 |                 5459 |        41796 |                9288 |            4320 |                   79.1 |              37476 |                     89.7 |
| iccad3      |          5552 |                 5552 |        48141 |               10698 |            4450 |                   80.2 |              43691 |                     90.8 |
| iccad4      |          4547 |                 4547 |        32067 |                7126 |            3481 |                   76.6 |              28586 |                     89.1 |
| iccad5      |          2742 |                 2742 |        19368 |                4304 |            2137 |                   77.9 |              17231 |                     89   |

Mitigation: report both standard test accuracy (comparable to published papers) and accuracy on the 'clean' test subset (base patterns not seen in training) for an honest generalization estimate.
