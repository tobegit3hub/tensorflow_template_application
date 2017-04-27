## Introduction

This is the general tool to convert CSV file to TFRecords file.

## Cancer

The example data in [cancer.csv](cancer.csv) looks like these.

From [](https://github.com/mark-watson/cancer-deep-learning-model)

```
3,7,7,4,4,9,4,8,1,1
1,1,1,1,2,1,2,1,1,0
4,1,1,3,2,1,3,1,1,0
7,8,7,2,4,8,3,8,2,1
9,5,8,1,2,3,2,1,5,1
```

The first 9th data are features and the last one is label.

```
- 0 Clump Thickness               1 - 10
- 1 Uniformity of Cell Size       1 - 10
- 2 Uniformity of Cell Shape      1 - 10
- 3 Marginal Adhesion             1 - 10
- 4 Single Epithelial Cell Size   1 - 10
- 5 Bare Nuclei                   1 - 10
- 6 Bland Chromatin               1 - 10
- 7 Normal Nucleoli               1 - 10
- 8 Mitoses                       1 - 10
- 9 Class (0 for benign, 1 for malignant)
```

## Usage

Convert CSV file to TFRecords file with this script.

```
python convert_cancer_to_tfrecords.py
```

To verify the TFRecords, you can iterate and print each example with this script.

```
python print_cancer_tfrecords.py
```
