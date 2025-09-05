# Verifying contextual sparsity ([Liu et al., 2023](https://proceedings.mlr.press/v202/liu23am/liu23am.pdf)) in multi-class classification.
Details will be updated soon.

---

## Setup Environment
Create the required environment using:
```bash
bash build_env.sh
```

## Download Datasets
Download all datasets (KEEL dataset, LIBSVM dataset, and UCI dataset) with:
```bash
bash get_data.sh
# Files will be stored in raw_data/
```

## Preprocess Datasets
Convert datasets into LIBSVM format and generate stratified 5-fold splits.
For datasets without stratified 5-fold splits (including some LIBSVM datasets and UCI datasets), stratified splitting is performed to preserve label distribution.

```bash
bash process_data.sh
# Files will be stored in svm_data/
```
