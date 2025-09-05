# Verifying contextual sparsity ([Liu et al., 2023](https://proceedings.mlr.press/v202/liu23am/liu23am.pdf)) in multi-class classification.

This project aims to verify contextual sparsity in multi-class classification.
Steps 1–3 are setup and data preparation, while Step 4 is the main part you need to complete.

---

## 1. Setup Environment
Create the required environment using:
```bash
bash build_env.sh
```

## 2. Download Datasets
Download all datasets (aloi and mnist)
```bash
bash get_data.sh
# Files will be stored in raw_data/
```

## 3. Preprocess Datasets
Convert datasets into LIBSVM format and generate stratified 5-fold splits.
```bash
bash process_data.sh
# Files will be stored in svm_data/
```

## 4. Check Contextual Sparsity
Your task is to complete `check_contextual_sparsity.py`.  
Once finished, run the following command:
```bash
python3 check_contextual_sparsity.py
```
The program should generate figures similar to the ones below, illustrating how the percentage of the top-k weights influences accuracy.

<p align="center">
  <img src="figures/aloi.png" alt="ALOI dataset" width="220">
  <img src="figures/mnist.png" alt="MNIST dataset" width="233">
</p>
