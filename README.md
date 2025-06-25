
# MSDCC – Multi‑Scale Decomposition Collaborative Classification

Reference PyTorch‑Lightning implementation accompanying the course report.

## Install
```bash
pip install -r requirements.txt
# or
conda env create -f environment.yaml
conda activate msdcc_env
```

## Train
```bash
python msdcc.py --data /path/to/patch_dataset
```
See `data_utils.py` for patch generation tools.
