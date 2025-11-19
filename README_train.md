# DCAL Minimal Runner

This workspace reproduces the global-local (GLCA) and pair-wise (PWCA) cross-attention strategy from `refs/dcal.md` with PyTorch.

## Environment
- Python 3.10+
- PyTorch 2.1+ with CUDA (if available) and torchvision
- Pillow, numpy

Install dependencies (example):
```
pip install torch torchvision
```

## Layout
- `weights/ViT-B_16.npz`: Google ViT-B/16 checkpoint copied from `refs/ViT-B_16.npz`.
- `src/models`: ViT backbone and DCAL head.
- `src/datasets`: CUB and VeRi dataset loaders.
- `src/tasks`: Training entrypoints for FGVC (`fgvc_cub.py`) and ReID (`reid_veri.py`).
- `tests/`: Lightweight smoke tests for the model.

## Data
- CUB-200-2011 files expected under `data/CUB_200_2011/` following the official split structure.
- VeRi-776 files should be placed under `data/VeRi_776/` with subfolders `image_train/`, `image_query/`, `image_test/` and metadata files (`name_*.txt`, `train_label.xml`, `test_label.xml`).

## Training Examples
### FGVC on CUB
```
PYTHONPATH=src python3 -m tasks.fgvc_cub \
  --data-root data/CUB_200_2011 \
  --weights weights/ViT-B_16.npz \
  --output runs/fgvc_cub \
  --log-interval 25 \
  --val-interval 1 \
  --wandb --wandb-project dcal --wandb-run-name cub-run
```
This trains for 100 epochs using AdamW, stochastic depth, GLCA (top 10%) and PWCA regularization. Validation accuracy uses SA+GLCA probability fusion.

### ReID on VeRi-776
```
PYTHONPATH=src python3 -m tasks.reid_veri \
  --data-root data/VeRi_776 \
  --weights weights/ViT-B_16.npz \
  --output runs/reid_veri \
  --log-interval 50 \
  --val-interval 1 \
  --wandb --wandb-project dcal --wandb-run-name veri-run
```
This config matches the paper: SGD optimizer, batch size 64 with 4 images per identity, local ratio 0.3, cosine LR decay. Evaluation reports mAP and Rank-1 using concatenated SA/GLCA tokens.

WANDB logging is optional; it activates when `--wandb` is specified. The API key is read from `WANDB_API_KEY`.
`--log-interval` controls how often batches are logged, while `--val-interval` controls validation frequency (in epochs).

## Testing
Run smoke tests to ensure the model graph builds:
```
PYTHONPATH=src pytest tests/test_forward.py -q
```


