# optimization

A simple end-to-end PyTorch training pipeline demonstrating:

- **Custom Dataset & DataLoader** — synthetic classification data with `pin_memory` and `non_blocking` transfers
- **Model** — configurable multi-layer MLP with batch normalization
- **Training loop** — AdamW optimizer, cosine LR schedule with linear warmup, train/val split, periodic evaluation

## Usage

```bash
cd optimization
python train.py
```
