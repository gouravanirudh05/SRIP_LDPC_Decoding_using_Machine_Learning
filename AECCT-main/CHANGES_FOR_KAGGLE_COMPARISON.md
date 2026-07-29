# Changes for the Kaggle comparison

This file records the changes made relative to the original AECCT implementation.

## Runtime and configuration changes

1. The default number of epochs was changed from `1000` to `250` per AECCT
   phase. AECCT still has two phases: 250 normal-training epochs followed by
   250 quantization-aware-training epochs, for 500 total epochs. This is the
   primary change intended to keep the run within Kaggle's 12-hour session.
2. The training dataset length was made configurable through
   `train_batches_per_epoch`. The Kaggle baseline uses `200` batches per epoch
   instead of the original `1000`, reducing the number of optimizer updates by
   5x while preserving batch size and model architecture.
3. The default test batch size was changed from `512` to `2048` to reduce GPU
   evaluation-loop overhead.
4. The cosine scheduler now uses `T_max=args.epochs` rather than a hard-coded
   `T_max=1000`, so the schedule matches the selected phase length.
5. The command line now accepts `--epochs`, `--workers`, `--batch_size`,
   `--train_batches_per_epoch`, `--test_batch_size`, `--lr`, and `--seed`.
6. The forced `config.workers = 12` before final evaluation was removed. The
   configured worker count is now used consistently.

## Settings used by the Kaggle notebook

- Code: `LDPC_N49_K24`
- Transformer blocks: `N_dec=6`
- Embedding dimension: `d_model=128`
- Attention heads: `h=8`
- Batch size: `128`
- Training batches per epoch: `200`
- Test batch size: `2048`
- Learning rate: `1e-4`
- Workers: `4`
- Seed: `42`
- Training Eb/N0 range: 2--7 dB
- Testing Eb/N0 range: 4--6 dB
- Training dataset size: `batch_size * 200` per epoch in this baseline.

AECCT still evaluates after phase 1 and after the final quantized inference
model. These are phase-level evaluations, not repeated per-epoch evaluations.

These settings are intended for a bounded-runtime preliminary baseline and are
not the untouched paper/repository defaults. For a paper-budget run, restore
`--epochs 1000 --train_batches_per_epoch 1000`, but it is likely to exceed one
Kaggle session.
