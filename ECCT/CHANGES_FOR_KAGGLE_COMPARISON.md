# Changes for the Kaggle comparison

This file records the changes made relative to the original ECCT implementation.

## Runtime and configuration changes

1. The default number of epochs was changed from `1000` to `500`.
2. Intermediate testing was disabled. The original code evaluated at epoch 1
   and every 300 epochs; it now evaluates only at the final epoch.
3. Training batches per epoch were made configurable through
   `--train_batches_per_epoch`. The Kaggle notebook uses `200` instead of the
   original `1000`.
4. The Kaggle notebook explicitly uses `--test_batch_size=2048` rather than
   `512` to reduce GPU evaluation-loop overhead.

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
- Training dataset size: `batch_size * 200` per epoch

The bounded-runtime comparison has equal optimizer-update budgets:

- ECCT: `500 epochs × 200 batches = 100,000 updates`
- AECCT: `250 epochs × 200 batches × 2 phases = 100,000 updates`

These settings are intended for a fair preliminary comparison within one
Kaggle session and are not the untouched paper/repository defaults.
