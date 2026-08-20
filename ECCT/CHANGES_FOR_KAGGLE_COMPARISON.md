# Changes for the Kaggle comparison

This file records the changes made relative to the original ECCT implementation.

## Runtime and configuration changes

1. The default number of epochs was changed from `1000` to `500`. The
   budget-matched notebook overrides this to `800` (see below).
2. Intermediate testing was reduced. The original code evaluated at epoch 1 and
   every 300 epochs; it now evaluates at `epochs // 2` and at the final epoch,
   which yields both AECCT-matched comparison points from a single run.
3. Training batches per epoch were made configurable through
   `--train_batches_per_epoch`. **The argparse default remains `1000`**, so this
   flag must be passed explicitly — see the warning below.
4. The Kaggle notebook explicitly uses `--test_batch_size=2048` rather than
   `512` to reduce GPU evaluation-loop overhead.

## Settings used by the Kaggle notebook

Use `Kaggle_ECCT_LDPC49.ipynb` in the repository root.

- Code: `LDPC_N49_K24`
- Transformer blocks: `N_dec=6`
- Embedding dimension: `d_model=128`
- Attention heads: `h=8`
- Batch size: `128`
- Epochs: `800`
- Training batches per epoch: `200`
- Test batch size: `2048`
- Learning rate: `1e-4`
- Workers: `4`
- Seed: `42`
- Training Eb/N0 range: 2--7 dB
- Testing Eb/N0 range: 4--6 dB
- Training dataset size: `batch_size * 200` per epoch

This gives equal optimizer-update budgets against the completed AECCT run:

- ECCT: `800 epochs × 200 batches = 160,000 updates`
- AECCT: `400 epochs × 200 batches × 2 phases = 160,000 updates`

Because ECCT also evaluates at `epochs // 2`, one run produces both matched
points: epoch 400 (80,000 updates) against AECCT phase-1 (FP32), and epoch 800
(160,000 updates) against the AECCT final ternary/int8 model. Expected
wall-clock is ~3.3 h on a T4.

`--batch_size` is deliberately left at `128`, identical to AECCT. Reducing it
would introduce a new confound instead of removing one; the training budget is
controlled by `--epochs` and `--train_batches_per_epoch`.

## Warning: `--train_batches_per_epoch` must be passed explicitly

`Main.py` declares `--train_batches_per_epoch` with `default=1000`. The earlier
run in `LDPC49_results/` (`LDPC__Code_n_49_k_24__27_07_2026_04_24_37`) was
launched from `kaggle-ecct-ldpc490c4f18b43d.ipynb`, which omits the flag. It
therefore trained on `500 × 1000 = 500,000` updates — **3.1x the AECCT budget** —
which invalidated that comparison.

That notebook is kept only as provenance for the existing log. **Do not use it to
produce new comparison runs.** A previous version of this file claimed equal
budgets of 100,000 updates per side; neither run used that figure, and the claim
has been corrected here.

These settings are intended for a fair preliminary comparison within one
Kaggle session and are not the untouched paper/repository defaults. See
`comparison/ECCT_vs_AECCT_LDPC49_COMPARISON.md` for the full analysis.
