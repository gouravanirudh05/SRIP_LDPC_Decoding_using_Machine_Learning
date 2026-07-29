# ECCT LDPC(49,24) run analysis

Source log:

`LDPC__Code_n_49_k_24__27_07_2026_04_24_37/logging.txt`

The extraction script is `analyze_ecct_log.py`. It creates:

- `training_metrics.csv`: one row per epoch, using the final batch of each epoch.
- `test_metrics.csv`: final BER/FER/loss values at 4, 5, and 6 dB, plus comparison columns.

Run:

```bash
python analyze_ecct_log.py LDPC__Code_n_49_k_24__27_07_2026_04_24_37/logging.txt
python plot_ecct_results.py
```

The plots are written to `plots/training_curves.png` and
`plots/ber_comparison.png`.

## Run configuration

From the log:

- Code: LDPC(49,24)
- Blocks: `N_dec=6`
- Embedding dimension: `d_model=128`
- Attention heads: `h=8`
- Epochs: 500
- Batches per epoch: 1,000
- Batch size: 128
- Test batch size: 2,048
- Learning rate: `1e-4`
- Seed: 42
- Parameters: 1,203,951
- GPU: Tesla T4

The training loss decreases from `0.169` at epoch 1 to `0.0265` at epoch
500. The final training BER and FER are approximately `0.0108` and `0.0971`.

The lowest logged training loss is `0.0256` at epoch 492. The checkpoint is
updated when training loss improves, so it most likely corresponds to epoch
492. The final test block in the log evaluates the in-memory epoch-500 model;
the script does not reload `best_model` before that evaluation. Therefore the
reported test values should be labelled as **final-epoch results**, not
necessarily best-checkpoint results. Re-evaluate `best_model` separately if
you want checkpoint-based BER/FER.

## Final measured test results

| Eb/N0 | BER | FER | -ln(BER) | Test samples |
|---:|---:|---:|---:|---:|
| 4 dB | 2.61e-3 | 3.00e-2 | 5.95 | 100,352 |
| 5 dB | 2.07e-4 | 3.07e-3 | 8.48 | 100,352 |
| 6 dB | 8.01e-6 | 1.46e-4 | 11.70 | 698,368 |

The observed frame-error counts are approximately 3,010, 308, and 102. The
6 dB point is therefore noisier than the paper benchmark, which targeted at
least 500 frame errors per SNR value.

## Comparison with the paper

The supplied paper is `Transformer-based-error-coding`. Its Table 1 reports,
for LDPC(49,24) and the `N=6, d=128` row:

| Eb/N0 | Paper -ln(BER) | This run -ln(BER) | Difference |
|---:|---:|---:|---:|
| 4 dB | 6.13 | 5.95 | -0.18 |
| 5 dB | 8.71 | 8.48 | -0.23 |
| 6 dB | 12.10 | 11.70 | -0.40 |

Thus this run is reasonably close, but slightly worse than the paper result:
the BER is approximately 1.2x, 1.3x, and 1.4x higher at 4, 5, and 6 dB.
The 6 dB difference should not be overinterpreted until it is re-evaluated
with at least 500 frame errors.

This is not yet an exact reproduction of the paper because:

1. The paper used 1,000 epochs; this run used 500.
2. The paper describes a training SNR range of 3--7 dB, whereas this code
   trains over `range(2, 8)`, i.e. 2--7 dB.
3. The paper used a cosine schedule ending near `5e-7`; this run used the
   modified repository schedule ending at `1e-6`.
4. The paper benchmark requested at least 500 error frames per SNR; this run
   collected only about 102 at 6 dB.
5. The paper describes GEGLU feed-forward layers, while this repository's
   `Model.py` uses a GELU activation followed by linear layers. This should be
   checked if exact paper reproduction is required.

The result is therefore a good preliminary validation of the implementation,
not a final claim of exact paper reproduction.
