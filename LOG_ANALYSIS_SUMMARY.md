# Direct Analysis of ECCT and AECCT Logs

## 1. Configuration

Both runs used the same principal configuration:

| Setting | ECCT | AECCT |
|---|---:|---:|
| Code | LDPC(49,24) | LDPC(49,24) |
| Decoder layers / `N_dec` | 6 | 6 |
| `d_model` | 128 | 128 |
| Attention heads | 8 | 8 |
| Batch size | 128 | 128 |
| Training batches/epoch | 200 | 200 |
| Learning rate | `1e-4` | `1e-4` |
| Test batch size | 2048 | 2048 |
| Seed | 42 | 42 |
| Workers | 4 | 4 |
| Total optimizer updates | 160,000 | 160,000 |

AECCT used two training phases:

- Phase 1: 400 epochs × 200 batches = 80,000 updates, FP32 model.
- Phase 2: 400 epochs × 200 batches = 80,000 updates, quantization-aware training.

Thus, the total optimizer-update budget is approximately matched between ECCT and AECCT.

The internal architectures are not identical. ECCT uses full masked attention, while AECCT uses sparse two-ring attention, Laplacian positional encoding, ternary weights, and int8 activations during its quantized phase.

## 2. Final test results

### Bit Error Rate

| Eb/N0 | ECCT BER | AECCT BER | AECCT relative to ECCT |
|---:|---:|---:|---:|
| 4 dB | `3.52 × 10⁻³` | `3.66 × 10⁻³` | 1.04× worse |
| 5 dB | `3.43 × 10⁻⁴` | `3.67 × 10⁻⁴` | 1.07× worse |
| 6 dB | `1.57 × 10⁻⁵` | `1.43 × 10⁻⁵` | 0.91×, apparently better |

At 4 and 5 dB, ECCT is slightly better. At 6 dB, AECCT appears slightly better. However, the 6 dB result is based on approximately 101 frame errors for each model, so its statistical uncertainty is relatively high.

The appropriate conclusion is:

> AECCT achieved decoding performance very close to ECCT for the LDPC(49,24) configuration. No meaningful accuracy degradation is visible in this run.

It would be too strong to conclude that AECCT is definitively better based on this single experiment.

## 3. What is FER?

FER means Frame Error Rate.

A frame is one complete LDPC codeword. If even one decoded bit is incorrect, the entire frame is counted as erroneous.

For example, at 5 dB:

- ECCT FER = `5.08 × 10⁻³`.
- Approximately 510 of 100,352 tested frames contained an error.

FER is important in communication systems because one incorrect bit usually causes the complete codeword to be rejected or retransmitted.

## 4. Training behavior

### ECCT

- Training time: approximately 3.18 hours.
- Test time: approximately 229 seconds.
- Final training loss: approximately `3.03 × 10⁻²`.

### AECCT

- FP32 phase: approximately 1.45 hours.
- Quantization-aware phase: approximately 3.23 hours.
- Total training time: approximately 4.68 hours.
- Test time: approximately 321 seconds.
- Final training loss: approximately `2.98 × 10⁻²`.

AECCT's QAT phase is slower in the current PyTorch implementation because quantization is simulated during training. This does not prove that the final hardware implementation is slower. The acceleration claim should be evaluated using the converted quantized inference model on suitable hardware.

## 5. Comparison with the ECCT paper

The ECCT paper reports approximately the following `-ln(BER)` values for LDPC(49,24), `N=6`, and `d=128`:

| Eb/N0 | Paper ECCT | This ECCT | This AECCT |
|---:|---:|---:|---:|
| 4 dB | 6.13 | 5.65 | 5.61 |
| 5 dB | 8.71 | 7.98 | 7.91 |
| 6 dB | 12.10 | 11.06 | 11.16 |

The ECCT reproduction is below the paper's reported values, but follows the same trend and is reasonably close. Potential reasons include:

- Different random training samples.
- Different checkpoint-selection behavior.
- A smaller training budget than the original paper.
- GPU and software differences.
- Different test-sample realizations.

Therefore, this should be described as a reproduction attempt rather than an exact replication.

## 6. Overall conclusion

The defensible conclusion is:

> We evaluated ECCT and AECCT on the same LDPC(49,24) code using the same principal hyperparameters and an equal total optimizer-update budget of 160,000 updates. AECCT achieved BER and FER values very close to ECCT across 4–6 dB Eb/N0. The results indicate that AECCT preserves decoding accuracy for this experiment while introducing sparse attention and quantized computation.

The experiment does not establish that AECCT is universally better than ECCT. A stronger claim would require:

- Multiple random seeds.
- Several LDPC codes.
- More SNR points.
- Direct inference-speed measurements.
- Hardware energy or latency measurements.

## 7. Source files

- ECCT log: [`../ECCT_log.txt`](../ECCT_log.txt)
- AECCT complete run log: [`../LDPC49_AECCT_results/LDPC__Code_n_49_k_24__11_08_2026_09_29_06/logging.txt`](../LDPC49_AECCT_results/LDPC__Code_n_49_k_24__11_08_2026_09_29_06/logging.txt)
- Extracted numerical results: [`data/test_results.csv`](data/test_results.csv)
- Detailed generated report: [`ECCT_vs_AECCT_LDPC49_CURRENT_REPORT.md`](ECCT_vs_AECCT_LDPC49_CURRENT_REPORT.md)
