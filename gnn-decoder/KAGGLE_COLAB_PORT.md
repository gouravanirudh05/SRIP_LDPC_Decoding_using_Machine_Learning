# Running `GNN_decoder_LDPC_5G test.ipynb` on Kaggle / Colab — changes and rationale

Date: 2026-08-11

Two files were modified so that the notebook runs unchanged on **Kaggle Notebooks**, **Google Colab**
and **locally**, instead of only on the machine it was written on:

| File | Nature of change |
| --- | --- |
| `GNN_decoder_LDPC_5G test.ipynb` | 4 new cells, 4 modified cells, rest untouched |
| `gnn_torch.py` | `train_gnn()` gained checkpointing + resume |

Nothing about the decoder architecture, the channel model, the code construction or the evaluation
methodology was changed. The GNN, `E2EModel`, `LDPC5GGNN`, `generate_pruned_pcm_5g` and
`transfer_gnn_weights` are byte-identical to before.

---

## 1. Why the notebook could not run as-is on Kaggle/Colab

| Blocker | Consequence on a hosted runtime |
| --- | --- |
| `from gnn_torch import ...` assumes the module sits next to the notebook | `ModuleNotFoundError` on Kaggle, where only the `.ipynb` is uploaded |
| No install cell; setup only documented in prose (`pip install ... sionna-no-rt`) | `ModuleNotFoundError: sionna` — Kaggle/Colab images ship neither Sionna 2 nor torch ≥ 2.9.1 |
| Hardcoded relative paths `results/` and `'results/ldpc_5g_ber.csv'` | Writes land in a non-persisted CWD; on Kaggle only `/kaggle/working` survives the session |
| Comments/prose assume macOS + Apple MPS | Misleading; gives no signal about whether a GPU was actually picked up |
| Training schedule is 635 000 iterations with a single save at the very end | Kaggle caps sessions at 12 h (and ~30 h/week GPU quota), Colab is preemptible → the session dies and **all** training is lost |

---

## 2. Notebook changes (cell by cell)

Final cell order (new cells in **bold**):

1. `8aa94ca2` — NVIDIA copyright (raw, unchanged)
2. `386f0ba4` — title markdown *(modified)*
3. **`setup-md-1` — "0. Runtime detection and dependency install" heading**
4. **`env-setup` — runtime detection + dependency install**
5. **`gnn-torch-bootstrap` — locate `gnn_torch.py`**
6. **`gnn-torch-bundle` — bundled fallback copy of `gnn_torch.py`**
7. `0a65e588` — Sionna/PyTorch imports *(modified)*
8. `1d293650` … `9f251a37` — hyperparameters/code/graph/baseline cells (`89b34d4c` *modified*, rest unchanged)
9. `56256283-…` — training cell *(modified)*
10. `959ac1d6`, `c08a030a-…`, `1a2605f7-…` — evaluation cells (unchanged)
11. `3f518bcd-…` — CSV export *(modified)*

The old standalone cell `bfcba1be-…` (`print(f'PyTorch device: {device}')`, MPS remark) was removed —
it duplicated the device print in the import cell and its Apple-Metal claim was wrong for Sionna,
which supports only CPU and CUDA.

### 2.1 New cell `env-setup` — runtime detection + install

What it does:

* Detects the runtime: `KAGGLE_KERNEL_RUN_TYPE` / `/kaggle/working` → Kaggle;
  `google.colab` in `sys.modules` / `COLAB_RELEASE_TAG` → Colab; otherwise local.
* Defines `WORK_DIR` (`/kaggle/working`, `/content`, or CWD) and `RESULTS_DIR = WORK_DIR/"results"`,
  which every later cell uses instead of a relative path.
  *Why:* on Kaggle only `/kaggle/working` is writable **and** persisted into the version output.
* Hard-fails with an explicit message if Python < 3.11.
  *Why:* `sionna-no-rt` 2.0.1 declares `Requires-Python: >=3.11`; the failure would otherwise be an
  opaque pip resolution error.
* Reads installed versions with `importlib.metadata.version(...)` **without importing torch**, then
  installs only what is missing: `torch>=2.9.1`, `sionna-no-rt>=2.0.0`.
  *Why:* `sionna-no-rt` 2.0.1 requires `torch>=2.9.1`, which is newer than the torch shipped in current
  Kaggle/Colab images, so an upgrade is normally unavoidable — but a *conditional* install keeps the
  cell a no-op on a machine that is already set up (and on re-runs after a restart). Probing via
  metadata rather than `import torch` matters because a module that has already been imported cannot
  be swapped in place.
* If torch/sionna were already imported in the session, it prints an explicit
  "restart the runtime, then re-run this cell" message instead of continuing into a half-upgraded state.

Version pins come from the actual wheel metadata of `sionna-no-rt` 2.0.1
(`Requires-Python >=3.11`, `torch>=2.9.1`, `numpy>=2.2.6`, `scipy>=1.15.3`, `matplotlib>=3.10.8`),
not from guesswork. Sionna 1.x is **not** an acceptable fallback: it is the TensorFlow implementation,
while this port targets the PyTorch backend introduced in Sionna 2.

### 2.2 New cells `gnn-torch-bootstrap` + `gnn-torch-bundle` — making `gnn_torch.py` importable

* `gnn-torch-bootstrap` searches, in order: the CWD and its two parents, `WORK_DIR`,
  `WORK_DIR/gnn-decoder`, every `gnn_torch.py` under `/kaggle/input/**` (i.e. an attached Kaggle
  dataset or GitHub-imported repo), and `/content/*/gnn_torch.py` (a repo cloned in Colab).
* `gnn-torch-bundle` holds a verbatim copy of `gnn_torch.py` in a raw string and writes it to
  `WORK_DIR` **only if the search found nothing**; it then puts the containing directory on `sys.path`.

*Why a bundled copy:* the repository is private (the GitHub URL returns 404 unauthenticated), so a
`git clone` bootstrap would require a token inside the notebook. Embedding the module makes the single
`.ipynb` file self-sufficient on Kaggle. *Why search first:* the local repo/dataset copy must win, so
that editing `gnn_torch.py` (with `%autoreload 2`) still takes effect and the embedded copy never
silently shadows a newer version.

**Maintenance note:** the embedded copy is a duplicate. After editing `gnn_torch.py`, refresh it —
the copy currently in the notebook was generated from, and verified byte-identical to, the updated
`gnn_torch.py`.

### 2.3 Modified cell `0a65e588` — imports and device report

* Import of `gnn_torch` split across lines (cosmetic only).
* `device = torch.device(sionna.phy.config.device)` kept — Sionna 2's `config.device` setter already
  defaults to `cuda:0` when a GPU is visible and `cpu` otherwise, so this is the correct
  runtime-agnostic selection.
* Added a print of `sionna.__version__`, `torch.__version__`, the device and
  `torch.cuda.get_device_name(...)`, plus an explicit warning when no GPU is found.
  *Why:* the most common Kaggle mistake is leaving the accelerator on "None"; silently training on CPU
  wastes hours. The stale "CPU on macOS" comment was replaced with the accurate statement that Sionna
  does not support Apple MPS.

### 2.4 Modified cell `89b34d4c` — parameters and session budget

Changed inside `params`:

* `"save_dir": str(RESULTS_DIR)` (was `"results/"`) — persisted, absolute, runtime-appropriate.
* `"save_weights_iter": 2000` (was `10000`) — this key was previously *unused*; it now drives real
  checkpointing, and 2000 bounds the work lost to a preemption to a few minutes.
* `"resume": True` — new key consumed by `train_gnn`.

Added after `params`, a `TRAIN_PRESET` switch (`"quick"` | `"medium"` | `"paper"`) that overrides
`train_iter`, `mc_iters` and `num_target_block_errors`:

| Preset | Training iterations | `mc_iters` | `num_target_block_errors` |
| --- | --- | --- | --- |
| `quick` (default) | 3 000 / 5 000 / 2 000 = 10 000 | 20 | 200 |
| `medium` | 20 000 / 50 000 / 30 000 = 100 000 | 50 | 300 |
| `paper` | 35 000 / 300 000 / 300 000 = 635 000 | 100 | 500 |

*Why:* the published 635 000-iteration schedule does not fit in one Kaggle session, so a first run must
be able to complete end-to-end and produce curves. **`quick` will not reproduce the paper's BER curves**
— it is there to validate the pipeline; use `medium`/`paper` across resumed sessions for real results.
The learning-rate values and their three-phase structure are unchanged, so a preset is a truncation of
the published schedule, not a different recipe.

### 2.5 Modified cell `56256283-…` — training

Logic unchanged (`train_gnn(e2e_gnn, params)`); added comments stating that training checkpoints to
`<save_dir>/<run_name>_ckpt.pt` and that re-running the cell after a timeout resumes.

### 2.6 Modified cell `3f518bcd-…` — CSV export

`np.savetxt('results/ldpc_5g_ber.csv', ...)` → `Path(params["save_dir"]) / "ldpc_5g_ber.csv"`.
*Why:* the hardcoded relative path was the one remaining write that ignored `save_dir` and would have
raised `FileNotFoundError` (or written to a non-persisted directory) on a hosted runtime.

### 2.7 Modified cell `386f0ba4` — title markdown

Documents the three supported runtimes, the checked requirements, the pre-run steps (Kaggle: GPU +
**Internet on**, since pip needs network access and Kaggle disables it by default; Colab: GPU runtime),
the possible one-time restart, the harmless `torchvision`/`torchaudio` dependency warnings after the
torch upgrade, and the training-budget situation.

---

## 3. `gnn_torch.py` change — checkpoint + resume in `train_gnn`

Signature: `train_gnn(model, params)` → `train_gnn(model, params, resume=None)`
(`resume=None` falls back to `params.get("resume", True)`, so existing call sites are unaffected).

* Every `params["save_weights_iter"]` steps, writes `{"model", "optimizer", "total"}` to
  `<save_dir>/<run_name>_ckpt.pt`.
  *Why the optimizer state and the iteration counter, not just weights:* Adam's moment estimates would
  otherwise be reset at each resume, producing a loss spike and effectively restarting the schedule;
  and without `total` the phase-dependent learning rate could not be restored.
* On entry, if `resume` and the checkpoint exists, it loads model + optimizer, sets `total`, and prints
  the resume point. `torch.load(..., weights_only=True)` is used — the payload is only tensors, dicts,
  lists and scalars, so the safe loader suffices.
* The phase loop now fast-forwards: a phase whose iterations are already covered by `total` is skipped
  (its length still added to the offset), and the first unfinished phase starts at
  `range(max(0, done - offset), num_iter)`. *Why:* this keeps each phase's learning rate tied to the
  same absolute iteration ranges as an uninterrupted run, so resuming yields the intended schedule
  rather than restarting phase 0 at a stale LR.
* Progress lines now show `Iteration {total}/{num_total_iter}`, and a final line reports where the
  weights were written. *Why:* with a resumable multi-session run, an absolute counter against the
  target is the only readable measure of progress.
* At the end it writes both the rolling checkpoint and, as before,
  `<run_name>_final.pt` (a plain `state_dict`, the format the notebook's `train = False` branch loads).

Unchanged: the loss (mean BCE-with-logits over all decoder iterations), the gradient clipping
(`clip_grad_value_(…, 10.0)`), the Adam optimizer, the training-SNR sampling, and the periodic BER eval.

---

## 4. How to run

**Kaggle**
1. Upload the `.ipynb` (nothing else required — the module is bundled).
2. Settings sidebar: *Accelerator* → GPU (T4/P100), *Internet* → **On**.
3. Run all. The install takes a few minutes; if the cell asks for a restart, restart and re-run it.
4. Results land in `/kaggle/working/results/`; *Save version* to persist them.
5. To continue training in a later session, attach the previous version's output as a dataset — the
   bootstrap search picks up `gnn_torch.py` from `/kaggle/input/**` — and copy
   `LDPC_5G_01_ckpt.pt` into `/kaggle/working/results/` before running the training cell.

**Colab**
1. *Runtime → Change runtime type → GPU*, then run all.
2. Results are in `/content/results/`; mount Drive and copy the checkpoint there if you want it to
   survive the session.

**Locally**: unchanged behaviour — `RESULTS_DIR` becomes `./results` and the install cell is a no-op in
an environment that already has Python 3.11+/torch 2.9.1+/Sionna 2.

---

## 5. Verification performed, and its limits

Checked against the real `sionna-no-rt` 2.0.1 wheel (downloaded and unpacked):

* `Requires-Python >=3.11`, `Requires-Dist: torch>=2.9.1` — the pins used by the install cell.
* `sionna.phy.config.device` exists and its setter auto-selects `cuda:0` / `cpu`.
* `PlotBER.simulate` accepts `compile_mode`, `soft_estimates`, `num_target_block_errors`,
  `max_mc_iter`, `forward_keyboard_interrupt`; `PlotBER.ber` exists.
* `sim_ber` converts a NumPy `ebno_dbs` via `torch.from_numpy`, so `np.arange(...)` remains valid.
* `sionna.phy.utils.metrics.compute_ber`, `LDPC5GDecoder.pcm`, `_nb_pruned_nodes`, `prune_pcm`, and
  `LDPC5GEncoder.{z, k_ldpc, n_ldpc}` all exist as used.

Also checked: every code cell compiles (magics stripped), the notebook is valid nbformat 4.5 JSON with
valid cell ids, and the embedded module string is byte-identical to `gnn_torch.py`.

**Not verified:** the notebook was not executed end-to-end. This machine has no torch/sionna install, so
the resume path, the GPU training loop and the BER curves have not been run — that first happens on
Kaggle/Colab.
