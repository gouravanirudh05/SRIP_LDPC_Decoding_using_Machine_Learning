"""Train and evaluate the hybrid BP + edge-aware GAT decoder.

Example from the GAT+BP directory::

    python train_gat_bp.py --code-type LDPC --code-n 49 --code-k 24 \
        --steps 5000 --eval-every 500 --save-dir Results_GAT_BP

The code database is the same database used by the ECCT/AECCT experiments.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch

from Codes import Get_Generator_and_Parity
from gat_bp import HybridBPGAT, bce_from_llr, hard_decode


def configure_logger(log_file: Path) -> logging.Logger:
    """Log identical plain-text records to both the console and a file."""
    logger = logging.getLogger("gat_bp_training")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    formatter = logging.Formatter("%(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(log_file, mode="w")
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ebn0_to_sigma(ebn0_db: torch.Tensor, rate: float) -> torch.Tensor:
    """AWGN standard deviation for BPSK with unit-energy symbols."""
    snr_db = ebn0_db + 10.0 * torch.log10(torch.tensor(2.0 * rate, device=ebn0_db.device))
    return torch.sqrt(1.0 / (10.0 ** (snr_db / 10.0)))


def sample_channel_batch(
    generator: torch.Tensor,
    batch_size: int,
    ebn0_db: torch.Tensor | float,
    device: torch.device,
    zero_codeword: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate random codewords, BPSK AWGN samples, and channel LLRs."""
    generator = generator.to(device=device, dtype=torch.float32)
    k, n = generator.shape
    rate = k / n
    if isinstance(ebn0_db, torch.Tensor):
        ebn0 = ebn0_db.to(device=device, dtype=torch.float32).reshape(-1)
        if ebn0.numel() == 1:
            ebn0 = ebn0.expand(batch_size)
        if ebn0.numel() != batch_size:
            raise ValueError("ebn0_db must be scalar or have one value per sample.")
    else:
        ebn0 = torch.full((batch_size,), float(ebn0_db), device=device)

    if zero_codeword:
        messages = torch.zeros((batch_size, k), device=device)
    else:
        messages = torch.randint(0, 2, (batch_size, k), device=device).float()
    codewords = torch.remainder(messages @ generator, 2.0)
    symbols = 1.0 - 2.0 * codewords
    sigma = ebn0_to_sigma(ebn0, rate).unsqueeze(1)
    received = symbols + sigma * torch.randn_like(symbols)
    channel_llr = 2.0 * received / sigma.square()
    return channel_llr, codewords.long(), messages.long(), ebn0


def error_counts(predicted: torch.Tensor, target: torch.Tensor) -> Tuple[int, int, int]:
    bit_errors = int((predicted != target).sum().item())
    frame_errors = int((predicted != target).any(dim=1).sum().item())
    return bit_errors, frame_errors, int(target.numel())


@torch.no_grad()
def evaluate(
    model: HybridBPGAT,
    generator: torch.Tensor,
    ebn0_values: Iterable[float],
    batches: int,
    batch_size: int,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    model.eval()
    result: Dict[str, Dict[str, float]] = {}
    for ebn0 in ebn0_values:
        final_bit_errors = final_frame_errors = 0
        bp_bit_errors = bp_frame_errors = 0
        total_bits = total_frames = 0
        for _ in range(batches):
            llr, target, _, _ = sample_channel_batch(
                generator, batch_size, ebn0, device
            )
            final_llr, aux = model(llr, return_aux=True)
            final_pred = hard_decode(final_llr)
            bp_pred = hard_decode(aux["bp_llr"])
            bit_errors, frame_errors, bits = error_counts(final_pred, target)
            final_bit_errors += bit_errors
            final_frame_errors += frame_errors
            bit_errors, frame_errors, _ = error_counts(bp_pred, target)
            bp_bit_errors += bit_errors
            bp_frame_errors += frame_errors
            total_bits += bits
            total_frames += target.shape[0]
        result[f"{ebn0:g}"] = {
            "bp_ber": bp_bit_errors / max(total_bits, 1),
            "bp_fer": bp_frame_errors / max(total_frames, 1),
            "gat_bp_ber": final_bit_errors / max(total_bits, 1),
            "gat_bp_fer": final_frame_errors / max(total_frames, 1),
            "frames": float(total_frames),
        }
    return result


def build_code(args: argparse.Namespace) -> Tuple[torch.Tensor, torch.Tensor]:
    class Code:
        pass

    code = Code()
    code.code_type = args.code_type
    code.n = args.code_n
    code.k = args.code_k
    generator, parity_check = Get_Generator_and_Parity(code, standard_form=False)
    generator_tensor = torch.from_numpy(generator).float()
    parity_check_tensor = torch.from_numpy(parity_check).float()
    if generator_tensor.shape != (args.code_k, args.code_n):
        raise ValueError(
            f"Expected generator shape {(args.code_k, args.code_n)}, got {tuple(generator_tensor.shape)}."
        )
    if parity_check_tensor.shape != (parity_check_tensor.shape[0], args.code_n):
        raise ValueError("Parity-check matrix has an unexpected number of columns.")
    if not torch.all(torch.remainder(generator_tensor @ parity_check_tensor.t(), 2.0) == 0):
        raise ValueError("The generator and parity-check matrices are inconsistent.")
    return generator_tensor, parity_check_tensor


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    generator, parity_check = build_code(args)
    model = HybridBPGAT(
        parity_check,
        bp_iterations=args.bp_iterations,
        bp_damping=args.bp_damping,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        head_dim=args.head_dim,
        dropout=args.dropout,
    ).to(device)
    generator = generator.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.steps, 1), eta_min=args.min_lr
    )
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_file = Path(args.log_file) if args.log_file else save_dir / "training.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(log_file)
    logger.info(f"device={device}")
    logger.info(f"parameters={sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    logger.info(f"code={args.code_type} (n={args.code_n}, k={args.code_k})")
    logger.info(f"config={vars(args)}")

    best_loss = float("inf")
    started = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        ebn0 = torch.empty(args.batch_size, device=device).uniform_(args.train_ebn0_min, args.train_ebn0_max)
        channel_llr, target, _, _ = sample_channel_batch(
            generator, args.batch_size, ebn0, device, zero_codeword=args.zero_codeword
        )
        final_llr = model(channel_llr)
        loss = bce_from_llr(final_llr, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        if step == 1 or step % args.log_every == 0:
            with torch.no_grad():
                train_pred = hard_decode(final_llr)
                ber = (train_pred != target).float().mean().item()
                fer = (train_pred != target).any(dim=1).float().mean().item()
            elapsed = time.time() - started
            logger.info(
                f"step={step:6d} loss={loss.item():.4e} ber={ber:.4e} fer={fer:.4e} "
                f"lr={scheduler.get_last_lr()[0]:.2e} time={elapsed:.1f}s"
            )

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": vars(args),
                    "parity_check": parity_check.cpu(),
                    "generator": generator.cpu(),
                    "best_loss": best_loss,
                },
                save_dir / "best_model.pt",
            )

        if args.eval_every > 0 and (step % args.eval_every == 0 or step == args.steps):
            metrics = evaluate(
                model,
                generator,
                args.eval_ebn0,
                args.eval_batches,
                args.eval_batch_size,
                device,
            )
            logger.info(json.dumps({"step": step, "evaluation": metrics}, sort_keys=True))

    logger.info(f"best_loss={best_loss:.4e}")
    logger.info(f"saved={save_dir / 'best_model.pt'}")
    logger.info(f"log_file={log_file}")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-type", choices=["BCH", "POLAR", "LDPC", "CCSDS", "MACKAY"], default="LDPC")
    parser.add_argument("--code-n", type=int, default=49)
    parser.add_argument("--code-k", type=int, default=24)
    parser.add_argument("--bp-iterations", type=int, default=5)
    parser.add_argument("--bp-damping", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--train-ebn0-min", type=float, default=2.0)
    parser.add_argument("--train-ebn0-max", type=float, default=6.0)
    parser.add_argument("--eval-ebn0", type=float, nargs="+", default=[2.0, 3.0, 4.0, 5.0, 6.0])
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--zero-codeword", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="", help="cuda, cpu, or empty for automatic selection")
    parser.add_argument("--save-dir", default="Results_GAT_BP")
    parser.add_argument("--log-file", default="", help="Log path; defaults to <save-dir>/training.log")
    return parser


if __name__ == "__main__":
    train(make_parser().parse_args())
