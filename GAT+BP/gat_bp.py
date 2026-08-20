"""Hybrid differentiable BP + edge-aware GAT decoder.

The BP stage is a conventional sum-product decoder on the Tanner graph.  The
GAT stage consumes the channel LLR, BP posterior, and BP edge messages and
learns a residual correction to the BP posterior.  No PyTorch-Geometric
dependency is required; the graph is represented by the non-zero entries of
the parity-check matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class TannerGraph:
    """Static edge layout for a binary parity-check matrix."""

    n_variables: int
    n_checks: int
    edge_variables: Tensor
    edge_checks: Tensor
    check_order: Tensor
    inverse_check_order: Tensor
    variable_slices: List[Tuple[int, int]]
    check_slices: List[Tuple[int, int]]
    check_degrees: Tensor

    @classmethod
    def from_parity_check(cls, parity_check: np.ndarray | Tensor) -> "TannerGraph":
        if isinstance(parity_check, Tensor):
            h = parity_check.detach().cpu().numpy()
        else:
            h = np.asarray(parity_check)
        if h.ndim != 2:
            raise ValueError("The parity-check matrix must be two-dimensional.")
        h = (h != 0).astype(np.uint8)
        n_checks, n_variables = h.shape
        raw_checks, raw_variables = np.nonzero(h)
        if len(raw_checks) == 0:
            raise ValueError("The parity-check matrix has no Tanner-graph edges.")
        if np.any(np.bincount(raw_variables, minlength=n_variables) == 0):
            raise ValueError("Every variable node must participate in a check.")
        if np.any(np.bincount(raw_checks, minlength=n_checks) == 0):
            raise ValueError("Every check node must contain at least one variable.")

        # Keep the canonical edge order grouped by variable for GAT aggregation.
        variable_order = np.lexsort((raw_checks, raw_variables))
        edge_variables = raw_variables[variable_order]
        edge_checks = raw_checks[variable_order]

        # BP check updates use the same edges grouped by check.  These two
        # permutations avoid scattering differentiable messages in-place.
        check_order = np.lexsort((edge_variables, edge_checks))
        inverse_check_order = np.argsort(check_order)

        variable_slices = _contiguous_slices(edge_variables, n_variables)
        check_sorted_checks = edge_checks[check_order]
        check_slices = _contiguous_slices(check_sorted_checks, n_checks)
        check_degrees = np.bincount(edge_checks, minlength=n_checks).astype(np.float32)

        return cls(
            n_variables=n_variables,
            n_checks=n_checks,
            edge_variables=torch.as_tensor(edge_variables, dtype=torch.long),
            edge_checks=torch.as_tensor(edge_checks, dtype=torch.long),
            check_order=torch.as_tensor(check_order, dtype=torch.long),
            inverse_check_order=torch.as_tensor(inverse_check_order, dtype=torch.long),
            variable_slices=variable_slices,
            check_slices=check_slices,
            check_degrees=torch.as_tensor(check_degrees, dtype=torch.float32),
        )

    @property
    def n_edges(self) -> int:
        return int(self.edge_variables.numel())

    def mean_by_check(self, edge_values: Tensor) -> Tensor:
        """Average edge values into check-node features."""
        ordered = edge_values.index_select(1, self.check_order.to(edge_values.device))
        chunks = [ordered[:, start:end].mean(dim=1) for start, end in self.check_slices]
        return torch.stack(chunks, dim=1)


def _contiguous_slices(labels: np.ndarray, count: int) -> List[Tuple[int, int]]:
    slices: List[Tuple[int, int]] = []
    start = 0
    for label in range(count):
        positions = np.flatnonzero(labels == label)
        if len(positions) == 0:
            raise ValueError(f"Node {label} has no Tanner-graph edges.")
        end = int(positions[-1]) + 1
        slices.append((start, end))
        start = end
    if start != len(labels):
        raise RuntimeError("Tanner-graph edge layout is not contiguous.")
    return slices


def _atanh_clamped(x: Tensor, eps: float = 1e-6) -> Tensor:
    x = x.clamp(-1.0 + eps, 1.0 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


class SumProductBP(nn.Module):
    """Differentiable flooding-schedule sum-product BP on a Tanner graph."""

    def __init__(self, graph: TannerGraph, iterations: int = 5, damping: float = 0.0):
        super().__init__()
        if iterations < 0:
            raise ValueError("BP iterations must be non-negative.")
        if not 0.0 <= damping < 1.0:
            raise ValueError("BP damping must be in [0, 1).")
        self.graph = graph
        self.iterations = int(iterations)
        self.damping = float(damping)
        self.register_buffer("edge_variables", graph.edge_variables)
        self.register_buffer("check_order", graph.check_order)
        self.register_buffer("inverse_check_order", graph.inverse_check_order)

    def _check_to_variable(self, variable_to_check: Tensor) -> Tensor:
        # Work in check-grouped order for the check update.
        ordered = variable_to_check.index_select(1, self.check_order)
        tanh_messages = torch.tanh(0.5 * ordered)
        check_messages: List[Tensor] = []

        for start, end in self.graph.check_slices:
            local = tanh_messages[:, start:end]
            degree = end - start
            if degree == 1:
                excluded = torch.ones_like(local)
            else:
                left = torch.cumprod(local, dim=1)
                right = torch.cumprod(local.flip(dims=(1,)), dim=1).flip(dims=(1,))
                first = right[:, 1:2]
                middle = left[:, :-2] * right[:, 2:]
                last = left[:, -2:-1]
                excluded = torch.cat((first, middle, last), dim=1)
            check_messages.append(2.0 * _atanh_clamped(excluded))

        ordered_messages = torch.cat(check_messages, dim=1)
        return ordered_messages.index_select(1, self.inverse_check_order)

    def forward(
        self, channel_llr: Tensor, return_messages: bool = False
    ) -> Tensor | Tuple[Tensor, Tensor, Tensor]:
        if channel_llr.ndim != 2 or channel_llr.shape[1] != self.graph.n_variables:
            raise ValueError(
                f"channel_llr must have shape [batch, {self.graph.n_variables}]."
            )

        variable_to_check = channel_llr.index_select(1, self.edge_variables)
        check_to_variable = torch.zeros_like(variable_to_check)

        for _ in range(self.iterations):
            new_check_to_variable = self._check_to_variable(variable_to_check)
            if self.damping:
                check_to_variable = (
                    self.damping * check_to_variable
                    + (1.0 - self.damping) * new_check_to_variable
                )
            else:
                check_to_variable = new_check_to_variable

            incoming = torch.zeros_like(channel_llr)
            incoming.scatter_add_(
                1,
                self.edge_variables.unsqueeze(0).expand(channel_llr.shape[0], -1),
                check_to_variable,
            )
            variable_to_check = (
                channel_llr.index_select(1, self.edge_variables)
                + incoming.index_select(1, self.edge_variables)
                - check_to_variable
            )

        posterior = channel_llr.clone()
        posterior.scatter_add_(
            1,
            self.edge_variables.unsqueeze(0).expand(channel_llr.shape[0], -1),
            check_to_variable,
        )
        if return_messages:
            return posterior, variable_to_check, check_to_variable
        return posterior


class EdgeAwareGAT(nn.Module):
    """One-hop check-to-variable GAT with BP edge messages as edge features."""

    def __init__(
        self,
        graph: TannerGraph,
        variable_dim: int = 4,
        check_dim: int = 5,
        edge_dim: int = 4,
        hidden_dim: int = 32,
        heads: int = 2,
        head_dim: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        if heads < 1 or head_dim < 1:
            raise ValueError("heads and head_dim must be positive.")
        self.graph = graph
        self.heads = heads
        self.head_dim = head_dim
        self.dropout = float(dropout)
        projected_dim = heads * head_dim

        self.query = nn.Linear(variable_dim, projected_dim, bias=False)
        self.key = nn.Linear(check_dim, projected_dim, bias=False)
        self.value = nn.Linear(check_dim, projected_dim, bias=False)
        self.edge_key = nn.Linear(edge_dim, projected_dim, bias=False)
        self.edge_value = nn.Linear(edge_dim, projected_dim, bias=False)
        self.attn_query = nn.Parameter(torch.empty(heads, head_dim))
        self.attn_key = nn.Parameter(torch.empty(heads, head_dim))
        self.attn_edge = nn.Parameter(torch.empty(heads, head_dim))
        self.update = nn.Sequential(
            nn.Linear(variable_dim + projected_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.xavier_uniform_(self.edge_key.weight)
        nn.init.xavier_uniform_(self.edge_value.weight)
        nn.init.xavier_uniform_(self.attn_query.unsqueeze(0))
        nn.init.xavier_uniform_(self.attn_key.unsqueeze(0))
        nn.init.xavier_uniform_(self.attn_edge.unsqueeze(0))

    def forward(
        self,
        variable_features: Tensor,
        check_features: Tensor,
        edge_features: Tensor,
        return_attention: bool = False,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        batch_size = variable_features.shape[0]
        n_checks = check_features.shape[1]
        projected_variables = self.query(variable_features).view(
            batch_size, -1, self.heads, self.head_dim
        )
        projected_checks = self.key(check_features).view(
            batch_size, n_checks, self.heads, self.head_dim
        )
        values = self.value(check_features).view(
            batch_size, n_checks, self.heads, self.head_dim
        )
        edge_keys = self.edge_key(edge_features).view(
            batch_size, -1, self.heads, self.head_dim
        )
        edge_values = self.edge_value(edge_features).view(
            batch_size, -1, self.heads, self.head_dim
        )

        edge_variables = self.graph.edge_variables.to(variable_features.device)
        edge_checks = self.graph.edge_checks.to(variable_features.device)
        query_edges = projected_variables.index_select(1, edge_variables)
        key_edges = projected_checks.index_select(1, edge_checks)
        value_edges = values.index_select(1, edge_checks) + edge_values
        logits = (
            query_edges * self.attn_query
            + key_edges * self.attn_key
            + edge_keys * self.attn_edge
        ).sum(dim=-1) / (self.head_dim**0.5)
        logits = F.leaky_relu(logits, negative_slope=0.2)

        # Each variable node normalizes over its incident check nodes.
        attention = torch.cat(
            [F.softmax(logits[:, start:end], dim=1) for start, end in self.graph.variable_slices],
            dim=1,
        )
        attention = F.dropout(attention, p=self.dropout, training=self.training)
        aggregated = torch.stack(
            [
                (attention[:, start:end].unsqueeze(-1) * value_edges[:, start:end]).sum(dim=1)
                for start, end in self.graph.variable_slices
            ],
            dim=1,
        ).reshape(batch_size, self.graph.n_variables, -1)
        updated = self.update(torch.cat((variable_features, aggregated), dim=-1))
        if return_attention:
            return updated, attention
        return updated


class HybridBPGAT(nn.Module):
    """BP followed by an edge-aware GAT residual correction."""

    def __init__(
        self,
        parity_check: np.ndarray | Tensor,
        bp_iterations: int = 5,
        bp_damping: float = 0.0,
        hidden_dim: int = 32,
        heads: int = 2,
        head_dim: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        graph = TannerGraph.from_parity_check(parity_check)
        self.graph = graph
        self.bp = SumProductBP(graph, iterations=bp_iterations, damping=bp_damping)
        self.gat = EdgeAwareGAT(
            graph,
            variable_dim=4,
            check_dim=5,
            edge_dim=4,
            hidden_dim=hidden_dim,
            heads=heads,
            head_dim=head_dim,
            dropout=dropout,
        )
        self.correction = nn.Sequential(
            nn.Linear(4 + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        # Start exactly at the BP decoder.  The refinement is learned as a
        # residual and cannot damage the initial BP solution at initialization.
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def _features(
        self,
        channel_llr: Tensor,
        bp_llr: Tensor,
        variable_to_check: Tensor,
        check_to_variable: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        variable_features = torch.stack(
            (
                channel_llr,
                bp_llr,
                bp_llr - channel_llr,
                bp_llr.abs(),
            ),
            dim=-1,
        )
        check_v2c = self.graph.mean_by_check(variable_to_check)
        check_c2v = self.graph.mean_by_check(check_to_variable)
        check_features = torch.cat(
            (
                check_v2c.unsqueeze(-1),
                check_c2v.unsqueeze(-1),
                check_v2c.abs().unsqueeze(-1),
                check_c2v.abs().unsqueeze(-1),
                (
                    self.graph.check_degrees.to(channel_llr.device)
                    / self.graph.check_degrees.max().clamp_min(1.0).to(channel_llr.device)
                )
                .to(channel_llr.device)
                .view(1, -1, 1)
                .expand(channel_llr.shape[0], -1, -1),
            ),
            dim=-1,
        )
        edge_features = torch.stack(
            (
                variable_to_check,
                check_to_variable,
                variable_to_check.abs(),
                check_to_variable.abs(),
            ),
            dim=-1,
        )
        return variable_features, check_features, edge_features

    def forward(
        self, channel_llr: Tensor, return_aux: bool = False
    ) -> Tensor | Tuple[Tensor, Dict[str, Tensor]]:
        bp_llr, variable_to_check, check_to_variable = self.bp(
            channel_llr, return_messages=True
        )
        variable_features, check_features, edge_features = self._features(
            channel_llr, bp_llr, variable_to_check, check_to_variable
        )
        gat_hidden, attention = self.gat(
            variable_features, check_features, edge_features, return_attention=True
        )
        correction = self.correction(torch.cat((variable_features, gat_hidden), dim=-1)).squeeze(-1)
        final_llr = bp_llr + correction
        if return_aux:
            return final_llr, {
                "bp_llr": bp_llr,
                "correction": correction,
                "attention": attention,
                "variable_to_check": variable_to_check,
                "check_to_variable": check_to_variable,
            }
        return final_llr


def hard_decode(llr: Tensor) -> Tensor:
    """Map LLRs (positive means bit 0) to binary hard decisions."""
    return (llr < 0).to(torch.long)


def bce_from_llr(llr: Tensor, target_bits: Tensor) -> Tensor:
    """Binary cross entropy where target bit 1 corresponds to negative LLR."""
    return F.binary_cross_entropy_with_logits(-llr, target_bits.float())
