"""PyTorch/Sionna 2 implementation of the GNN LDPC-decoder helpers.

This module is deliberately separate from :mod:`gnn`, which is retained for
the original TensorFlow/Sionna 0.x notebooks.
"""

from __future__ import annotations

from pathlib import Path
from time import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from sionna.phy import config
from sionna.phy.fec.ldpc import LDPC5GDecoder
from sionna.phy.utils import ebnodb2no
from sionna.phy.utils.metrics import compute_ber


def _mlp(in_features, hidden_features, out_features, num_layers, activation,
         use_bias):
    layers = []
    for i in range(num_layers):
        in_dim = in_features if i == 0 else hidden_features
        out_dim = out_features if i == num_layers - 1 else hidden_features
        layers.append(nn.Linear(in_dim, out_dim, bias=use_bias))
        if i != num_layers - 1:
            layers.append(nn.ReLU() if activation == "relu" else nn.Tanh())
    return nn.Sequential(*layers)


class UpdateEmbeddings(nn.Module):
    """One directed message-passing update over an LDPC Tanner graph."""

    def __init__(self, num_embed_dims, num_msg_dims, num_hidden_units,
                 num_mlp_layers, from_to_ind, num_targets, reduce_op,
                 activation, use_attributes, node_attribute_dims,
                 msg_attribute_dims, use_bias):
        super().__init__()
        self.reduce_op = reduce_op
        self.num_targets = num_targets
        self.register_buffer("from_ind", torch.as_tensor(from_to_ind[:, 0],
                                                          dtype=torch.long))
        self.register_buffer("to_ind", torch.as_tensor(from_to_ind[:, 1],
                                                        dtype=torch.long))
        self.msg_mlp = _mlp(2 * num_embed_dims + (msg_attribute_dims if use_attributes else 0),
                            num_hidden_units, num_msg_dims, num_mlp_layers,
                            activation, use_bias)
        self.embed_mlp = _mlp(num_msg_dims + num_embed_dims +
                              (node_attribute_dims if use_attributes else 0),
                              num_hidden_units, num_embed_dims,
                              num_mlp_layers, activation, use_bias)
        if use_attributes:
            self.node_attributes = nn.Parameter(torch.zeros(num_targets,
                                                             node_attribute_dims))
            self.msg_attributes = nn.Parameter(torch.zeros(len(from_to_ind),
                                                            msg_attribute_dims))
        else:
            self.node_attributes = None
            self.msg_attributes = None

    def forward(self, h_from, h_to):
        features = torch.cat((h_from[:, self.from_ind], h_to[:, self.to_ind]),
                             dim=-1)
        if self.msg_attributes is not None:
            features = torch.cat((features, self.msg_attributes.expand(
                features.shape[0], -1, -1)), dim=-1)
        messages = self.msg_mlp(features)
        batch_size, _, msg_dims = messages.shape
        index = self.to_ind.view(1, -1, 1).expand(batch_size, -1, msg_dims)
        if self.reduce_op in ("sum", "mean"):
            aggregate = torch.zeros(batch_size, self.num_targets, msg_dims,
                                    dtype=messages.dtype, device=messages.device)
            aggregate.scatter_add_(1, index, messages)
            if self.reduce_op == "mean":
                counts = torch.bincount(self.to_ind,
                                        minlength=self.num_targets).clamp_min(1)
                aggregate = aggregate / counts.view(1, -1, 1)
        elif self.reduce_op in ("max", "min"):
            fill = -torch.inf if self.reduce_op == "max" else torch.inf
            aggregate = torch.full((batch_size, self.num_targets, msg_dims), fill,
                                   dtype=messages.dtype, device=messages.device)
            aggregate.scatter_reduce_(1, index, messages,
                                      reduce="amax" if self.reduce_op == "max" else "amin",
                                      include_self=True)
        else:
            raise ValueError(f"Unknown reduce operation: {self.reduce_op}")
        if self.node_attributes is not None:
            aggregate = torch.cat((aggregate, self.node_attributes.expand(
                batch_size, -1, -1)), dim=-1)
        return self.embed_mlp(torch.cat((aggregate, h_to), dim=-1))


class GNNBP(nn.Module):
    """Graph-neural-network belief-propagation decoder implemented in PyTorch."""

    def __init__(self, pcm, num_embed_dims, num_msg_dims, num_hidden_units,
                 num_mlp_layers, num_iter, reduce_op="mean", activation="tanh",
                 output_all_iter=False, clip_llr_to=None, use_attributes=False,
                 node_attribute_dims=0, msg_attribute_dims=0, use_bias=False):
        super().__init__()
        pcm = np.asarray(pcm, dtype=np.int8)
        edges = np.stack(np.where(pcm), axis=1)
        self.num_cn, self.num_vn = pcm.shape
        self.num_embed_dims = num_embed_dims
        self.num_iter = num_iter
        self.output_all_iter = output_all_iter
        self.clip_llr_to = clip_llr_to
        self.llr_embed = nn.Linear(1, num_embed_dims, bias=use_bias)
        self.llr_inv_embed = nn.Linear(num_embed_dims, 1, bias=use_bias)
        kwargs = dict(num_embed_dims=num_embed_dims, num_msg_dims=num_msg_dims,
                      num_hidden_units=num_hidden_units, num_mlp_layers=num_mlp_layers,
                      reduce_op=reduce_op, activation=activation,
                      use_attributes=use_attributes,
                      node_attribute_dims=node_attribute_dims,
                      msg_attribute_dims=msg_attribute_dims, use_bias=use_bias)
        self.update_h_cn = UpdateEmbeddings(from_to_ind=np.flip(edges, 1),
                                            num_targets=self.num_cn, **kwargs)
        self.update_h_vn = UpdateEmbeddings(from_to_ind=edges,
                                            num_targets=self.num_vn, **kwargs)

    def forward(self, llr):
        if self.clip_llr_to is not None:
            llr = llr.clamp(-self.clip_llr_to, self.clip_llr_to)
        h_vn = self.llr_embed(llr.unsqueeze(-1))
        h_cn = torch.zeros(llr.shape[0], self.num_cn, self.num_embed_dims,
                           device=llr.device, dtype=llr.dtype)
        outputs = []
        for _ in range(self.num_iter):
            h_cn = self.update_h_cn(h_vn, h_cn)
            h_vn = self.update_h_vn(h_cn, h_vn)
            if self.output_all_iter:
                outputs.append(self.llr_inv_embed(h_vn).squeeze(-1))
        return outputs if self.output_all_iter else self.llr_inv_embed(h_vn).squeeze(-1)


class LinearEncoder(nn.Module):
    """Binary linear encoder defined by a systematic generator matrix."""

    def __init__(self, generator_matrix):
        super().__init__()
        self.register_buffer("generator_matrix", torch.as_tensor(generator_matrix,
                                                                    dtype=torch.float32))

    def forward(self, bits):
        return torch.remainder(bits @ self.generator_matrix, 2.0)


def generate_pruned_pcm_5g(decoder, n, verbose=True):
    """Return the 5G PCM after rate-matching and shortened-bit pruning."""
    enc = decoder.encoder
    pos_punc = np.concatenate((np.zeros(2 * enc.z), np.ones(n)))
    k_short = enc.k_ldpc - enc.k
    num_punc_bits = (enc.n_ldpc - k_short) - enc.n - 2 * enc.z
    pos_punc2 = np.concatenate((pos_punc, np.zeros(
        num_punc_bits - decoder._nb_pruned_nodes)))
    num_par_bits = enc.n_ldpc - k_short - enc.k - decoder._nb_pruned_nodes
    pattern = np.concatenate((pos_punc2[:enc.k], 2 * np.ones(k_short),
                              pos_punc2[enc.k:enc.k + num_par_bits]))
    shortened = np.where(pattern == 2)[0]
    pcm = np.asarray(decoder.pcm.todense())
    kept = np.setdiff1d(np.arange(pcm.shape[1]), shortened)
    pcm = pcm[:, kept]
    if verbose:
        print(f"Pruned 5G PCM: {pcm.shape}; shortened bits: {len(shortened)}")
    return pcm, pattern[kept]


class LDPC5GGNN(GNNBP):
    """GNN decoder with 5G LDPC rate recovery/restoration."""

    def __init__(self, encoder, *args, return_infobits=False, **kwargs):
        reference_decoder = LDPC5GDecoder(encoder, prune_pcm=True,
                                           return_infobits=False)
        pcm, pattern = generate_pruned_pcm_5g(reference_decoder, encoder.n,
                                               verbose=False)
        sentinel = int(np.sum(pattern == 1))
        gather_ind = np.full(len(pattern), sentinel, dtype=np.int64)
        gather_ind[np.where(pattern == 1)[0]] = np.arange(np.sum(pattern == 1))
        super().__init__(pcm, *args, **kwargs)
        self.encoder = encoder
        self.return_infobits = return_infobits
        self.llr_max = 20.0
        self.register_buffer("rm_ind", torch.as_tensor(gather_ind, dtype=torch.long))
        self.register_buffer("rm_inv_ind", torch.as_tensor(np.where(pattern == 1)[0],
                                                              dtype=torch.long))

    def forward(self, llr):
        llr_in = torch.cat((llr, torch.zeros_like(llr[:, :1])), dim=1)
        decoded = super().forward(llr_in[:, self.rm_ind])
        def restore(x):
            return x[:, :self.encoder.k] if self.return_infobits else x[:, self.rm_inv_ind]
        return [restore(x) for x in decoded] if self.output_all_iter else restore(decoded)


class E2EModel(nn.Module):
    """AWGN link model with QPSK-equivalent soft information for Sionna 2."""

    def __init__(self, encoder, decoder, k, n, return_infobits=False, es_no=False,
                 device=None):
        super().__init__()
        self.encoder, self.decoder = encoder, decoder
        self.k, self.n = k, n
        self.return_infobits, self.es_no = return_infobits, es_no
        # Sionna PHY 2.0 currently supports CPU and CUDA devices. Keep this
        # explicit rather than selecting Apple's MPS backend, which Sionna does
        # not register as a supported device.
        self.device_ = torch.device(device or config.device)
        self.to(self.device_)

    def forward(self, batch_size, ebno_db):
        ebno_db = torch.as_tensor(ebno_db, dtype=torch.float32, device=self.device_)
        coderate = 1.0 if self.decoder is None or self.es_no else self.k / self.n
        no = ebnodb2no(ebno_db, num_bits_per_symbol=2, coderate=coderate)
        bits = torch.randint(0, 2, (batch_size, self.k), device=self.device_, dtype=torch.float32)
        codeword = self.encoder(bits) if self.encoder is not None else bits
        # Equivalent QPSK bit channel. `no` is the complex noise variance
        # from ebnodb2no(): each real component has variance no/2 and the
        # QPSK component amplitude is 1/sqrt(2). Sionna decoders use logits
        # log(p(bit=1)/p(bit=0)).
        logits = (2.0 * (2.0 * codeword - 1.0) / no
                  + torch.randn_like(codeword) * (2.0 / torch.sqrt(no)))
        estimates = self.decoder(logits) if self.decoder is not None else logits
        return (bits if self.return_infobits else codeword), estimates


def transfer_gnn_weights(source, target):
    """Copy only compatible trainable GNN parameters between Tanner graphs.

    The graph connectivity buffers necessarily differ when a decoder is used
    for a different 5G codeword length. They must not be copied.
    """
    source_params = dict(source.named_parameters())
    target_params = dict(target.named_parameters())
    compatible = {
        name: value for name, value in source_params.items()
        if name in target_params and target_params[name].shape == value.shape
    }
    target.load_state_dict(compatible, strict=False)
    skipped = sorted(set(source_params) - set(compatible))
    if skipped:
        print("Skipped incompatible trainable parameters:", ", ".join(skipped))


def train_gnn(model, params):
    """Train a GNN decoder; saves a PyTorch checkpoint, not a TF `.npy` file."""
    Path(params["save_dir"]).mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"][0])
    total = 0
    start = time()
    for phase, batch_size in enumerate(params["batch_size"]):
        for group in optimizer.param_groups:
            group["lr"] = params["learning_rate"][phase]
        for _ in range(params["train_iter"][phase]):
            total += 1
            optimizer.zero_grad()
            ebno = torch.empty(batch_size, 1, device=model.device_).uniform_(
                params["ebno_db_train"][0], params["ebno_db_train"][1])
            codeword, outputs = model(batch_size, ebno)
            loss = torch.stack([F.binary_cross_entropy_with_logits(x, codeword)
                                for x in outputs]).mean()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 10.0)
            optimizer.step()
            if total % params["eval_train_steps"] == 0:
                with torch.no_grad():
                    target, out = model(params["batch_size_eval"], params["ebno_db_eval"])
                    ber = compute_ber(target, (out[-1] > 0).to(target.dtype)).item()
                print(f"Iteration {total}: loss={loss.item():.3f}, ber={ber:.5f}, "
                      f"duration={time()-start:.1f}s")
                start = time()
    torch.save(model.state_dict(), Path(params["save_dir"]) /
               f"{params['run_name']}_final.pt")
