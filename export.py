"""
This script has functions and utilities for model export.

Input:
- HuggingFace weights for any unquantized Qwen3-architecture model

Output:
- Checkpoint file (quantized to Q8_0) compatible with qwen3.c
"""

import argparse
import gzip
import json
import math
import os
import shutil
import struct
from pathlib import Path

import numpy as np
import torch
from torch import nn

from model import ModelArgs, Transformer

import json
from jinja2 import Template

# -----------------------------------------------------------------------------
# common utilities


def serialize_fp32(file, tensor):
    """writes one fp32 tensor to file that is open in wb mode"""
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f"{len(d)}f", *d)
    file.write(b)


def serialize_int8(file, tensor):
    """writes one int8 tensor to file that is open in wb mode"""
    d = tensor.detach().cpu().view(-1).numpy().astype(np.int8)
    b = struct.pack(f"{len(d)}b", *d)
    file.write(b)


def quantize_q80(w, group_size):
    """
    takes a tensor and returns the Q8_0 quantized version
    i.e. symmetric quantization into int8, range [-127,127]
    """
    assert w.numel() % group_size == 0
    ori_shape = w.shape
    w = w.float()  # convert to float32
    w = w.reshape(-1, group_size)
    # find the max in each group
    wmax = torch.abs(w).max(dim=1).values
    # calculate the scaling factor such that float = quant * scale
    scale = wmax / 127.0
    # scale into range [-127, 127]
    quant = w / scale[:, None]
    # round to nearest integer
    int8val = torch.round(quant).to(torch.int8)
    # dequantize by rescaling
    fp32val = (int8val.float() * scale[:, None]).view(-1)
    fp32valr = fp32val.reshape(-1, group_size)
    # calculate the max error in each group
    err = torch.abs(fp32valr - w).max(dim=1).values
    # find the max error across all groups
    maxerr = err.max().item()
    return int8val, scale, maxerr


def model_export(model, filepath, group_size=64):
    """
    Export the model weights in Q8_0 into .bin file to be read from C.
    That is:
    - quantize all weights to symmetric int8, in range [-127, 127]
    - all other tensors (the rmsnorm params) are kept and exported in fp32
    - quantization is done in groups of group_size to reduce the effects of any outliers
    """
    version = 1

    # let's first do some validation for this export type
    while model.params.dim % group_size != 0:
        group_size //= 2
        print(f"BACKOFF: reducing group size to {group_size} to fit hidden_dim")
    weights = [
        model.tok_embeddings.weight,
        *[layer.attention.wq.weight for layer in model.layers],
        *[layer.attention.wk.weight for layer in model.layers],
        *[layer.attention.wv.weight for layer in model.layers],
        *[layer.attention.wo.weight for layer in model.layers],
        *[layer.feed_forward.w1.weight for layer in model.layers],
        *[layer.feed_forward.w2.weight for layer in model.layers],
        *[layer.feed_forward.w3.weight for layer in model.layers],
    ]
    shared_classifier = torch.equal(model.tok_embeddings.weight, model.output.weight)

    if not shared_classifier:
        weights.append(model.output.weight)
    for w in weights:
        assert (
            w.numel() % group_size == 0
        ), f"weight {i} has numel {w.numel()}, not a multiple of group_size {group_size}"

    # write
    out_file = open(filepath, "wb")
    # first write out the header. the header will be 256 bytes
    # 1) write magic, which will be uint32 of "ajc1" in ASCII
    out_file.write(struct.pack("I", 0x616A6331))
    # 2) write version, which will be int
    out_file.write(struct.pack("i", version))
    # 3) write the params, which will be 7 ints
    p = model.params
    hidden_dim = model.layers[0].feed_forward.w1.weight.shape[0]
    n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
    header = struct.pack(
        "iiiiiiiiii",
        p.dim,
        hidden_dim,
        p.n_layers,
        p.n_heads,
        n_kv_heads,
        p.vocab_size,
        p.max_seq_len,
        p.head_dim,
        int(shared_classifier),
        group_size
    )
    out_file.write(header)

    pad = 256 - out_file.tell()  # pad rest with zeros; tell returns current pos
    assert pad >= 0
    out_file.write(b"\0" * pad)
    # now that the header is done, let's write out the model

    # first let's write out all the params that we are keeping in fp32: the norms
    for layer in model.layers:  # attention norms
        serialize_fp32(out_file, layer.attention_norm.weight)
    for layer in model.layers:  # MLP norms
        serialize_fp32(out_file, layer.ffn_norm.weight)
    serialize_fp32(out_file, model.norm.weight)  # final pre-classifier norm

    # write out the QK-RMSNorm weights (Qwen3)
    for layer in model.layers:
        serialize_fp32(out_file, layer.attention.lq.weight)
    for layer in model.layers:
        serialize_fp32(out_file, layer.attention.lk.weight)

    # now let's write out all the params that we are quantizing to Q8_0
    # note we skip classifier weights, which are shared with the embedding
    ew = []
    for i, w in enumerate(weights):
        # quantize this weight
        q, s, err = quantize_q80(w, group_size)
        # save the int8 weights to file
        serialize_int8(out_file, q)  # save the tensor in int8
        serialize_fp32(out_file, s)  # save scale factors
        # logging
        ew.append((err, w.shape))
        print(
            f"{i+1}/{len(weights)} quantized {tuple(w.shape)} to Q8_0 with max error {err:.8f}"
        )

    # print the highest error across all weights, should be very small, e.g. O(~0.001)
    ew.sort(reverse=True)
    print(f"max quantization group error across all weights: {ew[0][0]:.8f}")

    # write to binary file
    out_file.close()
    print(f"Written model checkpoint to {filepath}")

## Tokenizer functions

def bytes_to_unicode():
    """Reference GPT-2 byte→Unicode map."""
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))

def internal_to_bytes(U2B, token_str: str) -> bytes:
    return b''.join(
        bytes([U2B[ch]]) if ch in U2B else ch.encode('utf-8')
        for ch in token_str
    )

def build_tokenizer(model, file):
    # Build the reverse table once
    B2U = bytes_to_unicode()
    U2B = {u: b for b, u in B2U.items()}

    # Load tokenizer (adjust as needed)
    tokenizer = model.tokenizer

    # Get ID → token mapping
    vocab = tokenizer.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}
    all_tokens = [id_to_token[i] for i in sorted(id_to_token)]

    tokenizer_data = json.loads(tokenizer.backend_tokenizer.to_str())

    # Extract vocab and merge rules
    vocab = tokenizer_data["model"]["vocab"]
    merges = tokenizer_data["model"]["merges"]

    # Build merge rank table
    merge_rank = {''.join(tuple(merge if isinstance(merge, list) else merge.split())): i for i, merge in enumerate(merges)}

    # Create pseudo-score dictionary
    # Tokens from initial vocab get score 0 (unmerged tokens)
    # Merged tokens get scores based on merge rank
    pseudo_scores = {}
    for token_id, token in enumerate(all_tokens):
        # If this token was the result of a merge, it will appear in merge_rank
        rank = merge_rank.get(token)

        if rank is not None:
            score = -math.log(rank + 1)
        else:
            score = -1e6  # Initial vocab tokens
        pseudo_scores[token] = score

    max_token_length = max(len(t) for t in all_tokens)

    # Write to binary
    with open(file + ".tokenizer", "wb") as out_f:
        # Header: max_token_length, bos_token_id, eos_token_id
        out_f.write(struct.pack("<I", max_token_length))
        out_f.write(struct.pack("<I", model.bos_token_id))
        out_f.write(struct.pack("<I", model.eos_token_id))

        for id, token in enumerate(all_tokens):
            token_bytes = internal_to_bytes(U2B, token)
            out_f.write(struct.pack("f", pseudo_scores[token])) # merge score
            out_f.write(struct.pack("<I", len(token_bytes))) # 4 bytes: token length
            out_f.write(token_bytes)                         # UTF-8 bytes

    print(f"Written tokenizer model to {file}.tokenizer")

# -----------------------------------------------------------------------------
# Load / import functions


def load_hf_model(model_path):

    try:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("Error: transformers package is required to load huggingface models")
        print("Please run `pip install transformers` to install it")
        return None

    # load HF model
    hf_model = AutoModelForCausalLM.from_pretrained(model_path)
    hf_dict = hf_model.state_dict()

    # convert config to ModelArgs
    config = ModelArgs()

    config.dim = hf_model.config.hidden_size
    config.n_layers = hf_model.config.num_hidden_layers
    config.n_heads = hf_model.config.num_attention_heads
    config.n_kv_heads = hf_model.config.num_key_value_heads
    config.vocab_size = hf_model.config.vocab_size
    config.hidden_dim = hf_model.config.intermediate_size
    config.norm_eps = hf_model.config.rms_norm_eps
    config.max_seq_len = hf_model.config.max_position_embeddings
    config.head_dim = hf_model.config.head_dim if hasattr(hf_model.config, "head_dim") else config.dim // config.n_heads

    print(config)

    # create a new Transformer object and set weights
    model = Transformer(config)

    model.tok_embeddings.weight = nn.Parameter(hf_dict["model.embed_tokens.weight"])
    model.norm.weight = nn.Parameter(hf_dict["model.norm.weight"])

    model.tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.bos_token_id = hf_model.config.bos_token_id if hasattr(hf_model.config, "bos_token_id") else 0
    model.eos_token_id = hf_model.config.eos_token_id if hasattr(hf_model.config, "eos_token_id") else 0

    for layer in model.layers:
        i = layer.layer_id
        layer.attention_norm.weight = nn.Parameter(
            hf_dict[f"model.layers.{i}.input_layernorm.weight"]
        )
        layer.attention.wq.weight = nn.Parameter(
            hf_dict[f"model.layers.{i}.self_attn.q_proj.weight"]
        )
        layer.attention.wk.weight = nn.Parameter(
            hf_dict[f'model.layers.{i}.self_attn.k_proj.weight']
        )
        layer.attention.wv.weight = nn.Parameter(
            hf_dict[f"model.layers.{i}.self_attn.v_proj.weight"]
        )
        layer.attention.wo.weight = nn.Parameter(
            hf_dict[f"model.layers.{i}.self_attn.o_proj.weight"]
        )
        layer.attention.lq.weight = nn.Parameter(
            hf_dict[f"model.layers.{i}.self_attn.q_norm.weight"]
        )
        layer.attention.lk.weight = nn.Parameter(
            hf_dict[f"model.layers.{i}.self_attn.k_norm.weight"]
        )
        layer.ffn_norm.weight = nn.Parameter(
            hf_dict[f"model.layers.{i}.post_attention_layernorm.weight"]
        )
        layer.feed_forward.w1.weight = nn.Parameter(
            hf_dict[f"model.layers.{i}.mlp.gate_proj.weight"]
        )
        layer.feed_forward.w2.weight = nn.Parameter(
            hf_dict[f"model.layers.{i}.mlp.down_proj.weight"]
        )
        layer.feed_forward.w3.weight = nn.Parameter(
            hf_dict[f"model.layers.{i}.mlp.up_proj.weight"]
        )

    # final classifier
    model.output.weight = nn.Parameter(hf_dict["lm_head.weight"])
    model.eval()
    return model


# -----------------------------------------------------------------------------
# CLI entrypoint

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type=str, help="the output filepath")
    parser.add_argument("hfpath", type=str, help="huggingface model path")
    args = parser.parse_args()

    model = load_hf_model(args.hfpath)

    # export
    build_tokenizer(model, args.filepath)
    model_export(model, args.filepath)
