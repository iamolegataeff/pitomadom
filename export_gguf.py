#!/usr/bin/env python3
"""
PITOMADOM GGUF Exporter — Convert trained weights to GGUF format.

GGUF v3 format for Go/C inference without Python dependencies.

Usage:
  python3 export_gguf.py --weights pitomadom_rtl_weights.npz --output pitomadom.gguf
  python3 export_gguf.py --checkpoint pitomadom_final.pt --output pitomadom.gguf
"""

import struct
import numpy as np
import argparse
import sys
from pathlib import Path

# GGUF constants
GGUF_MAGIC = b'GGUF'
GGUF_VERSION = 3

# GGUF value types
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_STRING = 8
GGUF_TYPE_UINT64 = 10

# GGUF tensor types
GGUF_TENSOR_F32 = 0
GGUF_TENSOR_F16 = 1

ALIGNMENT = 32


def write_string(f, s: str):
    """Write GGUF string: uint64 length + bytes."""
    encoded = s.encode('utf-8')
    f.write(struct.pack('<Q', len(encoded)))
    f.write(encoded)


def write_kv_string(f, key: str, value: str):
    """Write a string key-value pair."""
    write_string(f, key)
    f.write(struct.pack('<I', GGUF_TYPE_STRING))
    write_string(f, value)


def write_kv_uint32(f, key: str, value: int):
    """Write a uint32 key-value pair."""
    write_string(f, key)
    f.write(struct.pack('<I', GGUF_TYPE_UINT32))
    f.write(struct.pack('<I', value))


def write_kv_int32(f, key: str, value: int):
    """Write an int32 key-value pair."""
    write_string(f, key)
    f.write(struct.pack('<I', GGUF_TYPE_INT32))
    f.write(struct.pack('<i', value))


def write_kv_float32(f, key: str, value: float):
    """Write a float32 key-value pair."""
    write_string(f, key)
    f.write(struct.pack('<I', GGUF_TYPE_FLOAT32))
    f.write(struct.pack('<f', value))


def align_offset(offset: int, alignment: int = ALIGNMENT) -> int:
    """Round up to next alignment boundary."""
    return (offset + alignment - 1) // alignment * alignment


def export_gguf(weights_path: str, output_path: str, use_f16: bool = False):
    """Export weights to GGUF format."""

    # Load weights
    if weights_path.endswith('.pt'):
        import torch
        checkpoint = torch.load(weights_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        tensors = {}
        for name, param in state_dict.items():
            tensors[name] = param.numpy() if hasattr(param, 'numpy') else np.array(param)
    else:
        data = np.load(weights_path)
        tensors = {name: data[name] for name in data.files}

    # Detect architecture from weight shapes
    dim = None
    ff_dim = None
    num_heads = None
    num_layers = 0

    for name, arr in tensors.items():
        if name == 'input_proj.weight':
            dim = arr.shape[0]
        elif name == 'layers.0.ff.0.weight':
            ff_dim = arr.shape[0]
        elif name == 'layers.0.dissonance_bias.distance_scale':
            num_heads = arr.shape[0]
        if name.startswith('layers.'):
            layer_num = int(name.split('.')[1])
            num_layers = max(num_layers, layer_num + 1)

    if dim is None:
        print("ERROR: Could not detect model dimension from weights")
        sys.exit(1)

    total_params = sum(arr.size for arr in tensors.values())
    tensor_type = GGUF_TENSOR_F16 if use_f16 else GGUF_TENSOR_F32

    print(f"Exporting GGUF:")
    print(f"  Architecture: dim={dim}, ff={ff_dim}, heads={num_heads}, layers={num_layers}")
    print(f"  Parameters: {total_params:,}")
    print(f"  Tensors: {len(tensors)}")
    print(f"  Format: {'f16' if use_f16 else 'f32'}")

    # Prepare metadata
    metadata = [
        ('general.architecture', 'pitomadom_rtl', 'string'),
        ('general.name', 'PITOMADOM Hebrew Root Transformer', 'string'),
        ('general.author', 'Arianna Method', 'string'),
        ('pitomadom.embedding_length', dim, 'uint32'),
        ('pitomadom.feed_forward_length', ff_dim or dim * 4, 'uint32'),
        ('pitomadom.attention.head_count', num_heads or 8, 'uint32'),
        ('pitomadom.block_count', num_layers, 'uint32'),
        ('pitomadom.letter_vocab_size', 25, 'uint32'),  # 22 letters + PAD + MASK + UNK
        ('pitomadom.output_classes', 22, 'uint32'),  # 22 Hebrew letters
        ('pitomadom.context_length', 64, 'uint32'),
    ]

    n_kv = len(metadata)
    n_tensors = len(tensors)

    # Convert tensors to target type
    tensor_data = {}
    for name, arr in tensors.items():
        arr = arr.astype(np.float32)
        if use_f16:
            arr = arr.astype(np.float16)
        tensor_data[name] = arr

    # Calculate offsets
    # First, write header + metadata + tensor infos to get data start offset
    # Then write tensor data with alignment

    with open(output_path, 'wb') as f:
        # === HEADER ===
        f.write(GGUF_MAGIC)
        f.write(struct.pack('<I', GGUF_VERSION))
        f.write(struct.pack('<Q', n_tensors))
        f.write(struct.pack('<Q', n_kv))

        # === METADATA ===
        for key, value, vtype in metadata:
            if vtype == 'string':
                write_kv_string(f, key, value)
            elif vtype == 'uint32':
                write_kv_uint32(f, key, value)
            elif vtype == 'float32':
                write_kv_float32(f, key, value)

        # === TENSOR INFOS ===
        # Calculate where tensor data starts
        # We need to write all tensor infos first, then pad to alignment
        tensor_info_start = f.tell()

        # Pre-calculate tensor info sizes to find data start
        tensor_names = sorted(tensor_data.keys())
        tensor_infos = []
        for name in tensor_names:
            arr = tensor_data[name]
            tensor_infos.append({
                'name': name,
                'shape': arr.shape,
                'type': tensor_type,
                'data': arr,
            })

        # Write placeholder tensor infos to calculate total header size
        info_size_per_tensor = []
        for info in tensor_infos:
            pos_before = f.tell()
            name_bytes = info['name'].encode('utf-8')
            f.write(struct.pack('<Q', len(name_bytes)))
            f.write(name_bytes)
            n_dims = len(info['shape'])
            f.write(struct.pack('<I', n_dims))
            for d in info['shape']:
                f.write(struct.pack('<Q', d))
            f.write(struct.pack('<I', info['type']))
            f.write(struct.pack('<Q', 0))  # placeholder offset
            info_size_per_tensor.append(f.tell() - pos_before)

        # Align to start tensor data
        header_end = f.tell()
        data_start = align_offset(header_end)
        padding_needed = data_start - header_end
        f.write(b'\x00' * padding_needed)

        # Calculate actual tensor offsets (relative to data_start)
        offsets = []
        current_offset = 0
        for info in tensor_infos:
            offsets.append(current_offset)
            tensor_size = info['data'].nbytes
            current_offset = align_offset(current_offset + tensor_size)

        # Go back and rewrite tensor infos with correct offsets
        f.seek(tensor_info_start)
        for i, info in enumerate(tensor_infos):
            name_bytes = info['name'].encode('utf-8')
            f.write(struct.pack('<Q', len(name_bytes)))
            f.write(name_bytes)
            n_dims = len(info['shape'])
            f.write(struct.pack('<I', n_dims))
            for d in info['shape']:
                f.write(struct.pack('<Q', d))
            f.write(struct.pack('<I', info['type']))
            f.write(struct.pack('<Q', offsets[i]))

        # Seek to data start
        f.seek(data_start)

        # === TENSOR DATA ===
        for i, info in enumerate(tensor_infos):
            # Verify alignment
            current = f.tell() - data_start
            expected = offsets[i]
            if current < expected:
                f.write(b'\x00' * (expected - current))

            f.write(info['data'].tobytes())

        total_size = f.tell()

    print(f"  Output: {output_path}")
    print(f"  File size: {total_size / 1e6:.1f} MB")
    print(f"  Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export PITOMADOM weights to GGUF')
    parser.add_argument('--weights', type=str, help='Path to .npz weights file')
    parser.add_argument('--checkpoint', type=str, help='Path to .pt checkpoint')
    parser.add_argument('--output', type=str, default='pitomadom.gguf', help='Output GGUF path')
    parser.add_argument('--f16', action='store_true', help='Export as float16')

    args = parser.parse_args()

    if not args.weights and not args.checkpoint:
        print("ERROR: Provide --weights or --checkpoint")
        sys.exit(1)

    source = args.weights or args.checkpoint
    export_gguf(source, args.output, use_f16=args.f16)
