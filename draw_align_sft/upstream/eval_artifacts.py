"""Optional lossless previews of already-scored OPSD validation images.

Reward values are from the original floating images, before PNG quantization.
This writer never samples, modifies model weights, or consumes any RNG.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch


def _sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _write_sample(path, record, png):
    """Permit only a byte-identical repeated fixed evaluation after resume."""
    metadata = path.with_suffix('.json')
    if path.exists() or metadata.exists():
        if not path.is_file() or not metadata.is_file():
            raise FileExistsError(f'Incomplete previous evaluation image pair: {path}')
        if json.loads(metadata.read_text()) != record or _sha_bytes(path.read_bytes()) != record['png_sha256']:
            raise FileExistsError(f'Evaluation image output already differs: {path}')
        return
    with path.open('xb') as stream:
        stream.write(png)
    with metadata.open('x') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def persist_eval_batch(*, images, prompts, prompt_metadata, rewards, valid, local_indices,
                       dataset_size, save_dir, global_step, rank, world_size,
                       eval_seed, bank, resolution, num_steps, guidance_scale,
                       dist_module=None, group=None):
    """Save each valid distributed sample once and collectively propagate I/O errors.

    The caller's unshuffled DistributedSampler uses ``local_index * world + rank``.
    Invalid tail padding executes collectives but produces neither image nor score
    metadata. Each rank writes its own directory. No group-zero ownership assumed.
    """
    saved = []
    error = None
    try:
        if bank not in {'evaluation_ema', 'student'}:
            raise ValueError('Unsupported evaluation parameter bank')
        count = len(prompts)
        if (images.ndim != 4 or images.shape[0] != count or images.shape[1] != 3
                or len(prompt_metadata) != count or len(valid) != count or len(local_indices) != count):
            raise ValueError('Image, prompt, metadata and sampler batch sizes disagree')
        if not (type(world_size) is int and world_size > 0 and type(rank) is int and 0 <= rank < world_size):
            raise ValueError('Invalid evaluation rank/world size')
        if not (type(dataset_size) is int and dataset_size > 0 and type(global_step) is int and global_step >= 0):
            raise ValueError('Invalid evaluation dataset size or step')
        if any(len(values) != count for values in rewards.values()):
            raise ValueError('Image and reward batch sizes disagree')
        folder = Path(save_dir) / 'eval_samples_all' / f'step_{global_step}' / f'rank_{rank}'
        for i in range(count):
            local_index = int(local_indices[i])
            dataset_index = local_index * world_size + rank
            if local_index < 0 or bool(valid[i]) != (dataset_index < dataset_size):
                raise ValueError('Sampler padding mask disagrees with the dataset index')
            if not bool(valid[i]):
                continue
            image = images[i].detach().to(device='cpu', dtype=torch.float32).contiguous()
            if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                raise ValueError('Preview image must be finite and in [0,1]')
            scores = {key: float(values[i]) for key, values in rewards.items()}
            if any(not math.isfinite(value) for value in scores.values()):
                raise ValueError('Nonfinite evaluation reward')
            if not isinstance(prompts[i], str) or not isinstance(prompt_metadata[i], dict):
                raise ValueError('Complete prompt and metadata mapping required')
            # Exact existing float image is scored; only the preview is quantized.
            pixels = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            buffer = io.BytesIO()
            Image.fromarray(pixels).save(buffer, format='PNG')
            png = buffer.getvalue()
            filename = f'sample_{dataset_index:05d}.png'
            record = {
                'format': 1, 'image': filename, 'dataset_index': dataset_index,
                'local_index': local_index, 'rank': rank, 'world_size': world_size,
                'dataset_size': dataset_size, 'is_padding': False,
                'prompt': prompts[i], 'prompt_metadata': prompt_metadata[i],
                'rewards': scores, 'reward_input': 'original_unquantized_float_image',
                'eval_seed': eval_seed,
                'eval_rank_seed': (eval_seed + rank) % (2**32) if eval_seed is not None else None,
                'eval_step': global_step, 'weights': bank,
                'resolution': resolution, 'num_steps': num_steps,
                'guidance_scale': guidance_scale, 'solver': 'flow', 'deterministic': True,
                'float_image_sha256': _sha_bytes(image.numpy().tobytes()),
                'float_image_dtype': 'float32', 'float_image_shape': list(image.shape),
                'png_sha256': _sha_bytes(png),
                'png_quantization': 'floor(float32_image * 255), RGB uint8; diagnostic preview',
            }
            # Validate JSON-serializability before creating either output file.
            json.dumps(record, ensure_ascii=False, allow_nan=False)
            folder.mkdir(parents=True, exist_ok=True)
            _write_sample(folder / filename, record, png)
            saved.append(record)
    except Exception as exc:
        error = f'rank {rank}: {type(exc).__name__}: {exc}'
    if world_size > 1:
        if dist_module is None:
            raise RuntimeError('Distributed preview saving requires the caller process group')
        errors = [None] * world_size
        dist_module.all_gather_object(errors, error, group=group)
    else:
        errors = [error]
    failures = [value for value in errors if value is not None]
    if failures:
        raise RuntimeError('Saving all evaluation images failed: ' + '; '.join(failures))
    return saved
