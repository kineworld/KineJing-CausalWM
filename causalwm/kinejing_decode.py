"""KineWorld optional tiled decoding extension; LTX-2 Community License.

Uses the upstream VAE tiler. This does not reduce transformer weight residency
and does not establish full-model compatibility with a 12 GB GPU.
"""
from __future__ import annotations

import torch


def decode_tiled_to_cpu(decoder, latent, *, generator, expected_frames):
    from ltx_core.model.video_vae.tiling import SpatialTilingConfig, TemporalTilingConfig, TilingConfig

    config = TilingConfig(
        spatial_config=SpatialTilingConfig(tile_size_in_pixels=256, tile_overlap_in_pixels=64),
        temporal_config=TemporalTilingConfig(tile_size_in_frames=32, tile_overlap_in_frames=8),
    )
    chunks = []
    count = 0
    spatial_shape = None
    for chunk in decoder.tiled_decode(latent, config, generator=generator):
        if chunk.ndim != 5 or chunk.shape[0] != 1 or chunk.shape[1] != 3:
            raise ValueError('VAE chunk must have shape [1,3,T,H,W]')
        if chunk.shape[2] == 0:
            continue
        if spatial_shape is None:
            spatial_shape = chunk.shape[-2:]
        if chunk.shape[-2:] != spatial_shape or not bool(torch.isfinite(chunk).all()):
            raise ValueError('Invalid VAE chunk dimensions or nonfinite values')
        count += chunk.shape[2]
        if count > expected_frames:
            raise ValueError('Tiled VAE emitted too many frames')
        # Transfer each chunk before concatenation: no full RGB movie on GPU.
        chunks.append(chunk[0].detach().to(device='cpu', dtype=torch.float32).clamp(-1, 1).permute(1, 0, 2, 3))
    if count != expected_frames or not chunks:
        raise ValueError(f'Tiled VAE emitted {count} frames; expected {expected_frames}')
    return torch.cat(chunks, dim=0)
