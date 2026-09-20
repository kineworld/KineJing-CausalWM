import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'packages/ltx-core/src'))
from causalwm.kinejing_decode import decode_tiled_to_cpu


class Decoder:
    def __init__(self, chunks): self.chunks = chunks
    def tiled_decode(self, latent, config, generator):
        assert config.temporal_config.tile_overlap_in_frames == 8
        yield from self.chunks


class DecodeTests(unittest.TestCase):
    def test_frame_order_range_and_cpu_output(self):
        chunks = [torch.full((1, 3, 2, 4, 4), -.5), torch.full((1, 3, 3, 4, 4), 2.)]
        y = decode_tiled_to_cpu(Decoder(chunks), None, generator=None, expected_frames=5)
        self.assertEqual(y.shape, (5, 3, 4, 4))
        self.assertEqual(y.device.type, 'cpu')
        self.assertTrue(torch.equal(y[:2], torch.full_like(y[:2], -.5)))
        self.assertTrue(torch.equal(y[2:], torch.ones_like(y[2:])))

    def test_bad_outputs_fail(self):
        for chunks in ([], [torch.zeros(1,3,6,4,4)], [torch.zeros(2,3,5,4,4)],
                       [torch.full((1,3,5,4,4), float('nan'))],
                       [torch.zeros(1,3,2,4,4), torch.zeros(1,3,3,5,4)]):
            with self.subTest(chunks=len(chunks)), self.assertRaises(ValueError):
                decode_tiled_to_cpu(Decoder(chunks), None, generator=None, expected_frames=5)

    def test_cli_default_keeps_upstream_decode(self):
        import inference
        parser = inference.build_parser()
        args = ['--image','i','--prompt','p','--checkpoint','c','--base-ckpt','b',
                '--text-encoder-dir','t','--out-dir','o']
        self.assertFalse(parser.parse_args(args).vae_tiling)
        self.assertTrue(parser.parse_args(args + ['--vae-tiling']).vae_tiling)


if __name__ == '__main__': unittest.main()
