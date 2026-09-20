# KineJing CausalWM derivative

KineWorld maintains this explicit fork of AetherLabsAI/CausalWM at
`09caadc9dbd84be3c64586ef380c51180628db99`. Original authorship, LICENSE and
NOTICE remain intact. Changes remain under the LTX-2 Community License.

## 致谢 / Acknowledgements

勘境感谢 Aether AI、Lightricks 以及相关开源项目的作者和贡献者。感谢你们愿意公开研究成果、代码与复现方法，让小团队也能学习、验证并继续探索。我们保留原始作者、提交历史、许可证和引用，以具体改动和测试说明勘境新增的工作。

We thank the original authors for their open-source contributions. Original model
results remain attributed to their authors; KineWorld's changes and validation limits
are documented below.

## Implemented change

Add `--vae-tiling` to the upstream inference command to use its existing overlapping
spatial/temporal VAE tiler. Decoded chunks move to CPU before concatenation. Default
inference remains unchanged. The selected tiles are 256 pixels / 64 overlap and
32 frames / 8 overlap. Frame count, finite values and chunk shape are checked.
The selected tiling settings are recorded in provenance.

This targets **decoder activation memory**, not transformer weight memory. It has
passed CPU plumbing tests with synthetic decoder outputs. Full CausalWM weights,
quality parity and peak-memory reductions have NOT been measured. Tiling may change
decoded pixels. There is no claim that the full pipeline fits a 12 GB GPU.

The upstream released interface uses a single image plus text, not the unpublished
action-conditioned multi-view leaderboard configuration. Its reported score is not
a KineJing result. No original model weights are redistributed by this fork.

KineWorld's separately trained small tri-view latent predictor and its measurements
live in [KineJing](https://github.com/kineworld/KineJing). It is not injected into this
generator, and the two are not claimed to be jointly trained.

## Check the modification

```bash
python -m unittest discover -s tests -v
python inference.py --help
```

Use all original inference arguments and add `--vae-tiling` when experimenting on
hardware that can already load the complete model. This is an engineering preview,
not a new model-performance release.
