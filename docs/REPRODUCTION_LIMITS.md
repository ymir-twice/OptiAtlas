# Reproduction and source provenance

The final published system is historical experiment 53: the continuous text
reward RL95 policy and the Agent-pair equal-weight group-normalized OPSD
generator (`9ecae447…`). Its original single-run Precision is 80.54%.
The later strict matched Dense/Agent-caption experiments are ablations.

`provenance/source_manifest.json` records the selected source path, original
SHA256, release SHA256, and whether portability edits were applied. Generator
training/reward code is selected from experiment 53's frozen source snapshot.
The independent strict-caption recipes retain their own actual configurations;
using this common code snapshot for a fresh run has not been established as a
bitwise reconstruction of those later runs.

The full historical RL95 application source was not recovered. In the existing
evidence audit, current reward.py, services.py and agent_loop.py differed from
the original run's recorded hashes. Current protocol.py matched. This release
uses the available application implementation and the actual run's sanitized
configuration. It does not claim all application code is byte-identical to the
historical training execution. VERL and DiffusionOPSD pins are recorded; the
selected upstream code and notices accompany the release.

Portable entry points replace paths, API routing and deployment lookup. The
generator export loader verifies the selected adapter SHA and preserves FP32
LoRA values and alpha/rank semantics; it reuses the original native sampling
method. Training checkpoints, image data, complete trajectories, score dumps,
and credentials are not part of this source tree.

Planning and static/import checks do not constitute a successful GPU run.
No fresh full SFT, GRPO, generator training, A100 validation, API scoring or
end-to-end benchmark reproduction has been performed for this staged release.
Recorded reference environments target CUDA 12.8 and H100/Blackwell deployments.
Rebuilding packages/native extensions on another device must be validated there.

The pipeline uses external vision-language judging during GRPO and scoring.
Supply an explicit OpenAI-compatible API base URL and environment variable
token. Changing the judge/provider may change results. An API failure is not a
semantic Cannot-determine label. Evaluation repeats measure the complete
generation-and-judging pipeline, not independent retraining.

The inference policy is an SFT-merged Qwen base plus the RL95 adapter. Applying
RL95 directly to raw Qwen, or applying SFT twice, yields a different model.
The generator is an independently trained adapter on the original Qwen-Image
base. RL95 itself was trained using a different frozen dense/raw generator.

The release policy adapter exporter was checked on CPU against all 608 FP32 tensors of the existing RL95 export: every tensor and dtype matched exactly. It reconstructs adapter-only FSDP shards and avoids the vendored generic merger's BF16 cast. This verifies the export boundary, not a fresh training run or the unavailable historical exporter implementation. The result is recorded in `provenance/cpu_export_validation.json`.
