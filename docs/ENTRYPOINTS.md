# Selected entry points

Use an editable source checkout. Copy `configs/paths.example.json` to `configs/paths.local.json` and set local model, data, output, and Python paths. Paths are resolved relative to that JSON file. `runtime.python` selects the policy environment; `runtime.generator_python` selects the generator environment.

Every command prints its resolved plan by default. Add `--execute` to run. Plans do not load models or call APIs.

| Task | Command | Preserved implementation |
|---|---|---|
| Environment | `python environments/setup.py policy` or `generator` | Pinned primary dependencies, vendored VERL, CUDA 12.8 torch |
| Policy SFT | `python -m optiatlas train-sft --paths configs/paths.local.json --gpus 8` | `optics_agent_sft.training.runner.run_training` |
| Merge SFT | `python -m optiatlas merge-sft --paths configs/paths.local.json` | `optics_agent_rl.merge_sft` |
| Frozen RL draw service | `python -m optiatlas serve-draw --paths configs/paths.local.json` | Dense/raw v5 generator, original draw HTTP protocol |
| Export trained policy | `python -m optiatlas export-policy --paths configs/paths.local.json --checkpoint /path/to/global_step_95 --output /path/to/new_adapter` | Adapter-only FSDP reconstruction preserving original dtype and exact tensor round trip |
| Policy GRPO | `python -m optiatlas train-rl --paths configs/paths.local.json --gpus 8` | Official VERL PPO with optical AgentLoop |
| Generator training | `python -m optiatlas train-generator --paths configs/paths.local.json --gpus 8` | Frozen exp53 native FM/OPSD trainer |
| Policy service | `python -m optiatlas serve-policy --paths configs/paths.local.json --gpus 1` | vLLM, SFT-merged base + RL95 adapter + recorded processor |
| Generate benchmark | `python -m optiatlas generate --paths configs/paths.local.json --gpus 1` | Original Agent runner, sampler, sharding and failure semantics |
| Score images | `python -m optiatlas score --help` | Included V6.1 scoring interface and sibling evaluator |

`train-generator --recipe` accepts `generator_final`, `generator_pair_mse`, `generator_pair_raw`, `generator_pair_fm`, `generator_pair_fm_lr1e5`, `generator_pair_gn_dino`, `generator_dense_raw`, `generator_dense_fm`, `generator_strict_dense`, and `generator_strict_agent`. Each keeps its recorded eight-worker/global-batch recipe.

`train-rl --recipe` accepts `policy_rl95`, `policy_rl_binary`, and `policy_rl_image`. The frozen drawing service uses the old dense/raw v5 generator for all three. Configure an explicit judge via `OPTIATLAS_API_BASE_URL`, `OPTIATLAS_JUDGE_MODEL`, and `OPTIATLAS_API_TOKEN`; use `OPTIATLAS_DRAW_TOKEN` for the drawing service. The original judge identity is documented in the paper evidence; selecting another provider/model is a changed experimental condition.

`serve-policy --recipe` accepts `rl95` (default), `raw`, `sft`, `binary`, and `image`. It checks the published model manifest for the SFT merge identity and verifies selected step95 adapter bytes before launch. The raw and SFT-only stage ablations omit RL. `OPTIATLAS_POLICY_TOKEN` configures the client; configure vLLM authentication through its supported environment if using an authenticated endpoint.

`generate --recipe` accepts `evaluation_final` (default), `evaluation_direct_base`, `evaluation_raw_policy_base`, `evaluation_sft_base`, `evaluation_sft_dense_raw`, `evaluation_sft_dense_fm`, `evaluation_dense_raw`, `evaluation_dense_fm`, `evaluation_pair_mse`, `evaluation_pair_raw`, `evaluation_pair_fm`, `evaluation_pair_fm_lr1e5`, `evaluation_pair_gn_dino`, `evaluation_policy_binary`, `evaluation_policy_image`, `evaluation_strict_dense`, and `evaluation_strict_agent`. Start the corresponding policy service before generation. Final and generator/strict ablations use RL95. Direct-base generation needs no policy service.

The historical final generator has adapter SHA `9ecae447a4c580742d1f33465ec6596a6a5d16eacaca77936905043e1580796f`; strict matched-caption adapters are separate ablations. FM export identities use the export-file SHA, supported by the preserved tensor-equality audit, not the different native `state.pt` SHA. `optiatlas.inference.install_release_provider()` supplies the portable loader and reuses `LocalFMCheckpointProvider.generate`; `load_qwen_lora_exact` preserves FP32 LoRA tensors.

Existing GPU and API services must be started in a suitable allocation. Single-node launcher resource counts are configurable; this release has no newly validated multi-node deployment recipe. A fresh environment build, GPU training/inference, and A100 compatibility remain unvalidated. See `REPRODUCTION_LIMITS.md`.

The exported HF dataset is loaded through an explicit published-format boundary. SFT validates the sanitized Gold dataset manifest while retaining source lineage. Generator preparation resolves shared image paths and selects the recorded caption view; strict Dense/Agent views share the exact same 13,866 ordered IDs. Original training tensorization, losses, rewards, and sampler equations are retained.
