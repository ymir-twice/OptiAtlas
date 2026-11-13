"""Plan or execute one explicit training/evaluation stage."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from .paths import ROOT, configure_imports, load_settings, resolve

TASKS = {"train-sft":"policy_sft", "train-rl":"policy_rl95", "train-generator":"generator_final", "generate":"evaluation_final"}


def prepared_config(task, name, settings):
    path = ROOT / "configs" / (name + ".json")
    cfg = resolve(json.loads(path.read_text()), settings)
    if task == "train-generator":
        if cfg["model"]["name"] != "qwen_image21" or cfg["algorithm"]["name"] not in {"flow_matching", "diffusion_opsd"}:
            raise ValueError("Only the selected Qwen FM/OPSD recipes are included")
        if cfg["algorithm"]["name"] == "diffusion_opsd" and cfg.get("reward") and cfg["reward"]["kind"] not in {"negative_mse","dinov2_hpsv2_mse","dinov3_hpsv2_mse"}:
            raise ValueError("This release exposes only the generator rewards used in the selected ablation table")
    if task == "train-rl":
        if cfg["actor_rollout_ref"]["model"]["path"] == settings["paths"]["policy_raw_base"]:
            raise ValueError("RL must initialize from the already SFT-merged base")
        # Use the portable source file, not a training-machine sampler path.
        cfg["trainer"]["v1"]["sampler"]["custom_sampler"]["path"] = str(ROOT / "optics_agent_rl/judge_skip_sampler.py")
    return cfg


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0]=="score":
        configure_imports()
        from .scoring import main as score_main
        return score_main(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=[*TASKS, "merge-sft", "export-policy", "serve-policy", "serve-draw", "score"])
    parser.add_argument("--paths", type=Path, default=ROOT / "configs/paths.example.json")
    parser.add_argument("--recipe", help="Explicit config basename; no historical experiment menu")
    parser.add_argument("--checkpoint",type=Path,help="For export-policy: native global_step_N directory")
    parser.add_argument("--output",type=Path,help="For export-policy: fresh adapter export directory")
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--execute", action="store_true", help="Run; default prints a dependency-light plan only")
    args, extra = parser.parse_known_args(argv)
    configure_imports()
    settings = load_settings(args.paths)
    if extra:
        parser.error("Unexpected arguments: " + " ".join(extra))
    if args.gpus < 1:
        parser.error("--gpus must be positive")
    if args.task == "serve-draw":
        from .inference import install_release_provider
        cfg = prepared_config("generate","evaluation_final",settings)
        image = cfg["benchmark"]["image"]
        image.update(checkpoint_path=settings["paths"]["generator_rl_training"],state_sha256="1fe0c8df1e2c49cb6f9d4f9cc4fb31d4ac7264d563c74254fd6665c351a37340")
        server = {"host":"127.0.0.1","port":18320,"model_name":"Qwen-Image-2.1","image":image,
                  "api_key_env":"OPTIATLAS_DRAW_TOKEN","allow_no_key":False,"max_pending_requests":16,"read_timeout_seconds":60}
        print(json.dumps({"task":args.task,"server":server,"executing":args.execute},indent=2))
        if args.execute:
            from optics_agent_rl.draw_server import create_server
            provider = install_release_provider()(**{k:v for k,v in image.items() if k not in {"provider","release_export"}})
            with create_server(server,provider) as service:
                service.serve_forever()
        return 0
    if args.task == "export-policy":
        if not args.checkpoint or not args.output:parser.error("export-policy requires --checkpoint and --output")
        command=[settings["runtime"]["python"],"-m","optiatlas.export_policy","--actor",str(args.checkpoint.resolve()/"actor"),
            "--output",str(args.output.resolve()),"--base",settings["paths"]["policy_sft_base"],"--execute"]
        config=None
    elif args.task == "merge-sft":
        command = [settings["runtime"]["python"], "-m", "optics_agent_rl.merge_sft", "--base", settings["paths"]["policy_raw_base"],
            "--adapter", settings["paths"]["sft_adapter"], "--output", settings["paths"]["policy_sft_base"], "--verification", "full"]
        config = None
    elif args.task == "serve-policy":
        from .inference import POLICIES, sha256, validate_sft_base
        policy=args.recipe or "rl95"
        if policy not in {*POLICIES,"raw","sft"}:
            parser.error("serve-policy --recipe must be rl95, binary, image, raw, or sft")
        base = Path(settings["paths"]["policy_raw_base" if policy=="raw" else "policy_sft_base"])
        model_name="OptiAtlas-Raw" if policy=="raw" else "OptiAtlas-SFT-base"
        tokenizer=str(base)
        if policy in POLICIES:
            adapter_name,adapter_key,tokenizer_key,expected_sha=POLICIES[policy]
            adapter=Path(settings["paths"][adapter_key]);tokenizer=settings["paths"][tokenizer_key]
        command = [settings["runtime"]["python"], "-m", "vllm.entrypoints.openai.api_server", "--model",str(base),
            "--tokenizer",tokenizer, "--served-model-name",model_name, "--enable-auto-tool-choice",
            "--tool-call-parser","qwen3_xml", "--tensor-parallel-size",str(args.gpus), "--max-model-len","28672"]
        if policy in POLICIES:
            command += ["--enable-lora","--lora-modules",adapter_name+"="+str(adapter),"--max-lora-rank","16"]
        config = None
        if args.execute:
            if policy!="raw":validate_sft_base(base)
            if policy in POLICIES and sha256(adapter/"adapter_model.safetensors") != expected_sha:
                raise ValueError("Policy adapter is not the selected published step95 export")
    else:
        config = prepared_config(args.task, args.recipe or TASKS[args.task], settings)
        if args.task == "train-rl":
            config["trainer"]["n_gpus_per_node"]=args.gpus
        if args.task in {"train-sft","train-generator"}:
            expected=config.get("expected_world_size",config.get("distributed",{}).get("expected_world_size"))
            if expected is not None and args.gpus!=expected:
                raise ValueError(f"This recipe records {expected} workers. Use --gpus {expected}, or review the global batch changes in a separately named recipe.")
        if args.task == "generate":
            config["benchmark_launcher"]["devices"] = list(range(args.gpus))
        target = Path(settings["paths"]["output"]) / ".resolved" / (args.task+"-"+(args.recipe or TASKS[args.task])+".json")
        python = settings["runtime"]["generator_python"] if args.task in {"train-generator","generate"} else settings["runtime"]["python"]
        command = [python]
        if args.task in {"train-sft", "train-generator"}:
            command += ["-m","torch.distributed.run","--standalone","--nnodes=1",f"--nproc_per_node={args.gpus}"]
        command += ["-m","optiatlas.worker","--task",args.task,"--config",str(target)]
        if args.execute:
            if args.task=="train-generator":
                from .generator_data import prepare_generator_data
                config=prepare_generator_data(config,settings["paths"]["dataset_root"],target.parent)
            if args.task=="train-rl":
                from .inference import validate_sft_base
                validate_sft_base(settings["paths"]["policy_sft_base"])
                if not settings["api"]["judge_base_url"].startswith(("http://","https://")) or not settings["api"]["judge_model"]:
                    raise ValueError("Set OPTIATLAS_API_BASE_URL and OPTIATLAS_JUDGE_MODEL before RL training")
                if not os.environ.get("OPTIATLAS_API_TOKEN"):
                    raise ValueError("Set OPTIATLAS_API_TOKEN before RL training")
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text(json.dumps(config,indent=2)+"\n")
    print(json.dumps({"task":args.task,"command":command,"config":config,"executing":args.execute},indent=2))
    if args.execute:
        return subprocess.call(command,cwd=ROOT,env={**os.environ,"PYTHONPATH":os.pathsep.join([str(ROOT),str(ROOT/"vendor/verl"),os.environ.get("PYTHONPATH","")])})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
