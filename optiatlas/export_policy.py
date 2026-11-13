"""Export an adapter-only VERL FSDP checkpoint without changing tensor dtype.

This is a release adapter for the vendored VERL shard format. It is not a
reconstruction of the unavailable historical RL95 export script.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .paths import configure_imports
from .inference import sha256, validate_sft_base, SFT_MERGE_SHA


def export_adapter(actor, output, base):
    configure_imports()
    import torch
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    from torch.distributed.tensor import DTensor
    from safetensors.torch import load_file, save_file
    from peft import LoraConfig
    actor,output,base=Path(actor).resolve(),Path(output).resolve(),Path(base).resolve()
    validate_sft_base(base)
    if output.exists():raise FileExistsError("Select a fresh export output directory")
    fsdp=json.loads((actor/"fsdp_config.json").read_text())
    meta=json.loads((actor/"lora_train_meta.json").read_text())
    world=int(fsdp["world_size"])
    if meta.get("r")!=16 or meta.get("lora_alpha")!=32:
        raise ValueError("Expected the selected policy LoRA rank16/alpha32 recipe")
    shards=[actor/f"model_world_size_{world}_rank_{rank}.pt" for rank in range(world)]
    if not all(path.is_file() for path in shards):raise ValueError("Missing native FSDP shard")
    before={path.name:sha256(path) for path in shards}
    states=[torch.load(path,map_location="cpu",weights_only=False) for path in shards]
    names=set(states[0])
    if not names or any(set(state)!=names for state in states):raise ValueError("FSDP tensor key sets differ")
    tensors={}
    for name in sorted(names):
        if ".lora_A." not in name and ".lora_B." not in name:
            raise ValueError("Only adapter-only checkpoints are supported")
        values=[state.pop(name) for state in states]
        if isinstance(values[0],DTensor):
            placement=tuple(values[0].placements)
            if len(placement)!=1 or any(not isinstance(v,DTensor) or tuple(v.placements)!=placement for v in values):
                raise ValueError("Only matching one-dimensional FSDP placements are supported")
            parts=[v.to_local() for v in values]
            mode=placement[0]
            if mode.is_shard():value=torch.cat(parts,dim=mode.dim).contiguous()
            elif mode.is_replicate():
                if any(not torch.equal(parts[0],v) for v in parts[1:]):raise ValueError("Replicated FSDP tensors differ")
                value=parts[0].contiguous()
            else:raise ValueError("Partial FSDP placements are unsupported")
        else:
            if any(isinstance(v,DTensor) or not torch.equal(values[0],v) for v in values[1:]):
                raise ValueError("Replicated tensors differ")
            value=values[0].contiguous()
        if not value.is_floating_point() or not torch.isfinite(value).all():raise ValueError("Invalid policy adapter tensor")
        key=name.replace(".default.weight",".weight")
        if key in tensors:raise ValueError("Duplicate export tensor key")
        tensors[key]=value
    del states
    if any(sha256(path)!=before[path.name] for path in shards):raise ValueError("Checkpoint changed during export")
    modules=sorted({key.split(".")[-3] for key in tensors})
    config=LoraConfig(r=16,lora_alpha=32,target_modules=modules,task_type="CAUSAL_LM",inference_mode=True).to_dict()
    for key in ("task_type","peft_type"):
        config[key]=getattr(config[key],"value",config[key])
    config["target_modules"]=modules
    config["base_model_name_or_path"]=str(base)
    output.mkdir(parents=True)
    weights=output/"adapter_model.safetensors"
    save_file(tensors,str(weights))
    restored=load_file(str(weights),device="cpu")
    if set(restored)!=set(tensors) or any(restored[k].dtype!=v.dtype or not torch.equal(restored[k],v) for k,v in tensors.items()):
        raise ValueError("Export tensor round trip failed")
    (output/"adapter_config.json").write_text(json.dumps(config,indent=2)+"\n")
    manifest={"schema_version":1,"format":"optiatlas-fsdp-adapter-export-v1","tensor_round_trip_exact":True,
        "required_base":"SFT309 three-epoch merged policy base","source_merge_manifest_sha256":SFT_MERGE_SHA,
        "source_shards_sha256":before,"weights_sha256":sha256(weights),"tensor_count":len(tensors),
        "tensor_dtypes":sorted({str(v.dtype) for v in tensors.values()}),
        "processor":"Use the source actor/huggingface processor directory.",
        "export_implementation_sha256":sha256(Path(__file__))}
    (output/"release_export_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--base",type=Path,required=True)
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args()
    if args.execute:print(json.dumps(export_adapter(args.actor,args.output,args.base),indent=2))
    else:print(json.dumps({"actor":str(args.actor),"output":str(args.output),"base":str(args.base),"executing":False},indent=2))


if __name__=="__main__":main()
