"""Resolve published image references before the unchanged generator trainer."""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
from uuid import uuid4
from .dataset_loader import load_generator, local, _digest


def prepare_generator_data(config, dataset_root, destination):
    cfg=deepcopy(config)
    selection=cfg.pop("release_dataset")
    variant,caption_field=selection["variant"],selection["caption_field"]
    if variant not in {"dense","agent"} or caption_field not in {"caption","dense_caption","agentv4_rl_prompt"}:
        raise ValueError("Unknown released generator training view")
    root=Path(dataset_root).resolve()
    relative=f"generator/{variant}/train.jsonl"
    published=json.loads((root/"release_manifest.json").read_text())
    source=local(root,relative)
    source_sha=_digest(source)
    if source_sha!=published["files"][relative]["sha256"]:
        raise ValueError("Published generator metadata hash changed")
    directory=Path(destination);directory.mkdir(parents=True,exist_ok=True)
    target=directory/f"generator_{variant}_{caption_field}_{source_sha[:16]}.jsonl"
    temporary=target.with_suffix("."+uuid4().hex+".tmp")
    count=0;seen=set()
    try:
        with temporary.open("w") as stream:
            for row in load_generator(root,variant):
                if not isinstance(row.get(caption_field),str) or not row[caption_field].strip():
                    raise ValueError("Missing caption for selected generator view")
                if row["id"] in seen:raise ValueError("Duplicate generator image ID")
                seen.add(row["id"])
                row["caption"]=row[caption_field]
                stream.write(json.dumps(row,ensure_ascii=False)+"\n");count+=1
        if count != (13867 if variant=="dense" else 13866):
            raise ValueError("Published generator view count differs from its training protocol")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    cfg["data"]["path"]=str(target.resolve())
    identity={"source_metadata_sha256":source_sha,"prepared_metadata_sha256":_digest(target),
        "variant":variant,"caption_field":caption_field,"records":count,
        "image_hashes":"The original generator preflight hashes referenced image bytes before training."}
    target.with_suffix(".identity.json").write_text(json.dumps(identity,indent=2)+"\n")
    return cfg
