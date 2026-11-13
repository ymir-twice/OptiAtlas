#!/usr/bin/env python3
"""Print or run an isolated environment build; no platform-specific mirror."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("profile",choices=["policy","generator"])
parser.add_argument("--execute",action="store_true")
parser.add_argument("--venv",type=Path)
args=parser.parse_args()
venv=(args.venv or ROOT/(".venv-"+args.profile)).resolve()
python=venv/"bin/python"
commands=[
    ["uv","venv","--python","3.12" if args.profile=="policy" else "3.11",str(venv)],
    ["uv","pip","install","--python",str(python),"--index-url","https://download.pytorch.org/whl/cu128",
     "torch==2.10.0","torchvision==0.25.0","torchaudio==2.10.0"],
    ["uv","pip","install","--python",str(python),"-r",str(ROOT/"environments"/(args.profile+".txt"))],
    ["uv","pip","install","--python",str(python),"--no-deps","-e",str(ROOT)],
]
if args.profile=="policy":
    commands += [["uv","pip","install","--python",str(python),"--no-deps","-e",str(ROOT/"vendor/verl")],
                 ["uv","pip","install","--python",str(python),"--no-build-isolation","flash-attn==2.8.3"]]
commands += [["uv","pip","check","--python",str(python)]]
print(json.dumps({"profile":args.profile,"commands":commands,
    "limits":"CUDA 12.8/H100 reference environment; no A100 or fresh-environment GPU validation claimed. Native extensions may require a compatible CUDA compiler."},indent=2))
if args.execute:
    for command in commands:
        subprocess.run(command,cwd=ROOT,check=True)
