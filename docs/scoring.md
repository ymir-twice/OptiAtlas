# Score existing generated images

The scoring entry point only judges existing images. It does not generate images,
train a model, or choose a provider. The default is an offline plan and does not
read the key environment variable or create a score directory.

```bash
python -m optiatlas score \
  --benchmark /path/to/OptiAtlas-Dataset/benchmark/v6_1/light_omni_v6_1.json \
  --generation-dir /path/to/existing_generation_output \
  --base-url https://api.example.com/v1 \
  --model YOUR_EXACT_JUDGE_MODEL \
  --key-env OPTIATLAS_JUDGE_TOKEN \
  --output /path/to/new_score_directory
```

After reviewing the plan, provide your own key through the named environment
variable and add `--execute` to run. No key files are loaded. Do not put keys in
URLs, CLI arguments, configuration files, or version control. The base URL and
model are always explicit. There is no provider fallback. A response naming a
different model is a technical failure unless that exact alias was explicitly
listed with `--allow-served-model`.

The generation directory must contain `manifest.json` with unique `records`.
Each requested record must preserve `sample_id`, the exact `generation_prompt`,
relative `image_path`, and `image_sha256`. Images are hashed again before judging.
Paths escaping the supplied directory are rejected. Alternatively, `--images DIR`
accepts `sample_id.png`, `.jpg`, `.jpeg`, or `.webp` and records that generation
provenance is unverified. Multiple image candidates for an ID are rejected.
`--limit N` explicitly creates a partial scoring scope; it is not a full benchmark.

V6.1 uses its already-frozen, per-question option permutations, version 6.1 and
seed 20261026. This CLI does not reshuffle them. The API receives only the current
English question/check, displayed options/rubrics, judge hint, generation request,
and actual generated image. Ideal answers, private permutation maps, provenance,
and metric conventions are not sent. The bundled label-neutral system text and
suffix are hash-checked against the historical V6.1 protocol. The recorded
historical judge was `google/gemini-3.6-flash`; using a different model or service
does not reproduce that judge automatically.

The response must be one JSON object with exactly `question_id`, `choice`, and
`evidence`. The original parser rejects duplicate keys, mismatched IDs, extra
keys, invalid labels, refusal/incomplete responses, and evidence outside 1–3
nonempty statements of at most 300 characters. It accepts a single JSON code
fence as the original parser did. Each displayed answer is mapped back locally
to canonical A=True, B=False, C=Cannot determine before computing counts.

Defaults match the original semantic request: one image and one question per
call, temperature 0, 4096 max output tokens, 120-second timeout, up to 4 total
attempts, 5-second initial retry delay and factor-2 backoff. HTTP 408/425/429/5xx
and parse errors may retry; authentication/client errors do not. This portable
CLI uses a single worker and optional `--requests-per-minute` pacing. It does not
recreate the old multi-process admission service. Explicit response schemas are
not sent because the original request used the system instructions and parser.

`answers.jsonl` stores only structured per-question decisions, image/question/
judge fingerprints, status and concise evidence. `summary.json` reports question-
micro P=A/(A+B), J=(A+B)/(A+B+C), and O=A/(A+B+C), plus category counts. Undefined
denominators are null. Technical/API/parse/corrupt/missing-image failures remain
unresolved and outside all denominators; they are never silently converted to C.
This entry point does not perform the separate historical post-hoc policy that
imputed C for proven terminal missing-generation artifacts. Existing unsubmitted
images are still judged and their generation status remains visible.

The output includes no saved HTTP request bodies, base64 images, or raw provider
response text. Existing score directories are never overwritten. Exit 0 means a
complete score run (or successful offline plan); exit 2 means technical failures
remain. Review scope and `complete` before reporting a benchmark number. A fresh
run can vary even at temperature 0; this CLI does not promise identical outputs.

Offline validation:

```bash
python -m unittest discover -s tests -p test_scoring.py -v
```

The tests use synthetic images and fake HTTP sessions. They never contact APIs.
