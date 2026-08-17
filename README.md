# Permit Scope Classifier

Classifies City of Toronto building permits by **unit form** and **construction scope**
using a local [Ollama](https://ollama.com) model. CSV in, CSV out — no cloud API, no
database, no API key.

Standalone successor to the `classify/` step in
[`to-multiplex-map`](../to-multiplex-map), which called the Claude API and wrote a dbt
seed. This version reads a raw permit export directly and runs entirely offline.

## What it produces

Two **independent** judgments per permit. They are deliberately not collapsed into one
enum: that would conflate a structural fact (is the extra unit a basement/secondary
suite tucked into a house) with a scope-of-work fact (was the building itself newly
built or altered). A basement suite can be part of a brand-new house just as easily as
a retrofit into an old one, so "mentions a basement" must not imply "nothing new was
built."

| Field | Values |
|---|---|
| `unit_form` | `basement_or_secondary_suite`, `standard` |
| `construction_type` | `new_construction_teardown`, `new_construction`, `garden_suite`, `conversion`, `addition`, `alteration`, `severance`, `unclear` |

Output columns: `permit_num`, `description_hash`, `description`, `unit_form`,
`construction_type`, `model`, `status`, `elapsed_ms`, `raw_response`.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

ollama pull qwen3.5:4b
```

Ollama must be running (`ollama serve`, or the desktop app).

## Usage

```powershell
# Benchmark first -- 100 permits tells you the real throughput on your machine
python classify_permits.py -i permits.csv -o classified.csv --limit 100

# Full run, resumable
python classify_permits.py -i permits.csv -o classified.csv --resume

# Cleared + active exports together (deduped across both)
python classify_permits.py -i cleared.csv active.csv -o classified.csv --resume
```

Ctrl-C is safe — results are flushed row by row. Re-run with `--resume` to continue,
adding `--retry-failed` to also redo any rows whose `status` isn't `ok`.

### Key flags

| Flag | Purpose |
|---|---|
| `--model` | Ollama model (default `qwen3.5:4b`) |
| `--parallel` | Concurrent requests (default 3) — the main throughput lever on CPU |
| `--limit N` | Classify only the first N descriptions |
| `--resume` / `--retry-failed` | Continue an interrupted run |
| `--source-status` | `cleared` additionally requires `STATUS='Closed'` |
| `--no-filter` | Classify every row, skipping the in-scope filter |
| `--num-ctx` | Context window (default 4096) |

## Input

A raw Building Permits export from Toronto Open Data
([cleared](https://open.toronto.ca/dataset/building-permits-cleared-permits/) /
[active](https://open.toronto.ca/dataset/building-permits-active-permits/)). Only
`PERMIT_NUM` and `DESCRIPTION` are strictly required; the rest drive the filter.

The script narrows to the in-scope population itself, mirroring what the warehouse
does downstream:

- latest `REVISION_NUM` per `PERMIT_NUM` (deduped across input files)
- `DWELLING_UNITS_CREATED > 0`
- `STRUCTURE_TYPE` not `SFD*` and not a non-residential type (Office, Hospital, …)
- `PROPOSED_USE` not matching `sfd` / `single`
- `STATUS` not Cancelled / Refused / Abandoned / Superseded / …
- non-empty `DESCRIPTION`

Pass `--no-filter` to bypass all of it.

## Performance

Built for a **CPU-only** machine. Numbers below were measured on an Intel Core Ultra
5 135U (12 cores, 15 W, no discrete GPU) with a single 16 GB DDR5-5600 stick, against
the real Toronto permit dataset (4,940 in-scope permits / 4,678 distinct descriptions).

Measured with `qwen3.5:4b`, per permit:

| Configuration | Per permit | Full dataset |
|---|---|---|
| 1 permit/request, verbose prompt + enum names | ~29 s | ~38 h |
| batch of 10, verbose | ~16.6 s | ~21 h |
| **batch of 10, compact codes (current default)** | **~10 s** | **~13 h** |
| same, on `qwen3.5:2b` | ~6 s | ~8 h |

So a full run is an overnight job. Use `--limit 100` to measure your own throughput
before committing to it.

### What actually moved the needle

**Batching is the biggest lever, and prefix caching does *not* replace it.** Ollama
re-read the ~1,000-token system prompt on every request — two back-to-back calls with
an identical system prompt both paid 18 s of prefill. At ~23–53 tok/s prefill on this
CPU, that preamble dominated everything else. `--batch-size 10` amortizes it across ten
permits.

**Compact output codes are worth ~2×.** Decoding runs at only ~3–6 tok/s, so the
response encoding is a first-order cost. The model answers `{"n":1,"f":"B","c":"L"}`
rather than spelling out `basement_or_secondary_suite`; the script expands the codes
before writing. Same answers, a third of the tokens.

**Thinking mode must stay off.** Reasoning traces cost 10–20× the output tokens for a
classification this small. The script sends `think: false` and strips any `<think>`
block that slips through.

**Parallelism helps much less than on a GPU** — one request already saturates the
cores — so `--parallel` defaults to 1. Raise it only if measurement on your machine
shows a gain.

**Deduplication is real but minor here**: only ~5% of descriptions repeat (262 of
4,940). It costs nothing, but don't expect it to rescue the runtime.

### Model sizing

Token generation is bound by memory bandwidth, so on-disk size predicts runtime almost
linearly — and the model must fit in RAM with room to spare, or the OS pages it off the
SSD and throughput collapses.

| Model | Size (Q4_K_M) | Fits in 16 GB? |
|---|---|---|
| `qwen3.5:2b` | 1.9 GB | yes — ~1.6× faster, lower accuracy |
| **`qwen3.5:4b`** | **3.4 GB** | **yes — default** |
| `gemma4:12b` | 7.6 GB | yes, ~2× slower |
| `gemma4:26b` | 18 GB | **no** — exceeds RAM, thrashes to disk |

Two hardware notes: close memory-hungry apps before a long run, and if the machine has
a free RAM slot, a second stick roughly doubles memory bandwidth on a single-DIMM
laptop — which roughly doubles tokens per second, and would also let a 26B model fit.
