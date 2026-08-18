# Permit Scope Classifier

Classifies City of Toronto building permits by **unit form** and **construction scope**
using a local [Ollama](https://ollama.com) model. CSV in, CSV out.

Standalone fork of the `classify/` step in
[`to-multiplex-map`](../to-multiplex-map), which called the Claude API and wrote a dbt
seed. This version reads a raw permit export directly and runs offline.

## What it produces

Two **independent** judgments per permit:

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

### Full run, resumable
python classify_permits.py -i permits.csv -o classified.csv --resume

### Cleared + active exports together
python classify_permits.py -i cleared.csv active.csv -o classified.csv --resume
```

Ctrl-C is safe — results are flushed row by row. Re-run with `--resume` to continue,
adding `--retry-failed` to also redo any rows whose `status` isn't `ok`.
```

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
[active](https://open.toronto.ca/dataset/building-permits-active-permits/)) -- or
another export with a similar schema. Only `PERMIT_NUM` and `DESCRIPTION` are required.

The script narrows to the in-scope population to gentle density permits first:

- latest `REVISION_NUM` per `PERMIT_NUM` (deduped across input files)
- `DWELLING_UNITS_CREATED > 0`
- `STRUCTURE_TYPE` not `SFD*` and not a non-residential type (Office, Hospital, …)
- `PROPOSED_USE` not matching `sfd` / `single`
- `STATUS` not Cancelled / Refused / Abandoned / Superseded / …
- non-empty `DESCRIPTION`

Pass `--no-filter` to bypass all of that.

## Performance

Built for a **CPU-only** machine. Numbers below were measured on an Intel Core Ultra
5 135U (12 cores, 15 W, no discrete GPU) with a single 16 GB DDR5-5600 stick, against
the real Toronto permit dataset (4,940 gentle density permits / 4,678 distinct descriptions).

Measured with `qwen3.5:4b`, per permit:

| Configuration | Per permit | Full dataset |
|---|---|---|
| 1 permit/request, verbose prompt + enum names | ~29 s | ~38 h |
| batch of 10, verbose | ~16.6 s | ~21 h |
| **batch of 10, compact codes (current default)** | **~10 s** | **~13 h** |
| same, on `qwen3.5:2b` | ~6 s | ~8 h |

So a full run is an overnight job. Use `--limit n` to measure your own throughput
before committing to a full run.
