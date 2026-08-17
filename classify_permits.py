"""Classify Toronto building permits by unit form and construction scope using a
local Ollama model. CSV in, CSV out, no cloud API.

Two independent judgments per permit -- they are deliberately NOT collapsed into a
single enum, because that conflates a structural fact (is the extra unit a
basement/secondary suite tucked into a house) with a scope-of-work fact (was the
building itself newly built or altered). Those are orthogonal: a basement suite can
be part of a brand-new house just as easily as a retrofit into an old one, so a
description that mentions "basement" must not force an "alteration" conclusion.

  - unit_form:         basement_or_secondary_suite | standard
  - construction_type: new_construction_teardown | new_construction | garden_suite |
                       conversion | addition | alteration | severance | unclear

Input is a raw Building Permits export from Toronto Open Data (cleared and/or
active). The script applies the in-scope filter itself -- latest revision per
permit, created dwelling units, non-SFD, live status -- so it needs nothing but the
CSV.

Sizing note: this targets a CPU-only machine. Token generation is bound by memory
bandwidth, so model size drives runtime almost linearly. qwen3.5:4b (~3.4GB) is the
default because it fits alongside a normal desktop session; anything in the 26B+
range needs more RAM than a 16GB laptop has.

Usage:
    ollama serve                       # in another terminal
    ollama pull qwen3.5:4b
    python classify_permits.py --input permits.csv --output classified.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import requests

DEFAULT_MODEL = "qwen3.5:4b"
DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# The model answers in single-letter codes, which are expanded to full names before
# anything is written out. This is not cosmetic: on a CPU-only box decoding runs at
# ~3-6 tok/s, and spelling out "new_construction_teardown" costs ~10 tokens where "T"
# costs 1. Measured on the target machine, compact codes plus batching cut the
# per-permit cost from ~29s to ~7s -- roughly a 4x speedup for no loss of accuracy.
UNIT_FORM_CODES = {"B": "basement_or_secondary_suite", "S": "standard"}
CONSTRUCTION_CODES = {
    "T": "new_construction_teardown",
    "N": "new_construction",
    "G": "garden_suite",
    "V": "conversion",
    "A": "addition",
    "L": "alteration",
    "X": "severance",
    "U": "unclear",
}

# The values that appear in the output CSV. Derived so they can't drift from the maps.
UNIT_FORMS = sorted(UNIT_FORM_CODES.values())
CONSTRUCTION_TYPES = sorted(CONSTRUCTION_CODES.values())

OUTPUT_FIELDNAMES = [
    "permit_num",
    "description_hash",
    "description",
    "unit_form",
    "construction_type",
    "model",
    "status",
    "elapsed_ms",
    "raw_response",
]

# Statuses that mean the permit died -- it never became real work, so it should not
# count as created housing. Applies to both the cleared and active exports.
DEAD_STATUSES = {
    "cancelled",
    "refused",
    "refusal notice",
    "abandoned",
    "revocation pending",
    "revocation notice sent",
    "pending cancellation",
    "superseded",
}

# STRUCTURE_TYPE values that are non-residential despite reporting created units.
EXCLUDED_STRUCTURE_TYPES = {
    "office",
    "hospital",
    "restaurant 30 seats or less",
    "home for the aged",
    "motel/hotel",
    "place of worship",
    "apartment hotel",
}

# Deliberately terse. Every token here is re-read on each request -- Ollama does not
# reliably reuse the cached prefix across calls, so a 1000-token preamble cost ~18s of
# prefill per batch on the target machine. This version keeps the domain rules that
# actually change answers and drops the exposition.
SYSTEM_PROMPT = """Classify City of Toronto building permits. For each numbered permit output two INDEPENDENT judgments:

f = unit form:
  B = the added unit is a basement suite, or a "secondary/second suite"/"2nd dwelling unit" -- ONE accessory unit inside what is still a house-scaled building.
  S = standard -- a true duplex/triplex/fourplex with peer units, a laneway/garden suite, a larger multi-unit or mixed-use building.

c = scope of work:
  T = an existing building is demolished AND rebuilt.
  N = a new building is constructed, no demolition -- including when it also contains a basement/secondary suite.
  G = a new garden/laneway suite (small detached structure in a rear yard), including converting a garage into one.
  V = an EXISTING building's use is converted ("convert SFD to duplex", "change of use"), no demolition or addition.
  A = the building's FOOTPRINT or ENVELOPE physically grows -- a front/rear/side addition, an extension, an extra storey. The building gets bigger.
  L = interior work only, within the existing envelope -- the building does NOT get bigger. Default for a plain basement/secondary suite proposal.
  X = severing/splitting a lot.
  U = not enough information (truncated text, bare revision notes, administrative language).

Rules:
- f and c are INDEPENDENT. N+B is common (new house built with a basement apartment); L+B is also common (basement suite retrofitted into an old house). The word "basement" alone does NOT mean the building is existing.
- New-build signals: "construct new", "proposed construction", vacant lot. Existing-building signals: "existing", "add a unit to", "convert".
- "ADD a unit/suite/2nd suite" means adding a DWELLING, not building an addition. Choose A only if the text says the structure itself gets bigger (addition, extension, extra storey, enlarge). "Proposal to add a 2nd suite in the basement of an existing detached dwelling" = L, NOT A.
- Demolition AND construction together = T, even if it says "duplex".
- Garage-to-laneway-suite = G, not A.
- An addition to the MAIN house = A, not N -- the original building still stands.
- Descriptions are often ALL CAPS, truncated, or contain typos. Classify on whatever text is present; use U only when there is genuinely nothing to go on.

Classify EVERY numbered permit. Return one object per permit with n = its number. JSON only."""

BATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "r": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer"},
                    "f": {"type": "string", "enum": sorted(UNIT_FORM_CODES)},
                    "c": {"type": "string", "enum": sorted(CONSTRUCTION_CODES)},
                },
                "required": ["n", "f", "c"],
            },
        }
    },
    "required": ["r"],
}

THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


@dataclass
class Permit:
    permit_num: str
    description: str

    @property
    def description_hash(self) -> str:
        return hashlib.md5(self.description.strip().encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Input: read + filter the raw permit export
# ---------------------------------------------------------------------------


def _norm(value: str | None) -> str:
    return (value or "").strip()


def _try_int(value: str | None) -> int | None:
    try:
        return int(float(_norm(value)))
    except (TypeError, ValueError):
        return None


def read_permits(paths: list[Path], source_status: str, no_filter: bool) -> list[Permit]:
    """Read one or more raw permit exports and narrow to the in-scope population.

    Mirrors the filtering the downstream warehouse applies: latest revision per
    permit number, created dwelling units, not a single-family detached home, not a
    non-residential structure type, and not a dead permit.
    """
    # permit_num -> (revision, row). Dedupes across files too, so passing the
    # cleared and active exports together resolves to one row per permit.
    latest: dict[str, tuple[int, dict[str, str]]] = {}

    for path in paths:
        with path.open(newline="", encoding="utf-8-sig", errors="replace") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise SystemExit(f"{path} appears to be empty.")
            # CKAN exports use uppercase headers, but be tolerant of casing.
            colmap = {name.strip().upper(): name for name in reader.fieldnames}
            required = {"PERMIT_NUM", "DESCRIPTION"}
            missing = required - colmap.keys()
            if missing:
                raise SystemExit(
                    f"{path} is missing required column(s): {', '.join(sorted(missing))}.\n"
                    f"Found: {', '.join(reader.fieldnames)}"
                )

            def get(row: dict[str, str], col: str) -> str:
                source = colmap.get(col)
                return _norm(row.get(source)) if source else ""

            for row in reader:
                permit_num = get(row, "PERMIT_NUM")
                if not permit_num:
                    continue
                revision = _try_int(get(row, "REVISION_NUM")) or 0
                existing = latest.get(permit_num)
                if existing is None or revision >= existing[0]:
                    latest[permit_num] = (revision, {c: get(row, c) for c in colmap})

    permits: list[Permit] = []
    for permit_num, (_revision, row) in latest.items():
        description = row.get("DESCRIPTION", "")
        if not description:
            continue

        if not no_filter:
            units = _try_int(row.get("DWELLING_UNITS_CREATED"))
            if units is None or units <= 0:
                continue

            structure_type = row.get("STRUCTURE_TYPE", "")
            if structure_type.upper().startswith("SFD"):
                continue
            if structure_type.lower() in EXCLUDED_STRUCTURE_TYPES:
                continue

            proposed_use = row.get("PROPOSED_USE", "").lower()
            if "sfd" in proposed_use or "single" in proposed_use:
                continue

            status = row.get("STATUS", "").lower()
            if status in DEAD_STATUSES:
                continue
            if source_status == "cleared" and status != "closed":
                continue

        permits.append(Permit(permit_num=permit_num, description=description))

    permits.sort(key=lambda p: p.permit_num)
    return permits


# ---------------------------------------------------------------------------
# Output: append-as-you-go so a multi-hour run is resumable
# ---------------------------------------------------------------------------


def load_existing(path: Path, retry_failed: bool) -> set[str]:
    """Return the description_hashes already classified, for --resume."""
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if retry_failed and row.get("status") != "ok":
                continue
            key = row.get("description_hash")
            if key:
                done.add(key)
    return done


class ResultWriter:
    """Appends rows under a lock so concurrent workers can't interleave writes."""

    def __init__(self, path: Path, append: bool) -> None:
        self._lock = threading.Lock()
        is_new = not (append and path.exists() and path.stat().st_size > 0)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a" if append else "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=OUTPUT_FIELDNAMES)
        if is_new:
            self._writer.writeheader()
            self._fh.flush()

    def write(self, row: dict) -> None:
        with self._lock:
            self._writer.writerow({k: row.get(k, "") for k in OUTPUT_FIELDNAMES})
            self._fh.flush()

    def close(self) -> None:
        self._fh.close()


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------


class OllamaClient:
    def __init__(self, host: str, model: str, num_ctx: int, timeout: int) -> None:
        self.url = host.rstrip("/") + "/api/chat"
        self.model = model
        self.num_ctx = num_ctx
        self.timeout = timeout
        self.session = requests.Session()
        # Reasoning models burn 10-20x the output tokens on a classification this
        # small, which turns a 4-hour run into a multi-day one. Ask for it off, and
        # fall back if this build of Ollama rejects the field.
        self.send_think = True

    def classify(self, descriptions: list[str]) -> tuple[dict[int, tuple[str, str]], str, int]:
        """Classify a batch. Returns {0-based index: (unit_form, construction_type)}.

        Indexes missing from the response are simply absent from the dict -- the
        caller decides how to handle them, rather than losing the whole batch.
        """
        numbered = "\n".join(f"{i + 1}. {d}" for i, d in enumerate(descriptions))
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": numbered},
            ],
            "stream": False,
            "format": BATCH_SCHEMA,
            "options": {
                "temperature": 0,
                "num_ctx": self.num_ctx,
                # ~22 output tokens per permit in compact-code form, plus headroom.
                "num_predict": 64 + 40 * len(descriptions),
            },
        }
        if self.send_think:
            payload["think"] = False

        started = time.monotonic()
        response = self.session.post(self.url, json=payload, timeout=self.timeout)
        if response.status_code == 400 and self.send_think and "think" in response.text.lower():
            self.send_think = False
            payload.pop("think", None)
            response = self.session.post(self.url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        elapsed_ms = int((time.monotonic() - started) * 1000)

        content = response.json().get("message", {}).get("content", "")
        cleaned = THINK_TAG_RE.sub("", content).strip()

        results: dict[int, tuple[str, str]] = {}
        for item in json.loads(cleaned).get("r", []):
            index = item.get("n")
            unit_form = UNIT_FORM_CODES.get(item.get("f", ""))
            construction_type = CONSTRUCTION_CODES.get(item.get("c", ""))
            if not isinstance(index, int) or unit_form is None or construction_type is None:
                continue
            if 1 <= index <= len(descriptions):
                results[index - 1] = (unit_form, construction_type)
        return results, cleaned, elapsed_ms


def preflight(client: OllamaClient, host: str) -> None:
    try:
        tags = requests.get(host.rstrip("/") + "/api/tags", timeout=10)
        tags.raise_for_status()
    except requests.RequestException as exc:
        raise SystemExit(
            f"Can't reach Ollama at {host} ({exc}).\n"
            "Start it with `ollama serve` in another terminal, or set --host."
        )

    installed = {m.get("name", "") for m in tags.json().get("models", [])}
    # Ollama reports "qwen3.5:4b"; accept a bare name matching the default tag too.
    if client.model not in installed and f"{client.model}:latest" not in installed:
        raise SystemExit(
            f"Model {client.model!r} is not installed. Pull it first:\n"
            f"    ollama pull {client.model}\n"
            f"Installed: {', '.join(sorted(installed)) or '(none)'}"
        )


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify Toronto building permits with a local Ollama model.",
    )
    parser.add_argument(
        "--input", "-i", type=Path, nargs="+", required=True,
        help="Raw permit export CSV(s). Pass the cleared and active exports together "
             "to dedupe across both.",
    )
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output CSV.")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL}).")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Ollama host (default: {DEFAULT_HOST}).")
    parser.add_argument(
        "--batch-size", "-b", type=int, default=10,
        help="Permits per request. The system prompt is re-read on every request, so "
             "batching amortizes it -- the single biggest speed lever here. Large "
             "batches risk the model dropping items (default: 10).",
    )
    parser.add_argument(
        "--parallel", "-p", type=int, default=1,
        help="Concurrent requests. Helps far less on CPU than on GPU, since one "
             "request already saturates the cores; raise only if measurement shows a "
             "gain, and keep <= OLLAMA_NUM_PARALLEL (default: 1).",
    )
    parser.add_argument("--limit", type=int, help="Only classify the first N permits. Use this to benchmark first.")
    parser.add_argument("--resume", action="store_true", help="Skip permits already in the output CSV.")
    parser.add_argument("--retry-failed", action="store_true", help="With --resume, redo rows whose status isn't 'ok'.")
    parser.add_argument(
        "--source-status", choices=["auto", "cleared", "active"], default="auto",
        help="'cleared' additionally requires STATUS='Closed'. 'auto' just drops dead "
             "permits, which is correct for either export (default: auto).",
    )
    parser.add_argument("--no-filter", action="store_true", help="Classify every row; skip the in-scope filter.")
    parser.add_argument(
        "--num-ctx", type=int, default=8192,
        help="Context window. Must hold the system prompt plus a full batch; raise it "
             "if you raise --batch-size (default: 8192).",
    )
    parser.add_argument("--timeout", type=int, default=300, help="Per-request timeout in seconds (default: 300).")
    args = parser.parse_args()

    for path in args.input:
        if not path.exists():
            raise SystemExit(f"Input not found: {path}")

    permits = read_permits(args.input, args.source_status, args.no_filter)
    print(f"{len(permits)} permits in scope.")
    if not permits:
        return

    # Descriptions repeat heavily across permits (boilerplate phrasing, revisions of
    # the same project). Classify each distinct description once and fan the result
    # back out -- usually the single largest cut to total runtime.
    by_hash: dict[str, list[Permit]] = {}
    for permit in permits:
        by_hash.setdefault(permit.description_hash, []).append(permit)
    print(f"{len(by_hash)} distinct descriptions ({len(permits) - len(by_hash)} duplicates skipped).")

    already_done = load_existing(args.output, args.retry_failed) if args.resume else set()
    work = [(h, group[0].description) for h, group in by_hash.items() if h not in already_done]
    work.sort(key=lambda item: item[0])
    if already_done:
        print(f"{len(already_done)} already in {args.output}, {len(work)} remaining.")
    if args.limit:
        work = work[: args.limit]
        print(f"--limit {args.limit}: classifying {len(work)} descriptions.")
    if not work:
        print("Nothing to do.")
        return

    client = OllamaClient(args.host, args.model, args.num_ctx, args.timeout)
    preflight(client, args.host)

    writer = ResultWriter(args.output, append=args.resume)
    counters = {"ok": 0, "failed": 0, "permits": 0}
    counter_lock = threading.Lock()
    started = time.monotonic()

    def process(batch: list[tuple[str, str]]) -> None:
        try:
            results, raw, elapsed_ms = client.classify([d for _h, d in batch])
        except Exception as exc:  # noqa: BLE001 -- one bad batch must not kill a long run
            results, raw, elapsed_ms = {}, str(exc)[:500], 0

        per_item_ms = elapsed_ms // max(len(batch), 1)
        for i, (description_hash, description) in enumerate(batch):
            result = results.get(i)
            if result is None:
                # Dropped by the model, or the whole batch errored. Recorded as an
                # error row so --retry-failed can pick it up rather than silently
                # leaving a gap.
                unit_form, construction_type, status = "", "", "error"
            else:
                unit_form, construction_type = result
                status = "ok"

            # Fan the classification back out to every permit sharing the text.
            for permit in by_hash[description_hash]:
                writer.write({
                    "permit_num": permit.permit_num,
                    "description_hash": description_hash,
                    "description": description,
                    "unit_form": unit_form,
                    "construction_type": construction_type,
                    "model": args.model,
                    "status": status,
                    "elapsed_ms": per_item_ms,
                    "raw_response": raw if status == "error" else "",
                })

            with counter_lock:
                counters["ok" if status == "ok" else "failed"] += 1
                counters["permits"] += len(by_hash[description_hash])

        with counter_lock:
            done = counters["ok"] + counters["failed"]
            rate = done / max(time.monotonic() - started, 1e-9)
            remaining = (len(work) - done) / rate if rate else 0
            print(
                f"  {done}/{len(work)} descriptions ({counters['failed']} failed) "
                f"{rate * 60:.1f}/min, ~{remaining / 3600:.1f}h left",
                flush=True,
            )

    batches = [work[i : i + args.batch_size] for i in range(0, len(work), args.batch_size)]
    print(
        f"Classifying with {args.model}: {len(batches)} batches of up to "
        f"{args.batch_size}, {args.parallel}x concurrency..."
    )
    try:
        with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
            list(pool.map(process, batches))
    except KeyboardInterrupt:
        print("\nInterrupted -- partial results are saved. Re-run with --resume.", file=sys.stderr)
        return
    finally:
        writer.close()

    elapsed = time.monotonic() - started
    print(
        f"Done in {elapsed / 60:.1f} min. "
        f"{counters['ok']} descriptions classified, {counters['failed']} failed, "
        f"covering {counters['permits']} permits -> {args.output}"
    )
    if counters["failed"]:
        print("Re-run with --resume --retry-failed to retry the failures.")


if __name__ == "__main__":
    main()
