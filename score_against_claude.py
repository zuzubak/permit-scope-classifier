"""Score local-model classifications against the Claude labels from to-multiplex-map.

The dbt seed there (dbt/seeds/llm_permit_scope.csv) holds ~5,100 permits already
classified by claude-haiku-4-5, keyed by permit_num. That makes it a ready-made
answer key for judging whether a local model is good enough to replace it -- no new
labelling required.

Treats the Claude labels as ground truth, which they are not exactly; disagreement
means "differs from the previous pipeline", not "wrong". The per-class breakdown is
the useful part, since it shows *where* a small model drifts rather than just how
often.

Usage:
    python score_against_claude.py --predictions data/sample_1pct.csv
"""
from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path

DEFAULT_LABELS = Path(__file__).resolve().parent.parent / "to-multiplex-map" / "dbt" / "seeds" / "llm_permit_scope.csv"


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def agreement_report(field: str, pairs: list[tuple[str, str]]) -> None:
    """pairs: (claude_value, local_value)."""
    total = len(pairs)
    agreed = sum(1 for a, b in pairs if a == b)
    print(f"\n{field}: {agreed}/{total} agree ({agreed / total * 100:.1f}%)" if total else f"\n{field}: no overlap")
    if not total:
        return

    # Per-class recall against the Claude label, plus the most common confusions.
    by_claude: dict[str, list[str]] = collections.defaultdict(list)
    for claude_value, local_value in pairs:
        by_claude[claude_value].append(local_value)

    print(f"  {'claude label':<28} {'n':>4}  {'agree':>6}   most common disagreement")
    for label in sorted(by_claude, key=lambda k: -len(by_claude[k])):
        predictions = by_claude[label]
        hits = sum(1 for p in predictions if p == label)
        misses = collections.Counter(p for p in predictions if p != label)
        worst = f"{misses.most_common(1)[0][0]} ({misses.most_common(1)[0][1]})" if misses else "-"
        print(f"  {label:<28} {len(predictions):>4}  {hits / len(predictions) * 100:>5.0f}%   {worst}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", "-p", type=Path, required=True, help="Output CSV from classify_permits.py.")
    parser.add_argument("--labels", "-l", type=Path, default=DEFAULT_LABELS, help="Claude-labelled seed CSV.")
    parser.add_argument("--show-disagreements", "-d", type=int, default=10, help="Print N example disagreements.")
    args = parser.parse_args()

    for path in (args.predictions, args.labels):
        if not path.exists():
            raise SystemExit(f"Not found: {path}")

    predictions = [r for r in load_csv(args.predictions) if r.get("status") == "ok"]
    labels = {r["permit_num"]: r for r in load_csv(args.labels)}

    matched = [(labels[r["permit_num"]], r) for r in predictions if r["permit_num"] in labels]
    print(f"{len(predictions)} local predictions, {len(labels)} Claude labels, {len(matched)} overlap.")
    if not matched:
        raise SystemExit("No permit_num overlap -- are these the same dataset?")

    model = matched[0][1].get("model", "?")
    print(f"Comparing {model} against {matched[0][0].get('model', '?')}.")

    for field in ("unit_form", "construction_type"):
        agreement_report(field, [(c.get(field, ""), l.get(field, "")) for c, l in matched])

    both = sum(1 for c, l in matched if c.get("unit_form") == l.get("unit_form")
               and c.get("construction_type") == l.get("construction_type"))
    print(f"\nboth fields agree: {both}/{len(matched)} ({both / len(matched) * 100:.1f}%)")

    if args.show_disagreements:
        print(f"\n--- up to {args.show_disagreements} disagreements ---")
        shown = 0
        for claude, local in matched:
            diffs = [f for f in ("unit_form", "construction_type") if claude.get(f) != local.get(f)]
            if not diffs:
                continue
            print(f"\n{local['permit_num']}: {local.get('description', '')[:150]}")
            for field in diffs:
                print(f"   {field}: claude={claude.get(field)!r} local={local.get(field)!r}")
            shown += 1
            if shown >= args.show_disagreements:
                break


if __name__ == "__main__":
    main()
