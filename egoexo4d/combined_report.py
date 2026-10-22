"""Combine per-scenario ``final_report.json`` files into overall tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from egoexo4d.paths import REPO_ROOT, load_config, manifests_dir, output_root
from egoexo4d.report import format_bytes, pct
from egoexo4d.scenarios import SCENARIOS, SLUGS


def _get(d: Any, *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def _int(value: Any, default: int = 0) -> int:
    if value in (None, "", "n/a"):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def scenario_summaries(
    reports: Dict[str, Optional[dict]],
) -> List[dict]:
    rows: List[dict] = []
    for slug in SLUGS:
        report = reports.get(slug)
        spec = SCENARIOS[slug]
        if not report:
            rows.append(
                {
                    "slug": slug,
                    "name": spec["parent_task_name"],
                    "status": "missing",
                }
            )
            continue
        takes = _get(report, "takes", default={}) or {}
        index = takes.get("index_stats") or {}
        dataset = _get(report, "dataset", default={}) or {}
        frames = _get(report, "frames", default={}) or {}
        fusion = _get(report, "fusion", default={}) or {}
        disk = _get(report, "disk", default={}) or {}
        splits = _get(report, "splits", "splits", default={}) or {}
        vocab = _get(report, "vocab", default={}) or {}
        prune = fusion.get("prune") or {}
        buckets = dataset.get("label_frequency_buckets") or {}
        rows.append(
            {
                "slug": slug,
                "name": str(
                    index.get("scenario")
                    or spec["parent_task_name"]
                ),
                "status": "ok",
                "in_takes_json": _int(
                    index.get("scenario_in_takes_json", index.get("cooking_in_takes_json"))
                ),
                "in_relations": _int(index.get("in_relations")),
                "with_mps": _int(index.get("with_mps")),
                "with_calib": _int(index.get("with_calib")),
                "with_vrs": _int(index.get("with_vrs")),
                "with_exo": _int(index.get("with_exo_video")),
                "selected_takes": _int(
                    takes.get("selected_takes", index.get("selected"))
                ),
                "zero_tier_a": _int(dataset.get("takes_with_zero_tier_a_rows")),
                "annotated_frames": _int(frames.get("relations_annotated_frames")),
                "frames_with_gaze": _int(frames.get("frames_with_gaze")),
                "frames_in_mask": _int(frames.get("frames_gaze_in_object_mask")),
                "frames_kept": _int(frames.get("frames_kept_after_head_filter")),
                "exported_images": _int(frames.get("exported_images")),
                "csv_rows": _int(dataset.get("rows", frames.get("dataset_rows"))),
                "unique_images": _int(dataset.get("unique_images")),
                "unique_labels": _int(dataset.get("unique_labels")),
                "label_min": _int(dataset.get("label_count_min")),
                "label_max": _int(dataset.get("label_count_max")),
                "rare_labels": _int(buckets.get("eq_1_to_4")),
                "missing_images": _int(dataset.get("missing_images")),
                "gaze_in_box_rows": _int(dataset.get("gaze_in_box_rows")),
                "gaze_in_box_pct": dataset.get("gaze_in_box_pct"),
                "prune_dropped_rows": _int(prune.get("dropped_rows")),
                "prune_dropped_labels": _int(prune.get("dropped_labels")),
                "train_rows": _int(_get(splits, "train", "rows")),
                "val_rows": _int(_get(splits, "val", "rows")),
                "test_rows": _int(_get(splits, "test", "rows")),
                "train_images": _int(_get(splits, "train", "unique_images")),
                "val_images": _int(_get(splits, "val", "unique_images")),
                "test_images": _int(_get(splits, "test", "unique_images")),
                "leakage_ok": bool(_get(report, "splits", "leakage_ok", default=True)),
                "vocab_size": _int(vocab.get("vocab_size")),
                "embed_exists": bool(vocab.get("embed_exists")),
                "embed_matches": vocab.get("embed_matches_vocab"),
                "jpeg_count": _int(disk.get("jpeg_count")),
                "extracted_bytes": _int(disk.get("extracted_bytes")),
                "raw_bytes": _int(disk.get("raw_bytes")),
                "output_root": str(report.get("output_root") or spec["output_root"]),
            }
        )
    return rows


def overall_summary(rows: Sequence[dict], *, union_labels: Optional[int] = None) -> dict:
    ok = [r for r in rows if r.get("status") == "ok"]
    def s(key: str) -> int:
        return sum(_int(r.get(key)) for r in ok)

    gaze_rows = s("gaze_in_box_rows")
    csv_rows = s("csv_rows")
    raw_bytes = next((_int(r.get("raw_bytes")) for r in ok if r.get("raw_bytes")), 0)
    return {
        "scenarios_ok": len(ok),
        "scenarios_missing": sum(1 for r in rows if r.get("status") != "ok"),
        "selected_takes": s("selected_takes"),
        "in_relations": s("in_relations"),
        "csv_rows": csv_rows,
        "unique_images": s("unique_images"),
        "jpeg_count": s("jpeg_count"),
        "unique_labels_sum": s("unique_labels"),
        "unique_labels_union": union_labels,
        "rare_labels": s("rare_labels"),
        "missing_images": s("missing_images"),
        "zero_tier_a": s("zero_tier_a"),
        "annotated_frames": s("annotated_frames"),
        "frames_with_gaze": s("frames_with_gaze"),
        "frames_in_mask": s("frames_in_mask"),
        "exported_images": s("exported_images"),
        "train_rows": s("train_rows"),
        "val_rows": s("val_rows"),
        "test_rows": s("test_rows"),
        "extracted_bytes": s("extracted_bytes"),
        "raw_bytes": raw_bytes,
        "gaze_in_box_pct": pct(gaze_rows, csv_rows),
        "prune_dropped_rows": s("prune_dropped_rows"),
        "embeds_ok": sum(1 for r in ok if r.get("embed_exists")),
        "leakage_ok": all(r.get("leakage_ok", True) for r in ok) if ok else True,
    }


def load_reports_from_disk() -> Dict[str, Optional[dict]]:
    out: Dict[str, Optional[dict]] = {}
    for slug in SLUGS:
        cfg = load_config(REPO_ROOT / SCENARIOS[slug]["config"])
        path = manifests_dir(cfg) / "final_report.json"
        if not path.exists():
            out[slug] = None
            continue
        out[slug] = json.loads(path.read_text())
    return out


def union_label_count() -> Optional[int]:
    labels: set[str] = set()
    found = False
    for slug in SLUGS:
        cfg = load_config(REPO_ROOT / SCENARIOS[slug]["config"])
        vocab_path = output_root(cfg) / "vocab2id.json"
        if not vocab_path.exists():
            continue
        found = True
        vocab = json.loads(vocab_path.read_text())
        if isinstance(vocab, dict):
            labels.update(str(k) for k in vocab)
        elif isinstance(vocab, list):
            labels.update(str(x) for x in vocab)
    return len(labels) if found else None


def rebuild_reports(*, skip_raw_frames: bool = True) -> Dict[str, Optional[dict]]:
    from egoexo4d.report import build_report, write_report

    out: Dict[str, Optional[dict]] = {}
    for slug in SLUGS:
        cfg = load_config(REPO_ROOT / SCENARIOS[slug]["config"])
        csv_path = output_root(cfg) / "annotations" / "train_verified.csv"
        if not csv_path.exists():
            out[slug] = None
            continue
        report = build_report(cfg, skip_raw_frames=skip_raw_frames)
        write_report(cfg, report)
        out[slug] = report
    return out


def _fmt(value: Any, *, kind: str = "int") -> str:
    if value is None or value == "":
        return "—"
    if kind == "int":
        return f"{_int(value):,}"
    if kind == "pct":
        try:
            return f"{float(value):.1f}%"
        except (TypeError, ValueError):
            return "—"
    if kind == "bytes":
        return format_bytes(_int(value))
    if kind == "bool":
        if value is True:
            return "yes"
        if value is False:
            return "no"
        return "—"
    return str(value)


def _cell(row: dict, key: str, *, kind: str = "int") -> str:
    if row.get("status") != "ok":
        return "—"
    return _fmt(row.get(key), kind=kind)


def ascii_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [len(h) for h in headers]
    body = [[str(c) for c in row] for row in rows]
    for row in body:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    def fmt_row(vals: Sequence[str]) -> str:
        return "  ".join(val.ljust(widths[i]) for i, val in enumerate(vals))
    rule = "  ".join("-" * w for w in widths)
    return "\n".join([fmt_row(headers), rule, *[fmt_row(r) for r in body]])


def format_tables(rows: Sequence[dict], overall: dict) -> str:
    scale = ascii_table(
        ["metric", *[r["name"] for r in rows], "Overall"],
        [
            ["status", *[r.get("status", "missing") for r in rows], f"{overall['scenarios_ok']}/6 ok"],
            ["selected takes", *[_cell(r, "selected_takes") for r in rows], _fmt(overall["selected_takes"])],
            ["CSV rows", *[_cell(r, "csv_rows") for r in rows], _fmt(overall["csv_rows"])],
            ["unique images", *[_cell(r, "unique_images") for r in rows], _fmt(overall["unique_images"])],
            ["JPEGs on disk", *[_cell(r, "jpeg_count") for r in rows], _fmt(overall["jpeg_count"])],
            ["labels (scenario)", *[_cell(r, "unique_labels") for r in rows], _fmt(overall["unique_labels_sum"])],
            ["labels (union)", *["—"] * len(rows), _fmt(overall.get("unique_labels_union"))],
            ["extracted size", *[_cell(r, "extracted_bytes", kind="bytes") for r in rows], _fmt(overall["extracted_bytes"], kind="bytes")],
        ],
    )
    takes = ascii_table(
        ["scenario", "in takes.json", "in Relations", "MPS", "calib", "VRS", "exo", "selected", "zero Tier-A"],
        [
            [
                r["name"],
                _cell(r, "in_takes_json"),
                _cell(r, "in_relations"),
                _cell(r, "with_mps"),
                _cell(r, "with_calib"),
                _cell(r, "with_vrs"),
                _cell(r, "with_exo"),
                _cell(r, "selected_takes"),
                _cell(r, "zero_tier_a"),
            ]
            for r in rows
        ]
        + [[
            "Overall",
            "—",
            _fmt(overall["in_relations"]),
            "—", "—", "—", "—",
            _fmt(overall["selected_takes"]),
            _fmt(overall["zero_tier_a"]),
        ]],
    )
    frames = ascii_table(
        ["scenario", "annotated", "with gaze", "gaze in mask", "exported", "CSV rows"],
        [
            [
                r["name"],
                _cell(r, "annotated_frames"),
                _cell(r, "frames_with_gaze"),
                _cell(r, "frames_in_mask"),
                _cell(r, "exported_images"),
                _cell(r, "csv_rows"),
            ]
            for r in rows
        ]
        + [[
            "Overall",
            _fmt(overall["annotated_frames"]),
            _fmt(overall["frames_with_gaze"]),
            _fmt(overall["frames_in_mask"]),
            _fmt(overall["exported_images"]),
            _fmt(overall["csv_rows"]),
        ]],
    )
    quality = ascii_table(
        ["scenario", "labels", "min", "max", "rare (1–4)", "pruned rows", "gaze in bbox", "missing JPEG"],
        [
            [
                r["name"],
                _cell(r, "unique_labels"),
                _cell(r, "label_min"),
                _cell(r, "label_max"),
                _cell(r, "rare_labels"),
                _cell(r, "prune_dropped_rows"),
                _cell(r, "gaze_in_box_pct", kind="pct"),
                _cell(r, "missing_images"),
            ]
            for r in rows
        ]
        + [[
            "Overall",
            _fmt(overall.get("unique_labels_union") or overall["unique_labels_sum"]),
            "—",
            "—",
            _fmt(overall["rare_labels"]),
            _fmt(overall["prune_dropped_rows"]),
            _fmt(overall["gaze_in_box_pct"], kind="pct"),
            _fmt(overall["missing_images"]),
        ]],
    )
    splits = ascii_table(
        ["scenario", "train rows", "val rows", "test rows", "embed", "leakage_ok"],
        [
            [
                r["name"],
                _cell(r, "train_rows"),
                _cell(r, "val_rows"),
                _cell(r, "test_rows"),
                _cell(r, "embed_exists", kind="bool"),
                _cell(r, "leakage_ok", kind="bool"),
            ]
            for r in rows
        ]
        + [[
            "Overall",
            _fmt(overall["train_rows"]),
            _fmt(overall["val_rows"]),
            _fmt(overall["test_rows"]),
            f"{overall['embeds_ok']}/{overall['scenarios_ok']}" if overall["scenarios_ok"] else "—",
            _fmt(overall["leakage_ok"], kind="bool"),
        ]],
    )
    missing = [r["name"] for r in rows if r.get("status") != "ok"]
    miss_line = (
        f"Missing reports: {', '.join(missing)}"
        if missing
        else "All six scenario reports present."
    )
    raw = _fmt(overall.get("raw_bytes"), kind="bytes")
    return "\n".join(
        [
            "========== Ego-Exo4D combined dataset ==========",
            "",
            f"Shared raw dump: {raw}  (counted once; not summed across scenarios)",
            "Overall unique images/rows are sums of per-scenario datasets (takes do not overlap).",
            "Overall labels (union) is the set union of vocab2id.json files when those exist.",
            miss_line,
            "",
            "SCALE",
            scale,
            "",
            "TAKES",
            takes,
            "",
            "FRAMES",
            frames,
            "",
            "LABELS / QUALITY",
            quality,
            "",
            "SPLITS",
            splits,
            "",
        ]
    )


def format_html(rows: Sequence[dict], overall: dict) -> str:
    def th(vals: Iterable[str]) -> str:
        return "<tr>" + "".join(f"<th>{v}</th>" for v in vals) + "</tr>"

    def td(vals: Iterable[str]) -> str:
        return "<tr>" + "".join(f"<td>{v}</td>" for v in vals) + "</tr>"

    def section(title: str, headers: Sequence[str], body: Sequence[Sequence[str]]) -> str:
        return (
            f"<h2>{title}</h2><table><thead>{th(headers)}</thead>"
            f"<tbody>{''.join(td(r) for r in body)}</tbody></table>"
        )

    kpis = [
        ("Scenarios", f"{overall['scenarios_ok']}/6"),
        ("Selected takes", _fmt(overall["selected_takes"])),
        ("CSV rows", _fmt(overall["csv_rows"])),
        ("Images", _fmt(overall["unique_images"])),
        ("Labels (union)", _fmt(overall.get("unique_labels_union") or overall["unique_labels_sum"])),
        ("Gaze in bbox", _fmt(overall["gaze_in_box_pct"], kind="pct")),
        ("Extracted", _fmt(overall["extracted_bytes"], kind="bytes")),
        ("Raw dump", _fmt(overall.get("raw_bytes"), kind="bytes")),
    ]
    kpi_html = "".join(
        f"<div class='kpi'><div class='k'>{k}</div><div class='v'>{v}</div></div>"
        for k, v in kpis
    )
    scale = section(
        "Scale by scenario",
        ["Scenario", "Takes", "CSV rows", "Images", "JPEGs", "Labels", "Extracted"],
        [
            [
                r["name"] + ("" if r.get("status") == "ok" else " (missing)"),
                _cell(r, "selected_takes"),
                _cell(r, "csv_rows"),
                _cell(r, "unique_images"),
                _cell(r, "jpeg_count"),
                _cell(r, "unique_labels"),
                _cell(r, "extracted_bytes", kind="bytes"),
            ]
            for r in rows
        ]
        + [[
            "Overall",
            _fmt(overall["selected_takes"]),
            _fmt(overall["csv_rows"]),
            _fmt(overall["unique_images"]),
            _fmt(overall["jpeg_count"]),
            _fmt(overall.get("unique_labels_union") or overall["unique_labels_sum"]),
            _fmt(overall["extracted_bytes"], kind="bytes"),
        ]],
    )
    takes = section(
        "Take coverage",
        ["Scenario", "takes.json", "Relations", "MPS", "calib", "VRS", "exo", "selected", "zero Tier-A"],
        [
            [
                r["name"],
                _cell(r, "in_takes_json"),
                _cell(r, "in_relations"),
                _cell(r, "with_mps"),
                _cell(r, "with_calib"),
                _cell(r, "with_vrs"),
                _cell(r, "with_exo"),
                _cell(r, "selected_takes"),
                _cell(r, "zero_tier_a"),
            ]
            for r in rows
        ],
    )
    frames = section(
        "Frame funnel",
        ["Scenario", "Annotated", "With gaze", "Gaze in mask", "Exported", "CSV rows"],
        [
            [
                r["name"],
                _cell(r, "annotated_frames"),
                _cell(r, "frames_with_gaze"),
                _cell(r, "frames_in_mask"),
                _cell(r, "exported_images"),
                _cell(r, "csv_rows"),
            ]
            for r in rows
        ]
        + [[
            "Overall",
            _fmt(overall["annotated_frames"]),
            _fmt(overall["frames_with_gaze"]),
            _fmt(overall["frames_in_mask"]),
            _fmt(overall["exported_images"]),
            _fmt(overall["csv_rows"]),
        ]],
    )
    quality = section(
        "Labels and quality",
        ["Scenario", "Labels", "Min count", "Max count", "Rare (1–4)", "Pruned rows", "Gaze in bbox", "Missing JPEG"],
        [
            [
                r["name"],
                _cell(r, "unique_labels"),
                _cell(r, "label_min"),
                _cell(r, "label_max"),
                _cell(r, "rare_labels"),
                _cell(r, "prune_dropped_rows"),
                _cell(r, "gaze_in_box_pct", kind="pct"),
                _cell(r, "missing_images"),
            ]
            for r in rows
        ]
        + [[
            "Overall",
            _fmt(overall.get("unique_labels_union") or overall["unique_labels_sum"]),
            "—",
            "—",
            _fmt(overall["rare_labels"]),
            _fmt(overall["prune_dropped_rows"]),
            _fmt(overall["gaze_in_box_pct"], kind="pct"),
            _fmt(overall["missing_images"]),
        ]],
    )
    splits = section(
        "Splits and embeddings",
        ["Scenario", "Train", "Val", "Test", "Embed", "Leakage OK"],
        [
            [
                r["name"],
                _cell(r, "train_rows"),
                _cell(r, "val_rows"),
                _cell(r, "test_rows"),
                _cell(r, "embed_exists", kind="bool"),
                _cell(r, "leakage_ok", kind="bool"),
            ]
            for r in rows
        ]
        + [[
            "Overall",
            _fmt(overall["train_rows"]),
            _fmt(overall["val_rows"]),
            _fmt(overall["test_rows"]),
            f"{overall['embeds_ok']}/{overall['scenarios_ok']}" if overall["scenarios_ok"] else "—",
            _fmt(overall["leakage_ok"], kind="bool"),
        ]],
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Ego-Exo4D combined dataset</title>
<style>
  body {{ font-family: ui-sans-serif, system-ui, sans-serif; margin: 24px; color: #111; }}
  h1 {{ font-size: 20px; font-weight: 650; margin: 0 0 8px; }}
  h2 {{ font-size: 15px; font-weight: 650; margin: 28px 0 8px; }}
  p.note {{ color: #555; font-size: 13px; margin: 0 0 16px; max-width: 72rem; }}
  .kpis {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 16px 0 8px; }}
  .kpi {{ border: 1px solid #ddd; padding: 10px 12px; min-width: 7.5rem; }}
  .kpi .k {{ font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: .04em; }}
  .kpi .v {{ font-size: 18px; font-weight: 650; margin-top: 2px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid #e5e5e5; white-space: nowrap; }}
  thead th {{ font-size: 11px; text-transform: uppercase; letter-spacing: .03em; color: #555; }}
  tbody tr:last-child th, tbody tr:last-child td {{ font-weight: 650; }}
</style>
</head>
<body>
<h1>Ego-Exo4D Relations dataset</h1>
<p class="note">Per-scenario figures come from each <code>final_report.json</code>.
Overall counts sum the six datasets (takes do not overlap). Label union uses
<code>vocab2id.json</code>. Raw dump size is shared and counted once.</p>
<div class="kpis">{kpi_html}</div>
{scale}{takes}{frames}{quality}{splits}
</body>
</html>
"""


def build_combined(
    reports: Optional[Dict[str, Optional[dict]]] = None,
    *,
    union_labels: Optional[int] = None,
) -> dict:
    if reports is None:
        reports = load_reports_from_disk()
    rows = scenario_summaries(reports)
    if union_labels is None:
        try:
            union_labels = union_label_count()
        except OSError:
            union_labels = None
    overall = overall_summary(rows, union_labels=union_labels)
    return {
        "scenarios": rows,
        "overall": overall,
        "text": format_tables(rows, overall),
        "html": format_html(rows, overall),
    }


def write_combined(payload: dict, prefix: Path) -> Dict[str, Path]:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    txt = prefix.with_suffix(".txt")
    html = prefix.with_suffix(".html")
    js = prefix.with_suffix(".json")
    txt.write_text(payload["text"] + "\n")
    html.write_text(payload["html"])
    serial = {"scenarios": payload["scenarios"], "overall": payload["overall"]}
    js.write_text(json.dumps(serial, indent=2, default=str) + "\n")
    return {"txt": txt, "html": html, "json": js}
