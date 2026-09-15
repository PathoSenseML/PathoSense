"""Independent conformance check of a finals JSON against the TEKNOFEST guide (§3, §3.2).

Deliberately shares NO code with predict.py: every rule below is transcribed from the
guide text, so a bug in the exporter cannot hide itself. Stdlib only.

    python pipelines/check_submission.py TEAM_924489_FINAL.json test1.csv [test2.csv ...]

Exit 0 = every written rule holds. Exit 1 = a rule is violated (printed).
"""
import csv
import json
import math
import re
import sys


def _reject_constant(name):  # NaN / Infinity / -Infinity are not JSON; the guide forbids them
    raise ValueError(f"forbidden JSON constant {name}")


def check(json_path, csv_paths, team_id=None):
    problems = []
    # --- file level: UTF-8, .json, recommended name, parses strictly, no comment lines
    if not json_path.lower().endswith(".json"):
        problems.append("file extension is not .json")
    raw = open(json_path, "rb").read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        return [f"not valid UTF-8: {e}"]
    if any(line.lstrip().startswith(("//", "#")) for line in text.splitlines()):
        problems.append("comment-like line present")
    try:
        obj = json.loads(text, parse_constant=_reject_constant)
    except ValueError as e:   # broken JSON structure, or NaN/Infinity literal
        return [f"JSON does not parse strictly: {e}"]
    # --- header fields
    for key in ("team_name", "team_id", "application_id", "competition_level"):
        if not isinstance(obj.get(key), str) or not obj.get(key).strip():
            problems.append(f"header field {key} missing or empty")
    if obj.get("competition_level") != "UNIVERSITE_VE_UZERI":
        problems.append(f"competition_level must be UNIVERSITE_VE_UZERI, got {obj.get('competition_level')!r}")
    tid = obj.get("team_id", "")
    if team_id is not None and tid != team_id:
        problems.append(f"team_id {tid!r} != expected {team_id!r}")
    m = re.match(r"^TEAM_(.+)$", tid)
    expected_name = f"TEAM_{m.group(1) if m else tid}_FINAL.json"
    if not json_path.endswith(expected_name):
        problems.append(f"file name should be {expected_name} (recommended by the guide)")
    # --- ids: every test id exactly once, unchanged, none extra
    test_ids = []
    for p in csv_paths:
        with open(p, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        col = next((c for c in ("id", "Variant_ID", "variant_id", "ID") if rows and c in rows[0]), None)
        if col is None:
            problems.append(f"{p}: no id column"); continue
        test_ids += [r[col] for r in rows]
    if len(set(test_ids)) != len(test_ids):
        problems.append("input files themselves repeat an id")
    preds = obj.get("predictions")
    if not isinstance(preds, list):
        return problems + ["predictions is not a list"], {}
    seen = [r.get("id") for r in preds]
    if len(seen) != len(set(seen)):
        problems.append("duplicate id in predictions")
    missing = set(test_ids) - set(seen)
    extra = set(seen) - set(test_ids)
    if missing:
        problems.append(f"{len(missing)} test ids missing, e.g. {sorted(missing)[:3]}")
    if extra:
        problems.append(f"{len(extra)} ids not in the test files, e.g. {sorted(extra)[:3]}")
    if len(preds) != len(test_ids):
        problems.append(f"{len(preds)} predictions for {len(test_ids)} test rows")
    # --- per record
    panels = {}
    for i, r in enumerate(preds):
        if r.get("predicted_class") not in ("0", "1"):
            problems.append(f"row {i}: predicted_class must be '0' or '1'"); break
        q = r.get("predicted_prob")
        if not isinstance(q, (int, float)) or isinstance(q, bool) or not math.isfinite(q) or not 0 <= q <= 1:
            problems.append(f"row {i}: predicted_prob not a number in [0,1]"); break
        # §3.2 university template: exactly id, panel, predicted_class, predicted_prob.
        # The §3 common-rule `probabilities` dict is tolerated as an optional extra and
        # validated when present (classes '0','1', values in [0,1], sum 1 ±0.001).
        extra = set(r) - {"id", "panel", "predicted_class", "predicted_prob", "probabilities"}
        if extra:
            problems.append(f"row {i}: fields not in the §3.2 template: {sorted(extra)}"); break
        if "probabilities" in r:
            pr = r["probabilities"]
            if not isinstance(pr, dict) or set(pr) != {"0", "1"}:
                problems.append(f"row {i}: probabilities must be a dict with exactly '0' and '1'"); break
            if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                       and 0 <= v <= 1 for v in pr.values()):
                problems.append(f"row {i}: probability value not a number in [0,1]"); break
            if abs(pr["0"] + pr["1"] - 1) > 0.001:
                problems.append(f"row {i}: probabilities do not sum to 1 (±0.001)"); break
        if not isinstance(r.get("panel"), str) or not r["panel"].strip():
            problems.append(f"row {i}: panel missing"); break
        panels[r["panel"]] = panels.get(r["panel"], 0) + 1
    # --- decimal separator: a comma inside a number literal cannot survive json.loads, but
    # a string-typed number could; enforce that probabilities are numbers, not strings (done above).
    if "," in re.sub(r'"(?:[^"\\]|\\.)*"', "", text).replace(", ", "").replace(",\n", ""):
        problems.append("suspicious comma outside strings/separators (decimal comma?)")
    return problems, panels


def main():
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(2)
    result = check(sys.argv[1], sys.argv[2:])
    problems, panels = (result, {}) if isinstance(result, list) else result
    for p in problems:
        print("FAIL:", p)
    if not problems:
        print("OK: every written guide rule holds. Panels:", panels)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
