"""Assemble a separate baseline package and prove same-machine offline JSON replay.

Never edits production files; fixtures are synthetic, outside the package. This
is not a clean-machine dependency-install test or organizer schema certification.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PANEL_FILES = {"general": "MASTER", "cancer": "KANSER", "pah": "PAH", "cftr": "CFTR"}
COUNTS = {"general": 3500, "cancer": 600, "pah": 350, "cftr": 120}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_hashes(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*"))
            if p.is_file() and p.suffix in (".py", ".joblib", ".txt", ".md", ".json")}


def child(package, fixture, destination, original_netns):
    import numpy as np
    import pandas as pd
    from threadpoolctl import threadpool_limits

    package = Path(package).resolve()
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    netns = os.readlink("/proc/self/ns/net")
    assert netns != original_netns, "process not in a separate network namespace"
    try:
        socket.create_connection(("198.51.100.1", 9), timeout=.2)
    except OSError as exc:
        network_error = exc.errno
        assert network_error in (errno.ENETUNREACH, errno.EHOSTUNREACH, errno.EPERM)
    else:
        raise AssertionError("isolated process unexpectedly reached a network")

    sys.path.insert(0, str(package / "pipelines"))
    import predict
    # Rehearsal-only candidate wrapper changes inference scheduling, not weights.
    sys.path.insert(0, str(package))
    import deterministic_predict
    deterministic_predict.configure()
    deterministic_predict.preflight(str(fixture))
    assert Path(predict.__file__).resolve() == package / "pipelines" / "predict.py"
    assert Path(predict.MODELS_DIR).resolve() == package / "models"
    started = time.perf_counter()
    options = dict(team_name="SYNTHETIC REHEARSAL", team_id="SYNTHETIC",
                   application_id="SYNTHETIC", prob_mode="p1", with_probabilities=True)
    with threadpool_limits(limits=4):
        first, second = destination / "first.json", destination / "second.json"
        predict.submit(str(fixture), output=str(first), **options)
        predict.submit(str(fixture), output=str(second), **options)
        assert first.read_bytes() == second.read_bytes(), "JSON replay differs"
        df = pd.read_csv(fixture)
        obj = json.loads(first.read_text())
        rows = obj["predictions"]
        assert len(rows) == len(df) and {r["id"] for r in rows} == set(df.id)
        expected = {r["id"]: r for r in rows}
        for row in rows:
            assert set(row["probabilities"]) == {"0", "1"}
            assert np.isfinite(row["predicted_prob"])
            assert row["predicted_prob"] == row["probabilities"]["1"]
            assert 0 <= row["predicted_prob"] <= 1
            assert abs(sum(row["probabilities"].values()) - 1) < 1e-12
        max_delta = 0.
        for panel in PANEL_FILES:
            art = predict.load_artifacts(panel)
            subset = df.loc[df.panel.str.lower() == panel]
            p = predict.score_frame(subset, art)["probs"]
            reference = np.array([expected[i]["predicted_prob"] for i in subset.id])
            np.testing.assert_array_equal(p, reference)
            for ident, prob in zip(subset.id, p):
                assert expected[ident]["predicted_class"] == str(int(prob >= art["threshold"]))
            small = subset.iloc[:6]
            order = [5, 2, 0, 4, 1, 3]
            shuffled = predict.score_frame(small.iloc[order], art)["probs"]
            np.testing.assert_array_equal(shuffled, p[:6][order])
            for i in (0, 1, 2):
                single = predict.score_frame(small.iloc[[i]], art)["probs"][0]
                max_delta = max(max_delta, float(abs(single - p[i])))
                assert single == p[i], "single-row versus batch drift"

        negatives = {}
        base = df.groupby("panel", sort=False).head(2).copy().reset_index(drop=True)
        for name in ("duplicate_id", "empty_id", "unknown_panel", "malformed_numeric", "missing_feature_column"):
            bad = base.copy()
            if name == "duplicate_id": bad.loc[1, "id"] = bad.loc[0, "id"]
            elif name == "empty_id": bad.loc[0, "id"] = ""
            elif name == "unknown_panel": bad.loc[0, "panel"] = "UNKNOWN"
            elif name == "missing_feature_column": bad = bad.drop(columns=["EK_7"])
            else: bad["EK_7"] = "not-a-number"
            path = destination / f"negative_{name}.csv"
            path.write_text(bad.to_csv(index=False))
            try:
                deterministic_predict.preflight(str(path))
                predict.submit(str(path), output=str(destination / f"negative_{name}.json"), **options)
            except (ValueError, TypeError, KeyError) as exc:
                negatives[name] = {"rejected": True, "error": str(exc)[:240]}
            else:
                negatives[name] = {"rejected": False}
        # A file with IDs/panels but no features must not silently look healthy.
        missing = destination / "missing_features.csv"
        base[["id", "panel"]].to_csv(missing, index=False)
        try:
            deterministic_predict.preflight(str(missing))
            predict.submit(str(missing), output=str(destination / "missing_features.json"), **options)
        except Exception as exc:
            negatives["all_features_missing"] = {"rejected": True, "error": str(exc)[:240]}
        else:
            negatives["all_features_missing"] = {"rejected": False, "risk": "valid-looking JSON from absent feature schema"}
    report = {"seconds": time.perf_counter()-started, "rows": len(df), "network_namespace": netns,
              "original_network_namespace": original_netns, "network_connect_errno": network_error,
              "predict_module": predict.__file__, "models_dir": predict.MODELS_DIR,
              "byte_identical_replay": True, "json_sha256": sha(first),
              "inference_policy": "RF n_jobs=1 in memory via deterministic_predict.py; weights unchanged", 
              "batch_shuffle_single_max_delta": max_delta, "negative_fixtures": negatives,
              "fixture_note": "Wholly synthetic numeric/categorical values; header names only from training CSV.",
              "scope": "same-machine installed environment, copied source and weights, network isolated; not clean-machine provisioning"}
    (destination / "replay_evidence.json").write_text(json.dumps(report, indent=2)+"\n")


def run(output):
    import numpy as np
    import pandas as pd

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    package = output / "package"
    package.mkdir()
    original = {**{str(p): sha(p) for p in (ROOT / "models").rglob("artifacts.joblib")},
                **{str(p): sha(p) for p in (ROOT / "pipelines").glob("*.py")}}
    shutil.copytree(ROOT / "pipelines", package / "pipelines", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for panel in PANEL_FILES:
        target = package / "models" / panel
        target.mkdir(parents=True)
        shutil.copy2(ROOT / "models" / panel / "artifacts.joblib", target / "artifacts.joblib")
    shutil.copy2(ROOT / "requirements.txt", package / "requirements.txt")
    import importlib.metadata
    versions = {d.metadata['Name']: d.version for d in importlib.metadata.distributions() if d.metadata['Name']}
    (package / "environment.json").write_text(json.dumps({'python': sys.version, 'installed_packages': dict(sorted(versions.items()))}, indent=2)+'\n')
    schema = {panel: [c for c in pd.read_csv(ROOT / 'data' / 'panels' / f'YARISMA_TRAIN_{suffix}.csv', nrows=0).columns if c not in ('Variant_ID', 'Label')] for panel, suffix in PANEL_FILES.items()}
    (package / "input_schema.json").write_text(json.dumps(schema, indent=2)+'\n')
    (package / "deterministic_predict.py").write_text('''"""Inference-only wrapper: deterministic RF accumulation and explicit feature-schema check."""
import argparse
import csv
import json
import sys
from pathlib import Path
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "pipelines"))
import predict

def preflight(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        columns = next(csv.reader(handle), [])
    if len(columns) != len(set(columns)):
        raise ValueError("Duplicate CSV header names; resolve input schema before inference")
    schema = json.loads((BASE / "input_schema.json").read_text())
    expected = set().union(*(set(value) for value in schema.values()))
    missing = sorted(expected - set(columns))
    if missing:
        raise ValueError(f"Missing {len(missing)} expected raw feature columns: {missing[:12]}; do not silently substitute absent headers with NaN")

def configure():
    original = predict.load_artifacts
    def load(panel):
        art = original(panel)
        own_blend = art.get("own", {}).get("blend") or {}
        transfer_blend = art.get("transfer", {}).get("art", {}).get("blend") or {}
        for blend in (own_blend, transfer_blend):
            if blend.get("rf_model") is not None:
                blend["rf_model"].set_params(n_jobs=1)
        return art
    predict.load_artifacts = load

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("infer", "submit"):
        raise SystemExit("Only infer/submit are allowed in this frozen inference package")
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--input", required=True)
    args, _ = pre.parse_known_args()
    preflight(args.input)
    configure()
    predict.main()
''')
    (package / "README.md").write_text("# Baseline offline replay package\n\n"
        "Research rehearsal copy; no training or final scoring data included.\n"
        "Use the existing Python 3.12 environment on this machine; dependencies are not bundled.\n"
        "environment.json records the installed versions; input_schema.json contains raw feature names only.\n"
        "The wrapper rejects missing/duplicate feature headers and uses serial RF inference without altering weights.\n\n"
        "`<python> deterministic_predict.py submit --input <external.csv> --output <external.json> "
        "--team-name <name> --team-id <id> --application-id <id> --prob-mode p1 --probabilities`\n\n"
        "Confirm probability fields/orientation and panel vocabulary with organizers before actual submission.\n")
    snapshot = tree_hashes(package)
    rng = np.random.default_rng(1729)
    frames = []
    for panel, suffix in PANEL_FILES.items():
        header = pd.read_csv(ROOT / "data" / "panels" / f"YARISMA_TRAIN_{suffix}.csv", nrows=0).columns
        n = COUNTS[panel]
        cols = [c for c in header if c not in ("Variant_ID", "Label")]
        data = {}
        for col in cols:
            if col == "AA_1": data[col] = rng.choice(["A", "C", "D", "E"], n)
            elif col == "AA_2": data[col] = rng.choice(["V", "H", "L", "G"], n)
            elif col.startswith("CAT_"): data[col] = np.full(n, "SYNTHETIC_UNKNOWN", dtype=object)
            elif col.startswith("AL_"): data[col] = np.power(10., rng.uniform(-12, -2, n))
            else: data[col] = rng.uniform(-1, 1, n)
        frame = pd.DataFrame(data)
        for col in cols:
            frame.loc[rng.random(n) < .25, col] = np.nan
        frame.loc[0, cols] = np.nan
        frame.insert(0, "panel", panel.upper())
        frame.insert(0, "id", [f"SYNTHETIC_{panel}_{i:05d}" for i in range(n)])
        frames.append(frame)
    fixture = output / "synthetic_input.csv"
    pd.concat(frames, ignore_index=True).sample(frac=1, random_state=1729).to_csv(fixture, index=False)
    original_netns = os.readlink("/proc/self/ns/net")
    command = ["unshare", "--user", "--map-root-user", "--net", sys.executable, "-I", "-B", str(Path(__file__).resolve()),
               "--child", "--package", str(package), "--fixture", str(fixture), "--output", str(output / "replay"),
               "--original-netns", original_netns]
    (output / "command.json").write_text(json.dumps(command, indent=2)+"\n")
    with (output / "replay.log").open("w") as log:
        subprocess.run(command, cwd=package, stdout=log, stderr=subprocess.STDOUT, check=True)
    assert snapshot == tree_hashes(package), "package contents changed during replay"
    assert all(sha(path) == value for path, value in original.items()), "production artifacts changed"
    (output / "integrity.json").write_text(json.dumps({"production_hashes": original,
        "package_hashes": snapshot, "fixture_sha256": sha(fixture), "no_production_changes": True}, indent=2)+"\n")
    print(f"Offline replay completed: {output / 'replay' / 'replay_evidence.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--package")
    parser.add_argument("--fixture")
    parser.add_argument("--original-netns")
    args = parser.parse_args()
    if args.child:
        child(args.package, args.fixture, args.output, args.original_netns)
    else:
        run(args.output)
