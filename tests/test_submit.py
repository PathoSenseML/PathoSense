"""Tests for the finals submission JSON (predict.py submit) — guide §3.2 hard rules.

Run: .venv/bin/python tests/test_submit.py (exit 1 on failure).
Pure assembly/validation; needs no trained models.
"""
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipelines"))

from predict import build_submission, validate_submission, write_submission, _panel_keys, \
    _infer_panel_from_name, _split_panel_assignment


def _ids(n):
    return [f"Varyant_{i + 1}" for i in range(n)]


def _obj(n=2):
    return build_submission("t", "TEAM_1", "B_1", _ids(n), ["CFTR"] * n,
                            [0.3, 0.9][:n], [0, 1][:n])


def test_matches_guide_example():
    """Guide example: class '0' with predicted_prob 0.87. Under the default (P(class 1)) that is
    P(1)=0.87 decided '0' by a threshold above 0.87 — our regime. The 'predicted' mode
    reproduces the same row from P(1)=0.13 instead."""
    r = build_submission("Takım", "TEAM_101", "BASVURU_101", ["Varyant_1"], ["CFTR"],
                         [0.87], [0], with_probabilities=False)["predictions"][0]
    assert r == {"id": "Varyant_1", "panel": "CFTR", "predicted_class": "0",
                 "predicted_prob": 0.87}, r
    r = build_submission("Takım", "TEAM_101", "BASVURU_101", ["Varyant_1"], ["CFTR"],
                         [0.13], [0], prob_mode="predicted", with_probabilities=False)["predictions"][0]
    assert r["predicted_prob"] == 0.87

def test_default_prob_is_p1_and_consistent_with_dict():
    """Default row = the §3.2 template exactly; predicted_prob is P(class 1) whatever the class."""
    for p, k in ((0.13, 0), (0.87, 0), (0.95, 1)):
        r = build_submission("t", "TEAM_1", "B_1", ["x"], ["PAH"], [p], [k])["predictions"][0]
        assert set(r) == {"id", "panel", "predicted_class", "predicted_prob"}, r
        assert r["predicted_prob"] == p, r
        r = build_submission("t", "TEAM_1", "B_1", ["x"], ["PAH"], [p], [k],
                             with_probabilities=True)["predictions"][0]
        assert r["predicted_prob"] == p == r["probabilities"]["1"], r

def test_default_filename_follows_guide():
    """TEAM_<TAKIM_ID>_FINAL.json: team_id 'TEAM_924489' must not become TEAM_TEAM_..."""
    import tempfile, pandas as pd
    from predict import submit
    base = os.path.join(os.path.dirname(__file__), "..", "data", "panels", "YARISMA_TRAIN_CFTR.csv")
    art = os.path.join(os.path.dirname(__file__), "..", "models", "cftr", "artifacts.joblib")
    if not (os.path.exists(base) and os.path.exists(art)):
        print("  test_default_filename_follows_guide SKIP (no data/models)"); return
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as d:
        pd.read_csv(base).head(2).drop(columns=["Label"]).to_csv(os.path.join(d, "c.csv"), index=False)
        os.chdir(d)
        try:
            out = submit(["cftr=c.csv"], "PathoSense", "TEAM_924489", "BASVURU_5077661")
        finally:
            os.chdir(cwd)
        assert os.path.basename(out) == "TEAM_924489_FINAL.json", out


def test_class_one_keeps_prob():
    r = build_submission("t", "TEAM_1", "B_1", ["x"], ["PAH"], [0.87], [1])["predictions"][0]
    assert r["predicted_prob"] == 0.87 and r["predicted_class"] == "1"


def test_prob_mode_predicted_flips_for_class_zero():
    r = build_submission("t", "TEAM_1", "B_1", ["x"], ["PAH"], [0.13], [0],
                         prob_mode="predicted", with_probabilities=True)["predictions"][0]
    assert r["predicted_prob"] == 0.87 and r["probabilities"]["1"] == 0.13


def test_probabilities_dict_sums_to_one():
    r = build_submission("t", "TEAM_1", "B_1", ["x"], ["PAH"], [0.3], [0],
                         with_probabilities=True)["predictions"][0]
    assert r["probabilities"] == {"0": 0.7, "1": 0.3}
    r = build_submission("t", "TEAM_1", "B_1", ["x"], ["PAH"], [0.3], [0])["predictions"][0]
    assert "probabilities" not in r   # default = template exactly


def test_header_fields():
    obj = build_submission("Takım Adı", "TEAM_101", "BASVURU_101", [], [], [], [])
    assert obj == {"team_name": "Takım Adı", "team_id": "TEAM_101",
                   "application_id": "BASVURU_101",
                   "competition_level": "UNIVERSITE_VE_UZERI", "predictions": []}


def test_valid_passes():
    validate_submission(_obj(), _ids(2))  # no raise


def _rejects(mutate, match):
    obj, ids = _obj(), _ids(2)
    mutate(obj)
    try:
        validate_submission(obj, ids)
    except ValueError as e:
        assert match in str(e), f"wrong reason: {e}"
        return
    raise AssertionError(f"expected ValueError containing {match!r}")


def test_rejections():
    _rejects(lambda o: o["predictions"].pop(), "exactly once")
    _rejects(lambda o: o["predictions"][1].__setitem__("id", o["predictions"][0]["id"]), "duplicate")
    _rejects(lambda o: o["predictions"][0].__setitem__("predicted_prob", 1.2), "out of")
    _rejects(lambda o: o["predictions"][0].__setitem__("predicted_prob", float("nan")), "out of")
    _rejects(lambda o: o["predictions"][0].__setitem__("predicted_class", "P"), "'0' or '1'")
    _rejects(lambda o: o.__setitem__("competition_level", "LISE"), "UNIVERSITE")
    _rejects(lambda o: o["predictions"][0].__setitem__("probabilities", {"0": 0.5, "1": 0.4}),
             "summing")
    obj, ids = _obj(), _ids(2)
    try:
        validate_submission(obj, ["Varyant_1", "Varyant_2X"])
    except ValueError as e:
        assert "missing" in str(e), e
    else:
        raise AssertionError("expected missing-id rejection")


def test_write_roundtrip_utf8(tmp=""):
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        obj = build_submission("Takım_Ø", "TEAM_1", "B_1", _ids(2), ["CFTR"] * 2,
                               [0.2, 0.8], [0, 1])
        p = os.path.join(d, "TEAM_1_FINAL.json")
        write_submission(obj, p)
        back = json.loads(open(p, encoding="utf-8").read())
        assert back["team_name"] == "Takım_Ø"
        assert back["predictions"][0]["predicted_prob"] == 0.2   # P(class 1) as-is, class 0 decided
        # decimal separator is a dot, not a comma (guide §3)
        assert "." in open(p, encoding="utf-8").read().split('"predicted_prob": ')[1].split("}")[0]
        try:
            write_submission(build_submission("t", "TEAM_1", "B_1", _ids(1), ["CFTR"],
                                              [float("nan")], [1]), p)
        except ValueError:
            pass
        else:
            raise AssertionError("NaN probability must be rejected at write (allow_nan=False)")


def test_panel_column_wins_over_fallback():
    import pandas as pd
    raw = pd.DataFrame({"panel": ["CFTR", "pah"]})
    assert _panel_keys(raw, "general", "f").tolist() == ["cftr", "pah"]

def test_organizer_panel_vocabulary_accepted():
    """Şartname names: MASTER, Kanser, CFTR, PAH — must route, any case."""
    from predict import _panel_key, PANEL_LABELS
    assert [_panel_key(v) for v in ("MASTER", "Kanser", "KANSER", "Genel", "cftr", "PAH")] \
        == ["general", "cancer", "cancer", "general", "cftr", "pah"]
    assert PANEL_LABELS == {"general": "MASTER", "cancer": "KANSER", "pah": "PAH", "cftr": "CFTR"}
    try:
        _panel_key("HEART")
    except ValueError as e:
        assert "HEART" in str(e)
    else:
        raise AssertionError("unknown panel must be rejected")

def test_fallback_covers_panel_less_file():
    import pandas as pd
    raw = pd.DataFrame({"id": ["a"]})
    assert _panel_keys(raw, "general", "f").tolist() == ["general"]
    try:
        _panel_keys(raw, None, "f.csv")
    except ValueError as e:
        assert "f.csv" in str(e) and "--panel" in str(e)
    else:
        raise AssertionError("expected missing-panel rejection naming the file")

def test_panel_guessed_from_filename():
    assert _infer_panel_from_name("YARISMA_TEST_KANSER.csv") == "cancer"
    assert _infer_panel_from_name("YARISMA_TEST_MASTER.csv") == "general"
    assert _infer_panel_from_name("/x/YARISMA_TEST_PAH.csv") == "pah"
    assert _infer_panel_from_name("cftr_test.csv") == "cftr"
    assert _infer_panel_from_name("mixed.csv") is None

def test_panel_assignment_parsing():
    assert _split_panel_assignment("general=g.csv") == ("general", "g.csv")
    assert _split_panel_assignment("CANCER=/x/k.csv") == ("cancer", "/x/k.csv")
    assert _split_panel_assignment("plain.csv") == (None, "plain.csv")
    assert _split_panel_assignment("a=b=c.csv") == (None, "a=b=c.csv")  # '=' in path

def test_multi_file_submit_end_to_end():
    """Separate per-panel CSVs (no panel column) merge into one JSON. Needs models."""
    import pandas as pd
    from predict import submit
    base = os.path.join(os.path.dirname(__file__), "..", "data", "panels")
    art = os.path.join(os.path.dirname(__file__), "..", "models", "general",
                         "artifacts.joblib")
    src = os.path.join(base, "YARISMA_TRAIN_MASTER.csv")
    if not (os.path.exists(src) and os.path.exists(art)):
        print("  test_multi_file_submit_end_to_end SKIP (no data/models)")
        return
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        df = pd.read_csv(src)
        f1 = os.path.join(d, "a.csv")
        f2 = os.path.join(d, "b.csv")
        df.head(3).drop(columns=["Label"]).to_csv(f1, index=False)
        df.iloc[3:5].drop(columns=["Label"]).to_csv(f2, index=False)
        out = os.path.join(d, "TEAM_X_FINAL.json")
        submit([f"general={f1}", f"general={f2}"], "t", "TEAM_X", "B_X", output=out)
        obj = json.loads(open(out, encoding="utf-8").read())
        assert len(obj["predictions"]) == 5, obj
        got = sorted(r["id"] for r in obj["predictions"])
        assert got == sorted(df["Variant_ID"].head(5).tolist()), got
        # no panel column in the files -> organizer label, never our internal key
        assert {r["panel"] for r in obj["predictions"]} == {"MASTER"}, obj["predictions"][0]
        try:
            submit([f"general={f1}", f"cancer={f2}"], "t", "TEAM_X", "B_X",
                   output=out, panel=["general", "general"])
        except ValueError as e:
            assert "conflicting panels" in str(e), e
        else:
            raise AssertionError("expected conflicting-panel rejection")

def main():
    checks = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    results = []
    for name, fn in checks:
        try:
            fn()
            results.append(True)
        except Exception:
            print(f"  {name} FAIL", flush=True)
            traceback.print_exc()
            results.append(False)
    print("ALL PASS" if all(results) else "FAILURES DETECTED")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
