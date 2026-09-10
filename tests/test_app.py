"""Presentation regressions; requires local models/data and app/requirements.txt.

Run: .venv/bin/python tests/test_app.py
Callable st.Page routing uses AppTest._page_hash = calc_hash(url_path), with no
production test hooks. The language is fixed when the interface is launched
(PATHOSENSE_LANG), so a session has one language: tests pick it per AppTest.
The variant stepper (prev/next) is browser-tested: its selectbox is seeded from
session state, which AppTest cannot serialize across a button click.
"""
import ast
import html
import io
import json
import os
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
from streamlit.testing.v1 import AppTest
from streamlit.util import calc_hash

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipelines"))
from predict import load_artifacts, score_frame
from explain import explain_frame, top_contributions
from utils import PANELS, TEST_PRIORS

APP = ROOT / "app/app.py"


def on(at, page):
    at._page_hash = calc_hash(page)
    at._run()
    assert not at.exception, at.exception
    return at


def run(page="overview", query_params=None, **state):
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.query_params = dict(query_params or {})
    for key, value in state.items():
        at.session_state[key] = value
    return on(at, page)


def upload(csv, **state):
    """A loaded file is the only source control: scoring it needs no other state. The mirror carries the panel
    the header was showing when the file was taken, which here is the panel the file is scored by — the two
    disagreeing is its own state, tested in test_the_file_remembers_which_panel_it_was_uploaded_for."""
    return run(page="predictions", upload_keep=("test.csv", csv, state.get("panel_keep", "general")), **state)


def metrics(at):
    return {m.label: m.value for m in at.metric}

def cards(at):
    """The cards of the row over the histogram — the batch's count (count_card) and the grids (quad_card).
    The count is drawn rather than st.metric so that it can carry the reference it is read against and the
    fraction as a bar; st.metric holds neither. Keyed by the subject of the label: (value, the qualifier
    after the separator, the fact on the marker), and a grid has no single value line."""
    out = {}
    for block in re.findall(r"<div class='card'>(.*?)</div>", markup(at), re.S):
        head, _, rest = block.partition("<b class='num'>")
        sub = re.search(r"<span class='sub'>(.*?)</span>", head)
        tip = re.search(r"data-tip='([^']*)'", head)
        plain = lambda s: html.unescape(re.sub(r"<[^>]+>", "", s)).strip(" ·")
        # The label ends where the card's own content starts: its qualifier, its marker, or the grid. A
        # card whose value is four counts has no value line, so without this the figures joined the key.
        out[plain(re.split(r"<span class='(?:sub|quad|hint)'", head)[0])] = (plain(rest.split("</b>")[0]),
                                                           plain(sub[1]) if sub else "",
                                                           html.unescape(tip[1]) if tip else "")
    return out


def verdicts(at):
    """The cells of the row's grids (quad_card): the four review flags, and the four trust verdicts where
    the file carries labels. Keyed by the name the cell carries for a reader who cannot see the mark —
    which states its own reference — to (count, the mark drawn, its tone, the tint); a flag cell's tone is
    `fl`, which is what tells the two cards' cells apart."""
    return {html.unescape(name): (count, glyph, tone.replace("tm tm-", ""), cls.strip())
            for cls, name, tone, glyph, count in re.findall(
                r"<span class='qc([^']*)' role='img' aria-label='([^']*)'[^>]*>"
                r"<span class='(tm tm-\w+|fl)'><span class='mk mk-(\w+)'></span></span><b>(\d+)<span", markup(at))}


def markup(at):
    """Markdown plus the st.html blocks (headings, notes, tables) the app renders, as emitted."""
    return "\n".join(e.value for e in list(at.markdown) + list(at.get("html")))


def text(at):
    """The same, unescaped: a claim relocated into a heading tooltip is asserted in its own words."""
    return html.unescape(markup(at))

def plain(at):
    """What the page reads as, tags dropped: a band heading is a subject and, where one is earned, a chip in a
    span of its own, so the line is asserted as one string here rather than as the markup that sets it. A tag
    becomes a space and runs of spaces collapse — without that, a chip would read as one word with the title
    it hangs on. Newlines are kept: they are what separates one block from the next."""
    return re.sub(r"[ \t]+", " ", re.sub(r"<[^>]+>", " ", text(at)))


def chips(at):
    """The app's one badge treatment (.chp): a claim the page must carry without a hover, drawn as a marker on
    a title rather than as another ` · clause` of it. Every site uses this class, so this is the whole set."""
    return [html.unescape(m) for m in re.findall(r"<span class='chp'>(.*?)</span>", markup(at))]


def hints(at):
    """Only the heading tooltips: the app's route for a fact that qualifies numbers without narrating them."""
    return "\n".join(html.unescape(m) for m in re.findall(r"class='hint' role='note' tabindex='0' data-tip='([^']*)'", markup(at)))


def chart_text(at):
    """Every string cell of every Altair dataset on the page: axis labels live in the data, not the spec."""
    return "\n".join(str(v) for e in at.get("vega_lite_chart") for ds in e.proto.datasets
                     for col in pa.ipc.open_stream(ds.data.data).read_all().columns for v in col.to_pylist())

def chart_column(at, i, name):
    """One named column of one chart's dataset: which field an axis draws, in the values it draws."""
    table = pa.ipc.open_stream(at.get("vega_lite_chart")[i].proto.datasets[0].data.data).read_all()
    return table.column(name).to_pylist()


def table_ids(at):
    """The results table in the order it is drawn: every row carries its variant id for the click bridge."""
    return re.findall(r'<tr[^>]*data-vid="(VAR_\d+)"', text(at))

def sample(n=25):
    return pd.read_csv(PANELS["general"], nrows=n)


def csv(df):
    return df.to_csv(index=False).encode()


def test_overview_is_canonical_not_batch():
    at = run(upload_keep=("bad.csv", b"bad", "general"))
    assert not at.error and len(at.metric) == 3   # recall, precision, F1: the fixed term, the term the set moves, and their mean
    value = text(at)
    for number in ("0.537 ± 0.009", "0.725 ± 0.007", "0.669 ± 0.007", "0.851 ± 0.009"):
        assert number in value
    assert "Eğitim n · prevalans" in value and "2815 · %73.7" in value      # the fitted row count beside the prior
    assert "Duyarlılık" in value and "0.735" in value                     # pooled recall, the one confusion rate both derivations share
    # Recall and precision stand either side of their harmonic mean, so the mandated metric is arithmetic the
    # reader can do between two cells: 2·0.735·0.423/(0.735+0.423) = 0.537, and likewise on the other three.
    assert "Kesinlik" in value and all(p in value for p in ("0.423", "0.649", "0.559", "0.862"))
    for tpr, prec, f1 in ((0.735, 0.423, 0.537), (0.822, 0.649, 0.725), (0.833, 0.559, 0.669), (0.840, 0.862, 0.851)):
        assert round(2 * tpr * prec / (tpr + prec), 3) == f1, (tpr, prec, f1)
    # The best single input, in the AUC cell it is read against: a parameter-free baseline measured on the same
    # rows is evidence, so it is set at the size of the number beside it and only its colour stays muted.
    assert "0.864 <span class='base'>· tek girdi phyloP 0.738</span>" in value    # named, not an id
    assert ".base {color:var(--muted)}" in value and "class='small'>· tek girdi" not in value
    # The results note is gone: the cross-validation claim is a tag on the heading, the rest is its tooltip.
    # Which metric it is follows the same rule: the table below draws an F1 column, so naming the metric a
    # third time on the line restated a heading over the column it names. It qualifies, so it hangs off the marker.
    # The tooltip is two sentences: the metric and its protocol, and the one thing this table is not. The fold
    # depth, the pooled recall and the single-input baseline were method narration and are gone from it.
    assert "Çapraz doğrulama sonuçları" in plain(at) and "Çapraz doğrulama (3 tohum)" not in value
    assert "test prevalansında" in chips(at)   # claim 6, as the chip on that heading and not as a clause of it
    assert "yarışma metriği ikili F1" not in plain(at) and "İkili F1, pozitif sınıf patojenik" in hints(at)
    assert "Kat sayısı" not in hints(at) and "havuzlanmış" not in hints(at)
    assert "her katta yeniden belirlenir" in hints(at)   # ... with the protocol that makes them defensible
    assert "Klinik doğrulama değildir" in hints(at)
    # ... and on the curve, a heading naming the one F1 the project is scored on and what the rest of it is
    # The subject and the one claim that may not go back into a hover. The F1 the project is scored on was a
    # third clause here; the table at the top of this same screen prints it per panel, with this panel marked.
    assert "Aynı modelin farklı kümelerde beklenen F1 değeri" in plain(at) and "ekstrapolasyon" in chips(at)
    assert "yarışma test kümesinde F1" not in plain(at) and "0.537 ± 0.009" in value
    assert "ekstrapolasyondur, ölçüm değildir" in at.metric[2].help and "ekstrapolasyondur" not in value
    # The three counts are a picture and not a sentence: each is a segment of the thousand-variant bars,
    # named for a reader who cannot see them, and the two error cells are the two the model got wrong.
    assert all(f"aria-label='{n} patojenik bulundu'" in value or f">{n} " in value for n in (105, 38, 143))
    assert value.count("class='pk-track'") == 2 and "38 patojenik kaçırıldı" in value and "143 benign yanlış işaretlendi" in value
    assert list(metrics(at)) == ["Duyarlılık", "Kesinlik", "F1"]
    assert not re.search(r"class='tablewrap[^']*' tabindex", value)  # non-scrollable reference tables are not tab stops (the ⓘ marks are)
    assert len(at.get("page_link")) == 0
    labels = chart_text(at)  # without a batch the AF-vs-flag heuristic cannot run: AL_ columns are named as allele frequencies
    assert "Dönüştürülmüş AF · popülasyon sütunu" in labels and "bayrak sütunu" not in labels
    # The top-10 axis carries the column id alone, unlike the contribution chart's "id · name": the bar
    # colour and the family table beside it already name the family, and on cancer the names took most of
    # the chart's width from the bars. The name stays in the data, as the tooltip's first row.
    spec = json.loads(at.get("vega_lite_chart")[1].proto.spec)
    assert spec["encoding"]["y"]["field"] == "feature"
    assert [tip["field"] for tip in spec["encoding"]["tooltip"]] == ["name", "family", "share"]
    ranked = chart_column(at, 1, "feature")
    assert len(ranked) == 10 and not any(" · " in f for f in ranked), ranked
    assert "Dönüştürülmüş AF · popülasyon sütunu 26" in chart_column(at, 1, "name"), "the name is a hover away"


def test_overview_charts_keep_the_reclaimed_height():
    """The prevalence curve and the ranked top-10 were squeezed below their design heights only to keep
    Overview inside the fold; the panel selector moving into the header gave the room back. Both are sized
    from the named constants, and cancer — the tallest cell — still ends short of 768 with room to spare."""
    at = run(viewport=(1366., 768.), panel_keep="cancer")
    prevalence, ranked = (json.loads(c.proto.spec) for c in at.get("vega_lite_chart"))
    assert prevalence["height"] == int(220 * 13 / 16), prevalence["height"]      # H(BAND) at the fixed root
    assert ranked["height"] == 200, ranked["height"]                             # MIN_BAND floor × 10 rows
    assert "labelLimit" not in ranked["encoding"]["y"]["axis"]   # bare ids need no limit of their own


def test_every_panel_and_language_on_all_pages():
    for panel in PANELS:
        for lang in ("tr", "en"):
            at = run(panel_keep=panel, lang=lang)
            for page in ("predictions", "explain", "overview"):
                on(at, page)
                assert not at.error, (panel, lang, page, at.error)
                assert at.session_state["panel_keep"] == panel
            assert ("extrapolation, not a measurement" if lang == "en" else "ekstrapolasyondur") in \
                "\n".join(m.help or "" for m in at.metric)


def test_flow_and_importance_note_are_derived_from_each_artifact():
    for panel, expected, absent in (
            ("general", "kazanç önemi: panelin kendi LightGBM'i.", "aktarım"),
            ("cancer", "%50 panelin kendi LightGBM'i + %50 aktarım LightGBM'i.", "XGBoost"),
            ("pah", "Genel panelinkiyle aynıdır", "kendi"),
            ("cftr", "aktarım LightGBM'i. Genel panelde eğitildiği için", "Gösterilmeyen")):   # "only" says it; the hidden members are not listed
        art = load_artifacts(panel)
        for lang in ("tr", "en"):
            at = run(panel_keep=panel, lang=lang)
            value = markup(at)
            flow = next(v for v in value.split("\n") if v.startswith("<div class='flow'"))
            assert f"--threshold:{art['threshold'] * 100:.8f}%" in flow
            assert ("data-branch='own'" in flow) == bool(art["own"])
            assert ("data-branch='transfer'" in flow) == bool(art["transfer"])
            if art["own"]:
                own = art["own"]
                assert f"LightGBM ×{len(own['models'])}" in flow
                assert str(len(own['models'][0].booster_.feature_name())) in flow
                count = 1 + sum(own.get("blend", {}).get(k) is not None for k in
                                ("rf_model", "cat_model", "xgb_model")) if own.get("blend") else 1
                assert flow.count("→ Platt</small>") == count
                assert f"data-weight='{1/count:.8f}'" in flow
            if art["transfer"]:
                ta = art["transfer"]["art"]
                for n in (ta["n_train"], ta["n_excluded"], len(ta["feats"])):
                    assert str(n) in flow
                weights = ta["blend_weights"] if ta["blend"] else [1.]
                for w in weights:
                    assert f"data-weight='{w:.8f}'" in flow
                assert "flow-excluded" in flow and "<s>" in flow
                assert flow.count("class='flow-platt'") == 1  # AFTER the transfer blend, never per internal member
                assert (("AF → panel içi rank" in flow or "AF → in-panel rank" in flow)
                        == bool(art["transfer"].get("target_grids")))
                if not ta["ppf"]["pop_grids"]:
                    assert "AF → rank" not in flow
            if lang == "tr":   # the members note is the top-10 heading's tooltip now, never a paragraph under it
                assert "<div class='small'>Yalnızca LightGBM" not in value
                note = next(h for h in hints(at).split("\n") if h.startswith("Yalnızca LightGBM"))
                assert expected in note and absent not in note, (panel, note)


def test_flow_geometry_changes_with_artifact_not_panel_name():
    # Exercise the renderer directly without importing the Streamlit entrypoint.
    tree = ast.parse(APP.read_text())
    strings = next(n for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "S" for t in n.targets))
    labels = ast.literal_eval(strings.value)["en"]
    tick_flip = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "TICK_FLIP" for t in n.targets))
    env = {"E": html.escape, "TEST_PRIORS": TEST_PRIORS, "TICK_FLIP": tick_flip,
           "t": lambda key, **kw: labels[key].format(**kw) if kw else labels[key],
           "percent": lambda w, d=1: f"{w:.{d}%}",
           "training_rows": lambda panel: {"n": 2815} if panel == "general" else None}
    for name in ("others", "pipeline_html"):
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(APP), "exec"), env)
    original = load_artifacts("cancer")
    art = dict(original, threshold=.63, transfer_w=.3)
    ta = dict(original["transfer"]["art"], blend_weights=(.2, .3, .5), n_train=1234, n_excluded=99)
    art["transfer"] = dict(original["transfer"], art=ta)
    flow = env["pipeline_html"]("cancer", art)
    assert "--threshold:63.00000000%" in flow and "1234 training rows" in flow
    assert "Own 70%" in flow and "Transfer 30%" in flow
    assert all(f"data-weight='{w:.8f}'" in flow for w in (.2, .3, .5, .7))
    # n_excluded counts IDs, so the lane draws the rows they removed beside it and the subtraction closes:
    # the source panels' fitted rows minus the scorer's training rows.
    assert "99 IDs excluded · 1581 rows matched" in flow and 2815 - 1581 == 1234
    # A threshold near the top of the scale draws its tick label inside the scale, not over the gutter.
    assert "class='flow-tick'" in flow
    assert "class='flow-tick flip'" in env["pipeline_html"]("cftr", dict(art, threshold=.90))


def test_the_screened_set_is_a_continuum_with_the_known_prevalences_marked():
    """One control, continuous, with the three known prevalences marked on its track and the contest's set as
    the default. The continuum is safe here only because of what sits beside it: the recall card holds the
    same number everywhere on the track, and the errors the model makes get visibly worse exactly where the
    F1 improves — at the curated prevalence it misses five times the pathogenic variants it misses on the
    contest's own test set. Those two hold at every stop, not only at the marked ones, which is what a reader
    dragging between the marks has to be able to see."""
    at = run()
    screen, test, curated = 0.01, TEST_PRIORS["general"], 0.7367673179396093
    assert at.session_state["prev_keep"] == test                                # the contest's set is the default
    labels = list(metrics(at))
    # Three bare labels and one chip: recall is the number that does not move, and the other two are of the
    # set the control above them names, which they no longer say back.
    assert labels == ["Duyarlılık", "Kesinlik", "F1"]
    assert '.st-key-metric_chip [data-testid="stMetricLabel"] p::after {content:"sabit"}' in text(at)
    # At rest the cards read the panel's own prior, so the table's precision and the card's are one number.
    assert "0.423" in text(at) and "1000 varyantta 143 patojenik" in text(at)
    # One control: a track whose stops are prevalences, with the three known ones on it exactly, and the
    # names beside them — never a bare percentage, and never a slider plus a second control for the same axis.
    track = next(w for w in at.select_slider if w.key == "prev_w")
    assert not at.slider and not any(w.key == "prev_w" for w in at.segmented_control)
    assert track.value == test and len(track.options) == 179
    assert (track.options[0], track.options[-1]) == ("%1.0", "%90.0")   # the domain covers every mark and stops there
    assert all(mark in track.options for mark in ("%1.0", "%14.3", "%73.7"))
    assert all(name in text(at) for name in ("Geniş tarama", "Yarışma test kümesi", "Küratörlü küme"))
    for prior, recall, f1, missed in ((screen, "0.735", "0.080", 3), (curated, "0.735", "0.819", 195),
                                      (0.45, "0.735", "0.758", 119)):   # the third is no set at all: between two marks
        next(w for w in at.select_slider if w.key == "prev_w").set_value(prior).run()
        assert at.session_state["prev_keep"] == prior
        value, shown = text(at), list(metrics(at).values())
        assert shown[0] == recall and shown[2] == f1 and f"{missed} kaçırıldı" in value
        # The claim the curve may not be read without stays on the chart's heading, wherever the control is left.
        assert "ekstrapolasyon" in chips(at) and "Aynı modelin farklı kümelerde beklenen F1 değeri" in plain(at)
    on(at, "overview")
    assert at.session_state["prev_keep"] == 0.45                                # a chosen position survives a rerun
    at.segmented_control[0].set_value("cancer").run()
    # A prevalence names a different set on every panel, so the control returns to this panel's contest set
    # rather than carrying a number that named one set and now names none.
    assert at.session_state["prev_keep"] == TEST_PRIORS["cancer"] and "1000 varyantta 167 patojenik" in text(at)
    # A cohort key kept by an older session, or handed in by a deep link, is a stale value and not a crash.
    assert run(prev_keep="curated").session_state["prev_keep"] == TEST_PRIORS["general"]


def test_the_data_controls_are_a_header_button_and_a_table_header_row():
    """No page toolbar at all. The upload control is one button in the header band beside the panel segments,
    with everything it does behind a dialog; the filter is the results table's own header row, in the column
    the table is in, and the two exports are that row's second control and the table's foot line. The source
    toggle and the sample-size slider are gone — a file means the file, no file means the demo sample."""
    at = run(page="predictions")
    assert not (at.sidebar.selectbox or at.sidebar.button or at.sidebar.multiselect or at.sidebar.slider)
    assert not at.selectbox and not at.slider   # no panel dropdown, no source control, no sample size
    assert "<h2" not in text(at) and "class=\'pt\'" not in text(at)
    assert not at.get("file_uploader")   # the dropzone is in the dialog, and the dialog is shut
    assert next(b for b in at.button if b.key == "upload_btn").label == "Dosya seç"
    assert len(at.get("multiselect")) == 1 and len(at.get("download_button")) == 1   # one filter, one file out
    options = at.multiselect[0].options   # the counts moved into the option labels, and stay whole-batch
    assert options[0].startswith("patojenik · ") and "sınırda · " in " ".join(options)   # one case for the whole list
    value = text(at)
    assert ".st-key-table_head > *" in value and ".st-key-controls" not in value   # the toolbar's own rules are gone
    assert ".st-key-page_actions" not in value
    at = run(page="overview")   # Overview reads no batch: no filter, no export, and nothing to drill into
    assert not (at.multiselect or at.get("file_uploader") or at.selectbox or at.get("download_button"))
    assert not at.get("page_link")


def test_each_page_link_rides_the_header_row_of_a_control_it_belongs_to():
    """With the toolbar gone the two links keep their slots but pay no height for them: Tahminler carries
    "Varyant incelemesi" on the results table's header row, at the right end of the filter that scopes the
    same rows, and the review page carries "← Tahminler" on the variant picker's row, ahead of the control
    that moves within the batch. The way in is the page's one filled control — the drill-down is the
    explainability story — and stays a link; the way back is quiet. One internal link per page: an external
    link, or a plain anchor, would load a page — and a page load is a new session."""
    rule = ':is(.st-key-table_actions,.st-key-review_head) [data-testid="stPageLink"] a {min-height:2.5rem'
    primary = '.st-key-table_actions [data-testid="stPageLink"] a {padding:0 0.75rem; border-radius:var(--r); background:var(--brand); color:#fff}'
    for page, label in (("predictions", "Varyant incelemesi"), ("explain", "Tahminler")):
        at = run(page=page)
        links = at.get("page_link")
        assert len(links) == 1, page
        assert links[0].proto.label == label and links[0].proto.page_script_hash and not links[0].proto.external, page
        css = text(at)
        assert rule in css and primary in css, page   # a full-height target on a row a 2.5rem control already sets; one of them filled
        # The export is not on this row: it is the foot line's, and it is outlined there — a brand line over
        # a tint, brand ink, never a second block of fill. The link is the loudest control and stays it.
        assert '.st-key-table_export button {min-height:1.75rem; padding:0 0.5rem;' in css, page
        assert 'border:1px solid var(--brand); border-radius:var(--r-sm); background:color-mix(in srgb, var(--brand) 7%, transparent);' in css, page
    css = text(run(page="predictions"))   # the filter takes the width the primary leaves: one bar over the table, no hole between them
    assert '.st-key-table_head > [data-testid="stLayoutWrapper"]:has(> .st-key-filters) {flex:1 1 auto !important; min-width:17rem}' in css
    assert ".st-key-filters {width:17rem" not in css


def test_one_file_leaves_the_table_and_it_leads_with_the_confirmed_pair():
    """One file leaves this table, and it claims no file spec — the organizers have published none. It leads
    with the one shape they have confirmed, a row per variant and the call as 0/1, and carries what the app
    read the rows by after it: the probability, the four flags as 0/1 and the file's own label where it has
    one. It keeps the table's foot line, laid over the line the table draws rather than rendered into it and
    pulled up by exactly the line's height plus the gaps, so the column it shares with the table is no
    taller — but it is drawn as a control there and not as a caption: a brand line, a brand tint and brand
    ink, one step under the filled drill-down above. The table caps itself against that column (100cqh) so
    its foot line keeps following its last row. The file is named for the panel that scored it: four of them
    exist on the day, one per panel, and the score is per panel."""
    at = run(page="predictions", lang="en")
    assert len(at.download_button) == 1
    export = at.download_button[0]
    assert export.label == "Predictions (0/1)"
    assert export.proto.help == "Variant_ID · prediction 0/1 · probability · flag columns · label"
    assert export.proto.type == "secondary" and export.proto.icon == ":material/download:"   # not the primary: the drill-down is
    assert not any(word in (export.label + export.proto.help).lower()
                   for word in ("format", "submission", "contest", "official"))   # no claim the Q&A does not support
    source = APP.read_text()
    assert 'file_name=f"pathosense_{panel}_predictions.csv"' in source and "_review.csv" not in source
    css = text(at)
    assert ".st-key-table_body {flex:1; min-height:14rem; container-type:size}" in css
    assert "max-height:calc(round(down, 100cqh - 2*var(--lh-sm) - 0.375rem" in css   # two foot lines: a long filter name takes the count onto its own
    assert ".st-key-table_export {margin-top:calc(-1 * (var(--lh-sm) + 0.25rem + 0.375rem)); height:var(--lh-sm); align-self:flex-end" in css
    assert ".key .count {margin-left:auto; padding-right:calc(0.5rem + var(--export-w))}" in css   # the count leaves it the room
    tr = run(page="predictions")
    assert [b.label for b in tr.download_button] == ["Tahminler (0/1)"]   # both languages, in lockstep
    # A file with no Label column has no label column to offer, and the help says so.
    blind = upload(csv(sample().drop(columns="Label")), lang="en")
    assert blind.download_button[0].proto.help == "Variant_ID · prediction 0/1 · probability · flag columns"
    at = upload(csv(sample().assign(Label=0)), filter_keep=("benign", "wrong"), lang="en")   # nothing listed
    # The table is still drawn, with one muted line where the rows would be: the header, the border, the
    # foot line and the export keep their places, and the flex-fill chain that binds the page to the fold
    # keeps matching. A message in place of the table switched all of it off.
    assert not at.info and "class='tablewrap tall'" in text(at) and len(at.download_button) == 1
    assert "<tr class='none'><td colspan='5'>No variants match this filter.</td></tr>" in text(at)


def test_the_upload_dialog_declares_the_panel_and_then_takes_the_file():
    """The header button opens the app's own dialog: which panel the file is for, and then the file — that,
    and once one is loaded its name and its Remove. The panel comes first because the bytes are declared for
    whatever is current when they are taken. It is not a copy of the header band's control, which is what
    put two identical segmented controls on one screen: it is that control, drawn here while the dialog is
    open and nowhere else, under one key writing the one mirror. In the dialog it is named and carries the
    full panel names; in the band it is collapsed and short. The rule that suppresses Streamlit's English
    "page not found" modal is keyed off this dialog's own container, so the framework's stays hidden while
    this one draws."""
    at = run(page="predictions", upload_open=True)
    assert '[data-testid="stDialog"]:not(:has(.st-key-upload_dialog)) {display:none}' in text(at)
    # No help on the uploader: its label is collapsed, which makes the framework draw the marker as a 0×0
    # button — a tab stop with nothing to point at. The CSV's requirements are the rejected-upload screen's,
    # which is where a file that does not meet them lands (test_a_rejected_upload_still_states_the_csv_requirements).
    assert not at.get("file_uploader")[0].proto.help
    assert [s.key for s in at.segmented_control] == ["panel_w"]   # one panel control on the screen, and it is in here
    panel = at.segmented_control[0]
    assert panel.label == "Gen paneli" and panel.proto.label_visibility.value == 0   # named where the file is declared
    assert panel.options == ["Genel (MASTER)", "Kalıtsal kanser", "PAH (fenilketonüri)", "CFTR (kistik fibrozis)"]
    next(s for s in at.segmented_control if s.key == "panel_w").set_value("cancer").run()
    assert at.session_state["panel_keep"] == "cancer"
    assert chips(at)[:2] == ["örneklem içi", "%50 ağırlık"]   # ... and the batch is scored against that panel's model
                                                             # (the share is cancer's alone; the heading no longer names the panel)
    assert at.session_state["upload_open"]            # ... with the dialog still open for the file itself
    at = run(page="predictions", upload_open=True, upload_keep=("juri_batch.csv", csv(sample()), "general"))
    assert not at.get("file_uploader") and "juri_batch.csv" in text(at)   # the loaded file's name, and its Remove
    next(b for b in at.button if b.key == "remove_upload").click().run()
    assert "upload_keep" not in at.session_state
    assert "Eğitim kümesinden örneklem" in plain(at)   # removing it returns to the demo sample


def test_the_variant_page_carries_no_upload_control():
    """The header control is global, so it is page-conditional: the variant page inspects one row of a batch
    it does not choose. It still reads whatever batch is loaded."""
    at = run(page="explain", upload_keep=("juri_batch.csv", csv(sample()), "general"))
    assert not at.get("file_uploader")
    assert not [b for b in at.button if b.key in ("upload_btn", "upload_file", "remove_upload")]
    assert len([s for s in at.segmented_control if s.key == "panel_w"]) == 1   # the panel control stays
    assert "juri_batch.csv" not in text(at) and at.session_state["upload_keep"][0] == "juri_batch.csv"
    assert len(next(w for w in at.selectbox if w.key == "pick_w").options) == 25   # ... and the page reads its rows


def test_the_panel_is_a_header_control_not_a_page_row():
    """The gene panel is global state — it changes all three pages — so it is one segmented control drawn
    into the header band beside the nav, on every page, and no page spends a row on it. Short labels keep
    it inside that band; a page that names the panel in its own right names it in full. Predictions does not:
    its heading describes the file, and the panel scoring it is the highlighted segment on the same screen —
    naming it there was the heading restating a control the reader is looking at."""
    for page, full in (("overview", "CFTR (kistik fibrozis)"),):
        at = run(page=page, panel_keep="cftr")
        control = at.segmented_control[0]
        assert control.key == "panel_w" and control.value == "cftr", page
        assert control.label == "Gen paneli" and control.proto.label_visibility.value == 2, page   # collapsed, still named
        assert control.options == ["Genel", "Kanser", "PAH", "CFTR"], page   # all four visible, one click apart
        assert full in plain(at), (page, full)         # the short form never stands alone on the page it describes
    for page in ("predictions", "explain"):   # neither says it back: the segment is on the same screen
        assert "CFTR (kistik fibrozis)" not in plain(run(page=page, panel_keep="cftr")), page
    assert "modeli" not in plain(run(page="predictions", panel_keep="cftr"))
    value = text(run())
    # Streamlit reserves the header band and exposes no API for it: the container is lifted into it by CSS,
    # and leaves the flow so that no page pays a row or even a flex gap for it.
    assert ".st-key-panelbar {position:fixed; top:0; right:2rem" in value
    assert "div:has(> .st-key-panelbar) {position:absolute; height:0}" in value
    assert "[data-testid=\"stButtonGroup\"] button[aria-checked=\"true\"]" in value   # the selection is a filled segment
    assert run(lang="en").segmented_control[0].options == ["General", "Cancer", "PAH", "CFTR"]


def test_the_review_page_is_a_nav_entry_nested_under_predictions():
    """Two tabs and a child. Variant review is a visible entry of st.navigation — so the band reaches it, as a
    row click and the table's header link do, and its own URL still routes — but the band draws it as
    Tahminler' child, not a third peer: the icon gives way to a ›, the gap to its parent closes. On /explain
    the framework marks that entry current, and no rule pretends the parent is where you are."""
    source = APP.read_text()
    pages = next(ast.literal_eval(n.value) for n in ast.parse(source).body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "PAGES" for t in n.targets))
    assert [(key, url, vis) for key, url, _, vis in pages] == [
        ("overview", "overview", "visible"), ("pred", "predictions", "visible"), ("explain", "explain", "visible")]
    assert "visibility=vis" in source and "st.switch_page(PAGE[\"explain\"])" in source
    # The nav's entry keeps the table's scroll offset on the way in, as the header's link and a row click do.
    assert """closest('[data-testid="stPageLink"] a, [data-testid="stTopNavLink"][href$="/explain"]')""" in source
    for page in ("overview", "predictions", "explain"):   # all three route
        css = text(run(page=page))
        # The marker is drawn from the app's own mark set, not typed: a › in generated content is also a ›
        # in the link's accessible name.
        assert '[data-testid="stTopNavLink"][href$="/explain"]::before {content:""; --mk:url("data:image/svg+xml' in css, page
        assert '"›"' not in css, page
        assert '[data-testid="stTopNavLink"][href$="/explain"] > span:first-child {display:none}' in css, page
        assert '[data-testid="stTopNavLink"][href$="/predictions"]' not in css, page   # the parent is not drawn current on its child's page

def test_the_variant_page_is_a_drawer_over_the_list():
    """Entering and leaving a variant keeps the list as it was: the bridge turns the page's own address writes
    (same path, new query) into history replacements so Back is one step out, Escape leaves through the page's
    own link, and the table's scroll offset is kept on the way in and put back on the way out. The stretch
    rule above the fold names the chart Python asked to stretch (the framework's height="100%"), never a
    column position, so a content-height chart sharing the column keeps its own height."""
    source = APP.read_text()
    assert "h.pushState = new p.Function('push', 'replace'" in source and "replace : push).call(history" in source
    assert "d.addEventListener('keydown', p.__psEsc, true)" in source and "p.__psTableWatch = new p.MutationObserver" in source
    css = text(run(page="explain", lang="en"))
    assert ('[data-testid="stElementContainer"][height="100%"]:has([data-testid="stVegaLiteChart"]) {flex:1; min-height:0}' in css
            and '[data-testid="stColumn"]:last-child' not in css)
    assert "position:sticky; top:0; z-index:1; background:#fff" in css and "scroll-snap-type:y mandatory" in css
    assert '[data-testid="stSelectboxVirtualDropdown"] [role="listbox"] {max-height:max(18.75rem, min(58.75rem, 100vh - 12rem))' in css

def test_take_row_click_is_an_event_not_a_state():
    """The component re-sends its last value on every rerun and reports None after a remount, so the nonce
    decides: one value drills in once, and the same row clicked twice in a row drills in twice."""
    node = next(n for n in ast.parse(APP.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "take_row_click")
    env = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(APP), "exec"), env)
    take, ss = env["take_row_click"], {}
    assert take(None, ss) is None and ss == {}                       # a mount, before any click
    assert take({"vid": "VAR_1", "n": 1}, ss) == "VAR_1"             # the click
    assert take({"vid": "VAR_1", "n": 1}, ss) is None                # re-sent on the next rerun: not a click
    assert take(None, ss) is None                                    # the component remounted
    assert take({"vid": "VAR_1", "n": 2}, ss) == "VAR_1"             # the same row again: a new nonce
    assert take({"vid": "VAR_2", "n": 3}, ss) == "VAR_2"
    assert ss["click_seen"] == 3
    for junk in ("VAR_2", 3, {}, {"n": 4}):                          # nothing malformed is a drill-in
        assert take(junk, ss) is None, junk

def test_a_results_row_is_the_way_into_the_review_page():
    """Every row carries its variant id for the click bridge to read, and its first cell is a button, so the
    column is keyboard-operable; the variant last inspected comes back marked, so returning lands where the
    presenter was. The click itself is a custom component and is browser-tested."""
    at = run(page="predictions", lang="en")
    value, ids = text(at), table_ids(at)
    assert len(ids) == 300
    assert f"""<tr data-vid="{ids[0]}"><td><button type='button' class='row'>{ids[0]}</button></td>""" in value
    assert value.count("class='row'") == 300 and "class='now'" not in value
    marked = text(run(page="predictions", lang="en", pick_keep={"general": ids[7]}))
    assert f"""<tr class='now' data-vid="{ids[7]}">""" in marked and marked.count("class='now'") == 1
    assert table_ids(run(lang="en")) == []   # Overview's reference tables are not a way anywhere

def test_a_variant_in_the_url_seeds_a_fresh_session():
    """A page load is a new session — which is why the drill-in is a component and not a link — so the URL is
    all that survives one: /explain?variant=ID reopens the review page on that variant, and the page writes
    the address back from Python as the pick moves. A saved pick outranks it; nothing follows it as a link."""
    ids = table_ids(run(page="predictions", lang="en"))
    at = run(page="explain", lang="en", query_params={"variant": ids[42]})
    assert at.session_state["pick_keep"]["general"] == ids[42]
    assert at.query_params["variant"] == [ids[42]] and at.query_params["panel"] == ["general"]   # the panel goes with it
    at = run(page="explain", lang="en", query_params={"variant": ids[42]}, pick_keep={"general": ids[9]})
    assert at.session_state["pick_keep"]["general"] == ids[9]        # the session's own pick wins
    assert at.query_params["variant"] == [ids[9]]                    # ... and the address is corrected to it
    assert run(page="predictions", lang="en").query_params == {}     # the table page carries no variant
    # A fresh session takes the panel the address names, so a shared variant of another panel reopens in it;
    # a session that already has a panel keeps it, and an unknown name is ignored.
    at = run(page="explain", lang="en", query_params={"panel": "cancer", "variant": "VAR_000687"})
    assert at.session_state["panel_keep"] == "cancer" and at.query_params["panel"] == ["cancer"]
    assert run(page="explain", lang="en", query_params={"panel": "cancer"}, panel_keep="pah").query_params["panel"] == ["pah"]
    assert run(page="explain", lang="en", query_params={"panel": "nope"}).session_state["panel_keep"] == "general"

def test_language_is_chosen_at_launch_not_in_the_page():
    """PATHOSENSE_LANG=en selects English when the interface starts; there is no in-app toggle."""
    for value, expected, visible in ((None, "tr", "Çapraz doğrulama sonuçları"), ("en", "en", "Cross-validated results"),
                                     ("tr", "tr", "Çapraz doğrulama sonuçları"), ("nonsense", "tr", "Çapraz doğrulama sonuçları")):
        os.environ.pop("PATHOSENSE_LANG", None)
        if value is not None:
            os.environ["PATHOSENSE_LANG"] = value
        try:
            at = run()
        finally:
            os.environ.pop("PATHOSENSE_LANG", None)
        assert at.session_state["lang"] == expected, value
        assert not [b for b in at.button if b.key == "lang_toggle"], value
        assert visible in text(at), value


def test_panel_prevalence_and_variant_survive_navigation():
    """The mirrors are what survive a rerun and a page change; the widgets are remounted from them."""
    at = run()
    at.segmented_control[0].set_value("cftr").run()
    next(w for w in at.select_slider if w.key == "prev_w").set_value(0.45).run()
    assert at.session_state["prev_keep"] == 0.45
    on(at, "explain")
    picker = next(w for w in at.selectbox if w.key == "pick_w")
    picker.set_value(picker.options and 0).run()
    selected = dict(at.session_state["pick_keep"])
    # An English session starts in English and stays there, so it is exercised from its own launch state:
    # AppTest can only re-serialize a localized format_func in the language of the session that built it.
    english = run(page="explain", lang="en", panel_keep="cftr", prev_keep=0.45,
                  pick_keep=dict(selected))
    for at in (at, english):
        for _ in range(2):
            on(at, "predictions")
            on(at, "overview")
            on(at, "explain")
            assert at.session_state["panel_keep"] == "cftr"
            assert at.session_state["prev_keep"] == 0.45
            assert at.session_state["pick_keep"] == selected


def test_layout_scale_and_hidden_probe():
    """The reading size is fixed, not fluid: the root, the three roles and every chart label are the same at
    every viewport the product is specified for, and extra height goes to plot area, never to labels."""
    for width, height in ((1024., 700.), (1366., 768.), (1920., 1080.), (2560., 1440.), (2560., 1800.)):
        at = run(viewport=(width, height))
        value = text(at)
        assert "html {font-size:13px}" in value and "clamp" not in value.split("</style>")[0].split("html {")[1][:40]
        spec = json.loads(at.get("vega_lite_chart")[0].proto.spec)
        assert spec["config"]["axis"]["labelFontSize"] == 12
    assert "visibility:hidden" in value
    # Three steps that carry different jobs, and a table whose leading is a density decision of its own.
    assert "--fs-sm:12px; --lh-sm:16px;" in value and "--fs:13.5px; --lh:17px;" in value
    assert "--fs-lg:20px; --lh-lg:24px;" in value and "--row-lh:16px" in value
    assert '(min-width:1101px) and (min-height:769px)' in value
    # The filter list is virtualised with a row height the framework computes in JS: without the override it
    # draws 40px rows for 1.4rem of text, at a density nothing else on the page uses.
    assert 'stMultiSelectDropdown"] [role="option"]' in value
    assert "zoom:" not in value and "transform:scale" not in value


def test_training_rows_get_no_aggregate_metrics():
    """The artifacts are fit on the whole panel CSV, so a batch of training rows has nothing to report."""
    count_cards = {"Called pathogenic", "Warning types", "Warning and label agreement"}   # the batch's count is the base of the first card
    blind_cards = {"Called pathogenic", "Warning types"}   # no labels: no verdict can be stated, so the row is the two cards that do not need them
    at = run(page="predictions", lang="en")                             # the fixed demo sample
    value = text(at)
    # Visible in a screenshot: what the file is, and whose rows. How many rows was a clause here too — the
    # count card below states the batch as the base of its own fraction ("176 / 300").
    assert "Training-set sample" in plain(at) and chips(at) == ["in-sample"] and "300 variants" not in plain(at)
    assert set(cards(at)) == count_cards and not at.metric   # counts of the file; not one metric of the model
    # Each count states the reference it is read against on its own card: a share of rows called pathogenic
    # describes the file, and the batch's own composition is the qualifier that stops 58.7% being read
    # against the 14.3% test prevalence on the sibling page. The misreading happens at a glance, so the
    # figure stays on the card and out of a tooltip.
    assert cards(at)["Called pathogenic"][:2] == ("176 / 300", "227 labelled")
    assert "not an output of the model" in cards(at)["Called pathogenic"][2]
    # The four states the truth column marks its rows with, all four on the card that is that column's key:
    # agreement crossed with whether anything warned, each cell stating what its count is a count of. The
    # bottom-left one — wrong, and no flag fired — is the competition's false-positive / false-negative
    # question, so it is the cell the card is built around: the only red, and the only cell that is tinted.
    # Beside them, on this path too, the warnings themselves: which of the four fired and on how many rows,
    # in the legend's order and in the same 2×2, so the two cards are one shape and their figures one pair
    # of lines. It replaced a card counting the flagged rows, which is the head of the warned column here.
    assert verdicts(at) == {"Correct, no warning: 197 / 220": ("197", "yes", "ok", ""),
                            "Correct, warned: 42 / 80": ("42", "yes", "warn", ""),
                            "Wrong, nothing warned: 23 / 220": ("23", "no", "bad", "lead"),
                            "Wrong, warned: 38 / 80": ("38", "no", "warn", ""),
                            "conservation missing: 37 / 300": ("37", "cons_missing", "fl", ""),
                            "non-standard AA code: 36 / 300": ("36", "complex_aa", "fl", ""),
                            "low-quality region: 4 / 300": ("4", "region_flag", "fl", ""),
                            "borderline: 40 / 300": ("40", "borderline", "fl", "")}
    # Four counts, so no single value line and no qualifier under the title: each figure states the rows it
    # is a count of, on the figure, which is what left both titles bare.
    assert cards(at)["Warning and label agreement"][:2] == ("", "")
    assert cards(at)["Warning types"][:2] == ("", "")
    assert "The four cells are all 300 rows" in cards(at)["Warning and label agreement"][2]
    assert "not a measure of the model" in cards(at)["Warning and label agreement"][2]
    # The suppression is a decision, not an absence: the cards carry counts and no metric of the model, the
    # heading says whose rows these are, and why they have no metrics of their own hangs off its marker. The
    # panel's own cross-validated result is not restated here — Overview is its home.
    assert "cross-validated F1" not in value
    assert "batch metrics would not measure performance" in hints(at) and "Batch metrics not shown" not in value
    assert "seed 42" not in plain(at) and "Fixed sample, seed 42." in hints(at)   # the seed stays reachable, off the line
    assert "Label comparison" not in value
    # The foot count is drawn only where the listing is not the batch. Unfiltered it said what the absence of
    # a filter already said; the batch size is stated once, as the base of the count card's own fraction.
    assert "300 / 300 rows" not in value and "first 300 shown" not in value
    assert cards(at)["Called pathogenic"][0] == "176 / 300"
    assert "class='tablewrap tall' tabindex='0'" in value and "True label" in value   # row-level agreement stays
    assert "<span class='tm tm-ok'><span class='mk mk-yes'></span></span>" in value and ">Correct<" not in value   # ... as a mark in the label cell
    big = text(upload(csv(pd.read_csv(PANELS["general"], nrows=600)), lang="en"))     # the 500-row cap, now only an upload reaches it
    assert "500 / 600 rows" in big and "first 500 shown" not in big
    df = sample()
    at = upload(csv(df), lang="en")                                     # an upload whose IDs are all training rows
    assert set(cards(at)) == count_cards and "cross-validated F1" not in text(at)
    assert "training data for this model" in hints(at)                   # said once, off the heading's marker
    blind = upload(csv(df.drop(columns="Label")), lang="en")             # ... and the same with no labels to compare
    assert set(cards(blind)) == blind_cards and not cards(blind)["Called pathogenic"][1]
    # Nothing can be called right or wrong here, so no trust mark is drawn at all and no verdict is named:
    # what is left is the warning itself, in the flag vocabulary, each count against the rows of the batch.
    assert "class='tm " not in text(blind) and not cards(blind)["Warning types"][1]
    assert all(f"/ 25</span>" in c for c in re.findall(r"<b>\d+<span class='base'>[^<]*</span>", text(blind)))
    assert all(re.fullmatch(r"[\w\- ]+: \d+ / 25", name) for name in verdicts(blind)), verdicts(blind)
    assert {glyph for _, glyph, _, _ in verdicts(blind).values()} == {"cons_missing", "complex_aa", "region_flag", "borderline"}
    assert "training data for this model" in hints(blind) and "cross-validated F1 is" not in text(blind)
    df.loc[df.index[:12], "Variant_ID"] = [f"new-{i}" for i in range(12)]
    at = upload(csv(df), lang="en")                                     # partly training rows: the metrics stay
    value = text(at)
    assert "ROC AUC" in metrics(at) and "13/25 IDs in the training set" in chips(at)   # the model's own metrics are still st.metric cards
    assert "not evidence of independence" in hints(at)
    assert "<div class='small'>Identifier overlap" not in value       # relocated out of the body, not restated
    assert "training data for this model" not in value and "in-sample" not in value


def test_provenance_follows_the_artifact_not_the_panel_name():
    """PAH and CFTR are scored by a General-trained model that excluded their variants: their rows are
    training data for the calibration and the threshold only, so the metrics are shown with that caveat."""
    for panel, rank in (("pah", True), ("cftr", False)):
        at = run(page="predictions", panel_keep=panel, lang="en")
        value, hint = text(at), hints(at)
        assert "ROC AUC" in metrics(at), panel
        assert chips(at)[:2] == ["transfer from General", "only calibration in-sample"], panel   # beside the optimistic F1
        # The line says the model never saw the rows; the tooltip says only what the line does not: that the
        # F1 and MCC under it are therefore optimistic.
        assert "F1 and MCC here are optimistic" in hint and "not held-out validation" in hint, panel
        assert "never saw these rows" not in hint, panel
        assert "Label comparison" in plain(at) and "not held-out" in chips(at), panel
        assert "unseen by the model" not in value, panel   # said once, by the heading's own tag
        assert "training data for this model" not in hint, panel
        assert ("AF rank transform" in hint) == rank, panel
    at = run(page="predictions", panel_keep="cancer", lang="en")
    assert "Label comparison" not in text(at) and chips(at)[:2] == ["in-sample", "50% weight"]
    # Said once: the tag beside the heading is the visible half, and the share that was fitted on these rows
    # hangs off the heading's marker — nothing under the batch repeats it.
    assert "cross-validated F1" not in text(at)
    assert hints(at).count("50% weighted share of the score") == 1
    # Without labels there is no such line, and the heading's tooltip is where the claim lives.
    assert "Every row here is training data" in hints(upload(csv(sample().drop(columns="Label")), lang="en"))


def test_prior_adjusted_f1_is_withheld_without_support():
    at = run(page="predictions", panel_keep="cftr", lang="en")   # the whole 111-row panel, ~21 benign rows
    assert "ROC AUC" in metrics(at) and not any(m.startswith("F1 (test prevalence") for m in metrics(at))
    assert "F1 at test prevalence withheld: at least 30 rows in each class are needed (90 pathogenic / 21 benign)." in text(at)
    at = run(page="predictions", panel_keep="pah", lang="en")
    assert any(m.startswith("F1 (test prevalence") for m in metrics(at))


def test_review_counts_describe_the_listed_variants():
    value = text(run(page="explain", filter_keep=("flag_borderline",), lang="en"))
    # How many the picker is offering and what narrowed it. Not how many of those carry a flag or sit near
    # the threshold: this page is about one variant, and under a flag filter that count was the filter's own
    # name said back as a number.
    assert "40/300 variants · Filter: borderline" in value and "40 flagged" not in value
    assert "40/300 variants" not in text(run(page="explain", lang="en"))   # unfiltered, "300 / 300" is a tautology


def test_the_wrong_call_filter_counts_its_rows_like_every_other_option():
    """The count used to be withheld on a training batch, where an error rate measures nothing — but the table's
    own foot line prints the same number the moment the filter is on, so the withholding hid it from nobody and
    left two controls saying different things about one set of rows. Withheld metrics are aggregates that read
    as performance; how many rows a filter lists is not one, and the heading says whose rows they are."""
    at = run(page="predictions", lang="en")                              # the demo sample: the model's own training rows
    wrong = next(o for o in at.multiselect[0].options if o.startswith("wrong call"))
    assert re.fullmatch(r"wrong call · \d+", wrong), wrong
    value = text(run(page="predictions", filter_keep=("wrong",), lang="en"))
    assert f"{wrong.split(' · ')[1]} / 300 rows" in value               # the number the foot line one column over prints
    assert "in-sample" in chips(run(page="predictions", filter_keep=("wrong",), lang="en"))
    df = sample(); df["Variant_ID"] = [f"new-{i}" for i in range(len(df))]
    at = upload(csv(df), lang="en")                                      # unknown provenance: metrics, and the count
    assert any(o.startswith("wrong call · ") for o in at.multiselect[0].options)
    assert all(o[:1] == o[:1].lower() for o in at.multiselect[0].options), at.multiselect[0].options


def test_claims_that_stop_a_misreading_stay_reachable():
    """The app is a showcase, not an explainer: the sentences went, the claims did not. Each one is
    either a visible tag or the tooltip of the exact element it qualifies."""
    for lang, tags, claims in (
            ("en", ("in-sample", "at test prevalence", "extrapolation",
                    "transfer from General", "only calibration in-sample"),
             ("batch metrics would not measure performance", "not evidence of independence",
              "not held-out validation", "an extrapolation, not a measurement",
              "not a decomposition of the calibrated or blended probability", "Not a clinical validation")),
            ("tr", ("örneklem içi", "test prevalansında", "ekstrapolasyon",
                    "Genel'den aktarım", "yalnızca kalibrasyon örneklem içi"),
             ("küme metrikleri performans ölçmez", "bağımsızlık kanıtı değildir",
              "Dış doğrulama değildir", "ekstrapolasyondur, ölçüm değildir",
              "ayrıştırması değildir", "Klinik doğrulama değildir"))):
        overview = run(lang=lang)
        general = run(page="predictions", lang=lang)                       # in-sample, labelled
        blind = upload(csv(sample().drop(columns="Label")), lang=lang)     # in-sample, unlabelled
        pah = run(page="predictions", panel_keep="pah", lang=lang)         # transfer
        review = run(page="explain", lang=lang)
        # A labelled upload whose IDs the training set does not hold. The independence claim qualifies the ID
        # comparison, and no demo page carries one (every demo ID is a training ID). The unlabelled finals
        # batch says the same on its batch heading, in its own words: test_an_unlabelled_batch_says_...
        novel = upload(csv(sample().assign(Variant_ID=[f"new-{i}" for i in range(25)])), lang=lang)
        pages = (overview, general, blind, pah, review, novel)
        visible = "\n".join(plain(a) for a in pages)
        # Every one of them is a chip on the line it qualifies, not a clause of that line and not a hover:
        # one treatment at every site, so a screenshot of any of these pages carries the claim as an object.
        marked = [c for a in pages for c in chips(a)]
        for tag in tags:
            assert tag in marked, (lang, tag)
            assert tag in visible, (lang, tag)
        reachable = visible + "\n" + "\n".join(hints(a) for a in pages) + "\n" + "\n".join(m.help or "" for m in overview.metric)
        for claim in claims:
            assert claim in reachable, (lang, claim)
        # The in-sample claim is said exactly once per batch, labelled or not, and the heading's marker is
        # where it is said: nothing is drawn under the batch to carry it. It is the tooltip and never the
        # line, and the visible `in-sample` tag is in the same screenshot either way.
        insample = claims[0]
        assert hints(general).count(insample) == 1 and hints(blind).count(insample) == 1, lang
        assert insample not in plain(general) and insample not in plain(blind), lang


def test_review_input_list_names_each_feature_once():
    """The column beside the contribution chart lists the features that chart ranks, with the raw input
    behind each one — the values are otherwise only in a hover tooltip, and the column ran out of content
    long before the chart did. The three standard inputs stay, and a standard input that is also a driver
    is listed once. The chart names its bars, so the list keys a feature the chart draws by its column id
    alone: no feature name is printed twice on one screen. A feature the chart does not draw carries its
    name in the list instead, because there is nothing beside it to cross-refer to."""
    for panel, lang in (("general", "en"), ("cancer", "tr")):   # cancer also draws member bars
        at = run(page="explain", panel_keep=panel, lang=lang)
        inputs = re.findall(r"<tr><td>(.*?)</td>", re.findall(r"<table class='kv'>(.*?)</table>", text(at), re.S)[-1])
        assert len(inputs) >= 9 and len(inputs) == len(set(inputs)), (panel, inputs)
        bars = [label for label in chart_text(at).split("\n") if " · " in label]   # axis labels are "id · name"
        drawn = {label.split(" · ", 1)[0] for label in bars}
        named = {label.split(" · ", 1)[1] for label in bars}
        assert not set(inputs) & named, (panel, set(inputs) & named)   # no name printed twice on one screen
        assert len(set(inputs) & drawn) >= 7, (panel, inputs)          # the drivers are keyed by their id
        assert "EK_7" in inputs, panel
    picker = next(w for w in at.selectbox if w.key == "pick_w")
    visibility = picker.proto.label_visibility
    assert visibility.value == visibility.COLLAPSED and picker.label == "Varyant"   # collapsed, still named


def test_enrichments_land_in_existing_cells():
    """Zero-height additions: the fitted row count and the single-input AUC baseline in the results table, the
    substitution chemistry on the amino-acid row, and the expected call rate as a tooltip on an unknown batch."""
    from schema import drop_all_nan_rows
    value = text(run(lang="en"))
    for panel in PANELS:
        art = load_artifacts(panel)
        fitted = len(drop_all_nan_rows(pd.read_csv(PANELS[panel]), verbose=False))   # not the raw CSV row count
        assert f"{fitted} · {art['training_prior']:.1%}" in value, panel
        if art["transfer"]:
            assert fitted == art["transfer"]["art"]["n_excluded"], panel               # the count the flow already draws
    assert re.search(r"0\.864 <span class='base'>· single input (?:phyloP|phastCons|GERP\+\+) 0\.7\d\d</span>", value)
    at = run(page="explain", panel_keep="cancer", lang="en")                      # AA_grantham is a driver here
    value = text(at)
    assert re.search(r"Amino-acid change</td><td><b>[A-Z] → [A-Z] · Grantham \d+ · BLOSUM62 -?\d+</b>", value), value
    inputs = re.findall(r"<tr><td>(.*?)</td>", re.findall(r"<table class='kv'>(.*?)</table>", value, re.S)[-1])
    assert "AA_grantham" in chart_text(at) and "AA_grantham" not in inputs      # read once, on the amino-acid row
    card = lambda a: cards(a)["Called pathogenic"][2]
    assert "Expected call rate" not in card(run(page="predictions", lang="en"))   # the training sample: meaningless there
    assert "Expected call rate" not in card(run(page="predictions", panel_keep="pah", lang="en"))
    df = sample()
    assert "Expected call rate" not in card(upload(csv(df), lang="en"))           # a file of training rows, likewise
    df["Variant_ID"] = [f"new-{i}" for i in range(len(df))]
    assert "Expected call rate at test prevalence (14.3%): 24.8%" in card(upload(csv(df), lang="en"))
    assert "24.8%" not in plain(upload(csv(df), lang="en"))                     # a tooltip, never a visible tag


def test_the_batch_block_describes_the_file_not_the_scores():
    """Under the score distribution, what the batch is made of: its commonest substitutions ranked, and its
    central Grantham and BLOSUM62. Counted here from the input columns with neither the app nor the model in
    the way. It is what the file contains and never how well the model did on it — the same block with and
    without Label — and a file whose chemistry is unusable draws nothing at all."""
    from collections import Counter
    from feature_engineering import GRANTHAM
    std = set("ACDEFGHIKLMNPQRSTVWY") | {"*"}
    for panel in PANELS:
        at = run(page="predictions", panel_keep=panel, lang="en")
        value = text(at)
        rows = pd.read_csv(PANELS[panel])
        rows = rows.sample(n=min(300, len(rows)), random_state=42)     # the demo sample, drawn the same way
        pairs = [(a, b) for a, b in zip(rows["AA_1"], rows["AA_2"]) if a in std and b in std]
        top = sorted(Counter(f"{a} → {b}" for a, b in pairs).items(), key=lambda kv: (-kv[1], kv[0]))[:4]
        grantham = sorted(GRANTHAM[a][b] for a, b in pairs if a in GRANTHAM and b in GRANTHAM[a])
        half = len(grantham) // 2
        median = grantham[half] if len(grantham) % 2 else (grantham[half - 1] + grantham[half]) / 2
        assert f"Substitution chemistry · median Grantham {median:g} · BLOSUM62 " in plain(at), panel
        drawn = re.findall(r"<b>(\S+ → \S+)</b>.*?</span></span>(\d+)</span>", value)
        assert drawn == [(change, str(k)) for change, k in top], (panel, drawn)
    at = run(page="predictions", panel_keep="pah", lang="en")   # above the label comparison, where there is one
    assert text(at).index("class='chem'") < text(at).index("Label comparison")
    df = sample(); df["Variant_ID"] = [f"new-{i}" for i in range(len(df))]
    block = lambda a: re.search(r"<div class='chem'>.*?</div>", text(a), re.S)[0]
    labelled, blind = upload(csv(df), lang="en"), upload(csv(df.drop(columns="Label")), lang="en")
    assert "ROC AUC" in metrics(labelled) and "ROC AUC" not in metrics(blind)
    assert block(labelled) == block(blind)          # the block is blind to the column the metrics come from
    # Which warnings fired and how many of each, drawn as figures on the card rather than spelled out in its
    # tooltip — the mark is the name, and the legend under the table on this same screen spells it. Every
    # figure states the rows it is counted against, as the cards beside it do.
    fired = {glyph: int(count) for count, glyph, tone, _ in verdicts(labelled).values() if tone == "fl"}
    assert set(fired) == {"cons_missing", "complex_aa", "region_flag", "borderline"}
    assert all(re.fullmatch(r"[\w\- ]+: \d+ / 25", name) for name, v in verdicts(labelled).items() if v[2] == "fl")
    card = cards(labelled)["Warning types"][2]
    # What the figures cannot say is left in the marker, and only that: a row can raise several warnings, so
    # the four do not add up to the rows that carry one. The names are not repeated there — they are on the
    # card as marks and under the table as words.
    warned = int(re.search(r"(\d+) of \d+ rows raised at least one", card)[1])
    assert max(fired.values()) <= warned <= sum(fired.values()), (card, fired)
    assert not any(name in card for name in ("conservation missing", "non-standard AA code", "low-quality region"))
    # What the warnings amount to against the model's own mistakes is the card beside it, because that
    # number is the point of the row and not a footnote to this one.
    assert "wrong" not in card and "Wrong, nothing warned: 2 / 20" in verdicts(labelled)
    assert text(labelled).count("<div class='key'>") == 1
    assert "Substitution chemistry" not in text(upload(csv(df.assign(AA_1="Ala", AA_2="Val")), lang="en"))


def test_the_call_is_a_line_not_a_column():
    """The table is sorted by score, so the call is monotonic under any filter and the model's answer is a
    place in the list: it is stated once, as the labelled line the list crosses, instead of once per row in
    words. The line names both classes and the threshold, so the class of a row is still readable without
    colour. It is not a row of the data — no id, so the click bridge passes over it — and where a filter
    leaves one class there is no crossing and no line."""
    at = run(page="predictions", lang="en")
    value = text(at)
    assert "Model call" not in value and "<td>Pathogenic</td>" not in value      # the words column is gone
    split = re.findall(r"<tr class='split'>.*?</tr>", value, re.S)
    assert len(split) == 1 and "data-vid" not in split[0]
    assert split[0] == ("<tr class='split'><td colspan='5'><div class='splitline'><span>Pathogenic ↑</span>"
                        "<span>Decision threshold 0.81</span><span>↓ Benign</span></div></td></tr>"), split[0]
    assert len(table_ids(at)) == 300                        # ... and it is not one of the rows it separates
    ids = table_ids(at)
    above, below = value.index(f'data-vid="{ids[0]}"'), value.index(f'data-vid="{ids[-1]}"')
    assert above < value.index(split[0]) < below            # drawn where the score crosses, not at either end
    for one_class in (("pathogenic",), ("benign",)):        # one class listed: no crossing, no line
        assert "class='split'" not in text(run(page="predictions", filter_keep=one_class, lang="en")), one_class
    assert "class='split'" in text(run(page="predictions", filter_keep=("pathogenic", "benign"), lang="en"))
    for panel in PANELS:   # the line carries the panel's own threshold, in both languages
        art = load_artifacts(panel)
        assert f"Karar eşiği {art['threshold']:.2f}</span>" in text(run(page="predictions", panel_keep=panel)), panel


def test_the_truth_sits_beside_the_score_and_the_row_names_its_substitution():
    """The order the columns are read in: which variant, what it changes, what the model made of it, what it
    actually is, and last the annotations that qualify the row. The unlabelled file is the shape the finals
    are run on, so it is the same table minus one column in the middle — never a stub at the end. The
    substitution comes from AA_1/AA_2; a multi-residue or missing code draws nothing at all and the ⇄ on the
    same row is what says why. None of it reaches the export, which keeps its own columns."""
    at = run(page="predictions", lang="en")
    labelled = text(at)
    head = lambda names: "".join(f'<th scope="col">{n}</th>' for n in names)
    hint = r" ?<span class='hint'[^>]*><span class='mk mk-info'></span></span>"   # hint_mark leads with a space
    assert head(["Variant", "Change", "Model score", "True label", "Review flags"]) in re.sub(hint, "", labelled)
    # The trust key hangs off the truth column's own marker: measured at 1366 in Turkish the three entries
    # want 265px on a legend line with 167px left on it, and a coloured mark among the four ink flag marks
    # in that legend would read as a fifth flag.
    assert re.search('<th scope="col">True label' + hint, labelled) and "Check: the call matches the label" in labelled
    # The truth is a word in plain ink and the agreement mark carries the row's reliability in a tone of its
    # own: green where the call matches the label and no flag fired, gold on either mark where one did, and
    # red on a wrong call nothing warned about. The word gave up its class colour so that the chip could be
    # a traffic light — a red cross beside a red "Pathogenic" would be one colour meaning two things a
    # character apart, and it is the word that yields, because the chip is the thing with three states.
    ok = "<span class='tm tm-ok'><span class='mk mk-yes'></span></span>"
    assert f"<span class='truth'><b class='cls'>Pathogenic</b> {ok}</span>" in labelled
    assert "<b class='cls'>Benign</b> <span class='tm tm-bad'><span class='mk mk-no'></span></span>" in labelled
    assert "style='color:#C0392B'>Pathogenic" not in labelled and ".cls {font-weight:600; color:var(--ink)}" in labelled
    assert "<span class='tm tm-warn'><span class='mk mk-yes'></span></span>" in labelled   # correct, and a flag fired
    assert "<span class='tm tm-warn'><span class='mk mk-no'></span></span>" in labelled    # wrong, and a flag warned
    tones = dict(re.findall(r"\.tm-(\w+) \{color:(#[0-9A-F]{6})\}", labelled))
    # The trust tones kept clear of every call and brand colour while the class word was painted red or
    # blue. That word is ink now, so nothing else in the cell is red and the failure chip is RED itself —
    # shared on purpose, since a second red a few degrees from the first would be the real defect. BLUE and
    # BRAND stay out of the set: blue still means a benign call on the same row's score bar, and the brand
    # teal is interface state everywhere in the app.
    assert set(tones) == {"ok", "warn", "bad"} and not set(tones.values()) & {"#2471A3", "#0F6E66"}, tones
    assert tones["bad"] == "#C0392B" and "#C0392B" not in {tones["ok"], tones["warn"]}, tones
    # The tone is carried by a filled chip with the glyph knocked out of it (one evenodd path: chip subpath,
    # then glyph subpath), not by a hairline — a tone on a 1.5px stroke is a thread of colour at the back of
    # a hall. The four review flags stay outlined at that one stroke weight, and solid-against-outlined is
    # what keeps the agreement mark out of their vocabulary instead of reading as a fifth flag.
    urls = dict(re.findall(r'\.mk-(\w+) \{--mk:url\("([^"]*)"\)\}', labelled))
    assert all("fill-rule='evenodd'" in urls[k] for k in ("yes", "no")), "the agreement marks are knockouts"
    # ... and the hole is 2px wide, the cap radius of both glyphs: below about 1.8 the negative space closes
    # up at 16px, and at the 2.5 it shipped with the white was taking more of the chip than the tone was.
    assert all("A1 1 0 0 1 " in urls[k] for k in ("yes", "no")), "the knockout is 2px"
    assert not any("fill='black'" in urls[k]
                   for k in ("cons_missing", "complex_aa", "region_flag", "borderline")), "flags stay hairline"
    # The cell is the widest class word plus one fixed gap, not the whole column: the marks still share one
    # x, close enough to the word to be read with it rather than with the review flags.
    assert ".truth {display:inline-flex; align-items:baseline; gap:0.375rem}" in labelled and ".truth .cls {min-width:" in labelled
    # What the flags amount to against the model's mistakes, as the card the row is built around.
    assert verdicts(at)["Wrong, nothing warned: 23 / 220"] == ("23", "no", "bad", "lead")
    # The same encoding on the page that shows one variant, so the two cannot disagree about that variant.
    assert re.search(r"True label: <b class='cls'>\w+</b> "
                     r"<span class='tm tm-(?:ok|warn|bad)'>", text(run(page="explain", lang="en")))
    rows = pd.read_csv(PANELS["general"]).sample(n=300, random_state=42).set_index("Variant_ID")
    for vid in table_ids(run(page="predictions", lang="en"))[:5]:
        assert f"<td>{rows.loc[vid, 'AA_1']} → {rows.loc[vid, 'AA_2']}</td>" in labelled, vid
    df = sample(); df["Variant_ID"] = [f"new-{i}" for i in range(len(df))]
    blind = text(upload(csv(df.drop(columns="Label")), lang="en"))
    assert head(["Variant", "Change", "Model score", "Review flags"]) in blind and "True label" not in blind
    complexed = text(upload(csv(df.assign(AA_1="Ala", AA_2="Val")), lang="en"))
    assert "Ala → Val" not in complexed                       # nothing, quietly — not "Ala → Val", not an error
    assert complexed.count("<span class='fl'><span class='mk mk-complex_aa'></span></span>") == 25 + 2   # every row, the legend under them, and the card that counts them
    # The export is not the view's columns: it keeps its own, and neither the divider line nor a symbol is
    # in it. It leads with the confirmed pair — the id and the call as 0/1 — and what the app read the call
    # by follows, the file's own label included where it carries one.
    source = APP.read_text()
    export = source[source.index("table_csv = pd.DataFrame("):source.index("with actions:")]
    assert '"Variant_ID": [ids[i] for i in order], "prediction": preds[order]' in export
    assert export.index("prediction") < export.index("probability") < export.index('"flag_" + k')
    assert 'table_csv["label"] = y[order]' in export
    assert not any(name in export for name in ("flag_symbols", "divider", "col_aa", "AA_1"))


def test_review_flag_rules_are_named_not_narrated():
    at = run(page="explain", filter_keep=("flag_borderline",), lang="en")   # the default pick is the closest to threshold
    value = text(at)
    assert "<li class='bad' tabindex='0' data-tip='Within 0.05 of the threshold.'" in value
    assert "may flip the call" not in value   # the distance is the fact; what it implies is not the tool's to say
    assert re.search(r"aria-description='Within 0\.05 of the threshold\.'>borderline</li>", value)   # ... and in the accessible description
    assert ">borderline</li>" in value                                    # the line itself is the rule's name


def test_review_flags_are_symbols_under_one_legend():
    """The results column is a symbol per flag, language-independent; the words are in the row's title and in
    the legend under the table, which shares its line with the row count. Every mark is drawn by one call
    (flag_mark) into one fixed box, so the cell's mark and the legend's mark are the same mark and the four
    of them share a footprint and a baseline instead of sitting at four heights."""
    at = run(page="predictions", filter_keep=("flag_cons_missing",), lang="en")
    value = text(at)
    # Four slots in the legend's order, so a mark is at the same x on every row: the two fired here draw
    # their glyphs, the two that did not leave their boxes empty (silent inside the role=img span).
    fl = lambda name: f"<span class='fl'><span class='mk mk-{name}'></span></span>" if name else "<span class='fl'></span>"
    marks = fl("cons_missing") + fl("complex_aa") + fl("") + fl("")
    names = "conservation missing, non-standard AA code"   # the cell's accessible name and its tooltip, one string
    cell = f"<span class='fls' role='img' aria-label='{names}' data-tip='{names}'>{marks}</span>"
    assert f"<td>{cell}</td>" in value
    cells = re.findall(r"<span class='fls'[^>]*>(.*?)</span></td>", value)
    assert cells and all(c.count("<span class='fl'>") == 4 for c in cells)   # every flagged row: one slot per flag
    unflagged = text(run(page="predictions", lang="en"))
    assert "<td></td></tr>" in unflagged and fl("") * 4 not in unflagged
    assert "<td>conservation missing" not in value            # the prose is gone from the cell
    assert fl("cons_missing") + "conservation missing" in value   # ... and named once, in the legend
    # Drawn, not typed: four paths in one stroke weight, and none of the codepoints they replaced.
    assert all(f"mk-{k}" in value for k in ("cons_missing", "complex_aa", "region_flag", "borderline"))
    assert not any(glyph in value for glyph in "∅⇄▲◫")
    assert ".fl {display:inline-block; width:16px; height:16px; vertical-align:middle" in value
    assert ".mk {display:inline-block; width:16px; height:16px; vertical-align:middle" in value
    assert "<span class='count'>37 / 300 rows · Filter: conservation missing</span></div>" in value
    assert "<div class='small'>37 / 300 rows" not in value            # the count moved under the table
    assert text(run(page="predictions", filter_keep=("flag_cons_missing",))).count(marks) == value.count(marks)


def test_hover_is_one_tooltip_on_every_surface():
    """Three things answer a hover — the app's own marks, the framework's help icons and the charts' Vega
    tooltip — and they are one object. No native title= anywhere: the OS draws that in its own face, at its
    own delay, and it cannot be styled. Every mark of the app's own carries data-tip for the one document
    tooltip the probe installs, and keeps an accessible name or description beside it. One CSS rule styles
    all three surfaces, and the delay and dismissal are one pair of numbers, written once, read by the
    script and by the Vega element's transitions."""
    source = APP.read_text()
    assert 'js_expressions=TIP_JS + ";" + PREV_JS + ";" + ROW_CLICK_JS' in source and "title='" not in source
    for page, panel in (("overview", "pah"), ("predictions", "general"), ("explain", "general")):
        value = text(run(page=page, panel_keep=panel, lang="en", filter_keep=("flag_borderline",)))
        assert " title=" not in value, page
        tips = re.findall(r"<(?:span|li)[^>]*data-tip='[^']*'[^>]*>", value)
        assert tips, page
        assert all("aria-label='" in tag or "aria-description='" in tag for tag in tips), (page, tips)
    css = text(run(lang="en"))
    assert '#ps-tip, [data-testid="stTooltipContent"], #vg-tooltip-element {' in css
    assert "--tip-open:200ms; --tip-close:300ms" in css and "TIP_OPEN, TIP_CLOSE = 200, 300" in source
    assert "#vg-tooltip-element.visible {opacity:1; transition:opacity var(--tip-fade) var(--tip-open)" in css
    assert re.search(r'#ps-tip, \[data-testid="stTooltipContent"\], #vg-tooltip-element \{[^}]*font-size:var\(--fs-sm\)', css)   # the 12px floor, on all three


def test_review_family_totals_are_a_chart_and_the_steppers_join_the_picker():
    """The family totals are their own small diverging chart above the contribution chart — the whole, then the
    detail — and the text key that used to state the same five numbers is gone. The two steppers are inside
    the picker's row as one bordered pair, not a column of their own; a hovered results row says what a click
    does with a mark drawn out of flow, so the row keeps its height."""
    at = run(page="explain", lang="en")
    assert "Family totals" not in text(at) and "class='key'" not in text(at)
    charts = at.get("vega_lite_chart")
    assert len(charts) == 3                                              # gauge, family totals, contributions
    family = pa.ipc.open_stream(charts[1].proto.datasets[0].data.data).read_all().to_pydict()
    assert {"family", "contrib", "color"} <= set(family) and 3 <= len(family["family"]) <= 6
    assert "Evolutionary conservation" in family["family"] and all(round(v, 3) for v in family["contrib"])   # zero-net families are not drawn
    spec = json.loads(charts[1].proto.spec)
    assert spec["height"] < json.loads(charts[2].proto.spec)["height"] / 2  # a small chart, not a second subject
    assert {b.key for b in at.button if b.key in ("pick_prev", "pick_next")} == {"pick_prev", "pick_next"}
    css = text(at)
    assert ".st-key-pick_step {flex-wrap:nowrap; gap:0}" in css and "st.columns([6, 1]" not in APP.read_text()
    # The mark is a flex item of its label, centred in a square button, not an inline glyph on a zero-height baseline.
    assert "button p {font-size:0; line-height:0; display:flex; align-items:center; justify-content:center}" in css
    assert ".st-key-pick_step button {min-height:2.5rem; width:2.5rem; padding:0;" in css
    assert ".st-key-review_head [data-testid=\"stSelectbox\"] div:has(> [role=\"combobox\"]) {background:var(--surf)" in css
    mark = '.tall table.grid tr[data-vid]:is(:hover,:has(button.row:focus-visible)) td:last-child::after {content:"manage_search"'
    assert mark in css and "position:absolute; right:0.5rem; top:50%; transform:translateY(-50%)" in css


def test_score_agreement_is_not_amino_acid_chemistry():
    at = run(panel_keep="cancer", lang="en")
    assert "Mean of selected, scaled features" in chart_text(at)   # the top-10 chart, not "Mean of anchor scores"
    value = text(at)
    assert "Other" in value and "Mean and spread of scaled features (score agreement)" in value   # a mean, a count and a spread
    assert "Seçilmiş ölçeklenmiş özellik ortalaması" in chart_text(run(panel_keep="cancer"))


def test_upload_overlap_and_missing_ids_are_not_validation():
    df = sample()
    df["Variant_ID"] = [f"new-{i}" for i in range(len(df))]
    at = upload(csv(df), lang="en")
    value = text(at)
    assert "test.csv" in plain(at) and "25 variants" not in plain(at) and "ROC AUC" in metrics(at)
    assert "0/25 IDs in the training set" in chips(at)
    # Deleting one column is what defeats the overlap check, and it used to buy a full set of metrics with the
    # caveat in a tooltip: F1 0.897 · AUC 0.986 on rows whose provenance the app cannot check, for a panel
    # whose published AUC is 0.864. Everywhere else in the app an aggregate the model may have been fitted on
    # is withheld rather than captioned, so this path withholds it too — the cards, and the confusion counts
    # they can be rebuilt from. Both headings still carry the fact, so a crop of either one carries it.
    at = upload(csv(df.drop(columns="Variant_ID")), lang="en")
    value, line = text(at), plain(at)
    assert "Label comparison" in line and "not validation" in chips(at)
    assert "test.csv" in line and "overlap unknown" in chips(at)
    assert "cannot be checked against the training set" in value and "Metrics are withheld" in value
    assert not any(m.label in ("ROC AUC", "MCC") or m.label.startswith("F1") for m in at.metric), metrics(at)
    assert "TP " not in value and "cross-validated F1" not in value
    # Withholding without a way out would be a dead end: the recovery is the column whose absence caused it.
    assert "Add a Variant_ID column and upload the file again." in hints(at)
    # The rows themselves are untouched: nothing about a single variant depends on the batch's provenance.
    assert "class='tablewrap tall'" in value and "True label" in value and value.count("data-vid=") == 25
    assert at.download_button   # ... and so is the export of them
    assert "row 1" in value and "row 0" not in value   # an operator matching a spreadsheet counts from 1
    tr_at = upload(csv(df.drop(columns="Variant_ID")))
    tr = plain(tr_at)
    # The overlap-unknown fact is the heading's chip; the label-comparison line states only what it qualifies.
    assert "Etiket karşılaştırması" in tr and "doğrulama değil" in chips(tr_at)
    assert "örtüşme bilinmiyor" in chips(tr_at) and "satır 1" in tr


def test_an_unlabelled_batch_says_whether_its_rows_are_in_the_training_set():
    """The finals batch: novel variants, no Label column, so no metric on the page. The app runs the ID
    comparison on every batch and used to throw this one away — the heading was bare. It is the strongest
    honest claim the project can make and it belongs on the one screen the jury will be looking at."""
    df = sample().drop(columns="Label")
    novel = df.assign(Variant_ID=[f"new-{i}" for i in range(len(df))])
    at = upload(csv(novel), lang="en")
    assert "test.csv" in plain(at) and "not in the training set" in chips(at)
    # The claim is checked-against-our-training-IDs, not independence, and it is not the labelled batch's
    # note: there are no labels here for "these are not validation results" to be about.
    assert "a non-matching ID is not on its own evidence of independence" in hints(at)
    assert "not validation results" not in text(at) and "Label comparison" not in text(at)
    assert not any(m.label.startswith("F1") for m in at.metric)
    half = pd.concat([df.iloc[:12], novel.iloc[12:]])   # twelve training IDs among thirteen new ones
    assert "12/25 IDs in the training set" in chips(upload(csv(half), lang="en"))
    tr = upload(csv(novel))
    assert "eğitim kümesinde yok" in chips(tr)
    assert "tek başına bağımsızlık kanıtı değildir" in hints(tr)


def test_the_file_remembers_which_panel_it_was_uploaded_for():
    """A file is declared for whatever the panel control says when its bytes are taken — the control in the
    dialog that takes them, which is the header band's own, drawn there while the dialog is open. Switching
    it afterwards rescores the same bytes with another panel's model, which is the operator error that
    produces a confident wrong answer. The declaration rides the file and the heading says when the two
    disagree. It is memory, not a second panel control: only a new file re-declares."""
    payload = csv(sample().drop(columns="Label"))
    keep = ("general_batch.csv", payload, "general")
    assert "uploaded for" not in text(run(page="predictions", upload_keep=keep, lang="en"))
    at = run(page="predictions", upload_keep=keep, panel_keep="cftr", lang="en")
    assert "general_batch.csv" in plain(at) and chips(at)[0] == "file was uploaded for General"
    assert "dosya Genel paneli için yüklendi" in chips(run(page="predictions", upload_keep=keep, panel_keep="cftr"))
    # Driven in Turkish: the dialog's segments carry the full panel names, and AppTest formats a value with
    # the launch language's dictionary, so an English run cannot address an option by its English label.
    at = run(page="predictions", upload_open=True, upload_keep=keep)
    next(s for s in at.segmented_control if s.key == "panel_w").set_value("cftr").run()
    assert at.session_state["panel_keep"] == "cftr"                      # the one control, in the dialog while it is open
    assert at.session_state["upload_keep"][2] == "general"               # ... and the declaration is the file's, unchanged
    assert "dosya Genel paneli için yüklendi" in plain(at)
    # The uploader itself writes the third element; AppTest cannot hand a widget a file, so that path is
    # browser-verified (an upload declared for General, then the header switched to CFTR).
    assert 'st.session_state["upload_keep"] = (up.name, up.getvalue(), st.session_state.get("panel_keep", "general"))' in APP.read_text()


def test_the_batch_prevalence_f1_is_not_drawn():
    """An F1 at the batch's own prevalence was the largest number on this page — 0.880 on PAH, 0.916 on CFTR —
    at a prevalence over 80% pathogenic that the product will never meet, on the rows the threshold and the
    Platt calibration were fitted on. What is left is the same F1 rescaled to the test prior and the two
    metrics that do not depend on a prior at all."""
    for panel, labels in (("pah", ["F1 (test prevalence 28.6%)", "ROC AUC", "MCC"]), ("cftr", ["ROC AUC", "MCC"])):
        at = run(page="predictions", panel_keep=panel, lang="en")
        assert [m.label for m in at.metric] == labels, panel   # the count cards are the app's own, not st.metric
        assert "batch prevalence" not in text(at) and "küme prevalansı" not in text(at), panel
    assert "küme prevalansı" not in text(run(page="predictions", panel_keep="pah"))


def test_a_failed_file_can_be_removed_from_the_review_page():
    """The variant page carries no upload control — except while the loaded file is one that could not be
    scored, which would otherwise be removable only from the page it already failed on."""
    at = run(page="predictions", upload_keep=("bad.csv", b"a,b\n1,x\n", "general"))
    assert at.error
    on(at, "explain")
    assert at.error and next(b for b in at.button if b.key == "upload_file").label == "bad.csv"
    on(at, "predictions")
    next(b for b in at.button if b.key == "upload_file").click().run()   # the dialog it opens carries Remove
    next(b for b in at.button if b.key == "remove_upload").click().run()
    on(at, "explain")
    assert not at.error and not [b for b in at.button if b.key in ("upload_btn", "upload_file")]


def test_a_non_standard_substitution_draws_no_amino_acid_row():
    """The results table leaves the Change cell blank for a multi-residue or non-standard code; the review page
    said the same absence four times over ("missing → missing · Grantham missing · BLOSUM62 missing"). It draws
    nothing either — the ⇄ on the row and in the flag list is what says why."""
    df = pd.read_csv(PANELS["general"])
    std = set("ACDEFGHIKLMNPQRSTVWY")
    odd = df[~(df["AA_1"].astype(object).isin(std) & df["AA_2"].astype(object).isin(std))].iloc[0]
    batch = pd.concat([odd.to_frame().T, df.head(24)])
    at = run(page="explain", upload_keep=("odd.csv", csv(batch), "general"),
             pick_keep={"general": str(odd["Variant_ID"])}, lang="en")
    value = text(at)
    assert str(odd["Variant_ID"]) in value and "non-standard AA code" in value   # the rule, named once
    assert "Amino-acid change" not in value and "missing → missing" not in value


def test_the_three_answers_ride_the_tooltip_of_what_they_qualify():
    """Three facts a jury asks for in the first minutes, each in the help of the exact element it is about and
    nowhere on the line: how these numbers were validated, why most columns carry an id instead of a name, and
    what the Platt step does."""
    for lang, protocol, columns, platt in (
            ("en", "refitted inside every fold", "anonymised by the competition", "rescales the score to a probability"),
            ("tr", "her katta yeniden belirlenir", "anonimleştirilmiştir", "skoru olasılığa çevirir")):
        overview = run(lang=lang)
        assert protocol in hints(overview) and protocol not in plain(overview), lang
        review = run(page="explain", lang=lang)
        assert columns in hints(review) and columns not in plain(review), lang
        flow = text(run(panel_keep="pah", lang=lang))   # the → Platt node of the scoring flow carries it
        assert platt in flow and platt not in plain(run(panel_keep="pah", lang=lang)), lang


def test_upload_revisit_and_remove_returns_to_the_demo_sample():
    payload = csv(sample())
    at = run(page="predictions", upload_keep=("my_batch_v2.csv", payload, "general"))
    on(at, "overview"); on(at, "explain"); on(at, "predictions")
    assert at.session_state["upload_keep"] == ("my_batch_v2.csv", payload, "general")
    assert "my_batch_v2.csv" in plain(at) and "25 varyant" not in plain(at)
    # The file is named twice, verbatim (no markdown emphasis): as the batch heading, and as the label of
    # the header button that is the whole upload control — which is how the header says a file is loaded.
    assert text(at).count("my_batch_v2.csv") == 1
    assert next(b for b in at.button if b.key == "upload_file").label == "my_batch_v2.csv"
    at.session_state["upload_open"] = True   # Remove lives in the dialog, beside the name of the file it drops
    on(at, "predictions")
    next(b for b in at.button if b.key == "remove_upload").click().run()
    value = text(at)   # removing the file returns to the demo sample, never to an empty state
    assert "Eğitim kümesinden örneklem" in plain(at) and chips(at) == ["örneklem içi"]
    assert cards(at) and "CSV girdi gereksinimleri" not in value
    assert next(b for b in at.button if b.key == "upload_btn").label == "Dosya seç"   # ... and the header with it


def test_the_demo_sample_is_fixed_and_deterministic():
    """No sample-size control: 300 rows, seed 42, the same rows on every run and on every page."""
    expected = pd.read_csv(PANELS["general"]).sample(n=300, random_state=42).reset_index(drop=True)
    at = run(page="predictions", lang="en")
    ids = table_ids(at)
    assert len(ids) == 300 and set(ids) == set(expected["Variant_ID"].astype(str))
    assert ids == table_ids(run(page="predictions", lang="en"))


def test_single_class_and_text_labels():
    df = sample(); df["Label"] = 1; df["Variant_ID"] = [f"new-{i}" for i in range(len(df))]
    at = upload(csv(df), lang="en")
    assert not at.metric and "Single-class batch" in text(at)         # AUC, MCC and F1 are undefined here
    assert "MCC" not in metrics(at) and all(v != "—" for v, _, _ in cards(at).values())
    assert cards(at)["Called pathogenic"][1] == "25 labelled"          # a batch is described whether or not it can be scored against
    df["Label"] = "Pathogenic"
    at = upload(csv(df), lang="en")
    assert set(cards(at)) == {"Called pathogenic", "Warning types"} and not at.metric   # ... but a Label column the app rejected counts nothing
    assert "Label comparison" not in text(at)
    # A caveat about the batch rides the batch heading, as every other one does: a tag on the line, the
    # sentence and the offending value in its tooltip. It was a notice box in the framework's blue — the
    # colour that on these pages means a benign call — and with it gone stAlert only ever paints an error.
    assert not at.info and "Label column ignored" in chips(at)
    assert "must be 0 or 1" in hints(at) and "Pathogenic" in hints(at)   # never silently dropped: the rule and the offending value


def test_a_rejected_upload_still_states_the_csv_requirements():
    """The requirements guidance has no waiting state left to live in, so it must survive where it is
    actionable: on a file that could not be scored (and, before the fact, in the uploader's own help)."""
    for payload in (csv(sample(0)), b"a,b\n1,x\n2,y\n", b'"unterminated', b""):
        at = upload(payload)
        assert at.error and "skorlanamadı" in at.error[0].value.lower()
        assert not any(x in at.error[0].value for x in ("KeyError", "Index(", "Traceback", "ParserError"))
        value = text(at)
        assert "CSV girdi gereksinimleri" in value and "Variant_ID" in value
        assert "0 = Benign, 1 = Patojenik" in value   # the label column says which class is which
        assert "AL_1, AL_2, AL_3" in value            # examples sorted numerically, not AL_1, AL_10, AL_100
    at = upload(csv(sample(0)))   # a header with no rows
    assert "en az bir varyant satırı içeren bir CSV yükleyin" in at.error[0].value
    # Both numbers, not the eight names alone: "zorunlu 8 model girdisi eksik" with 294 absent read as
    # "add eight columns", and the operator's file is missing nearly every input the model takes.
    err = upload(b"a,b\n1,x\n").error[0].value
    total, miss, required = map(int, re.search(r"(\d+) model girdisinin (\d+) tanesi eksik, (\d+) tanesi zorunlu", err).groups())
    assert total > 200 and miss == total and required == 8 and "CAT_1" in err
    value = upload(b"a,b\n1,x\n", panel_keep="cftr").error[0].value  # transfer-only panel: mandatory groups before tolerated AL_
    assert "AA_1, AA_2, CAT_1" in value and "AL_" not in value and "0 tanesi zorunlu" in value


def test_supported_missing_numeric_columns_remain_supported():
    df = sample().drop(columns=["EK_7", "AL_1", "Variant_ID"])
    for panel in PANELS:
        at = upload(csv(df), panel_keep=panel)
        assert not at.error and not at.warning and "test.csv" in plain(at)
    at = upload(csv(df.drop(columns=[f"AL_{i}" for i in range(2, 80) if f"AL_{i}" in df])))
    assert not at.error and "test.csv" in plain(at)
    assert "yalnızca %7" in at.warning[0].value  # coverage below 0.9 warns but still scores
    assert any(c.startswith("özellik kapsamı %7") for c in chips(at))   # ... and it rides the heading, so a crop of the batch carries it too


def test_filter_scopes_table_and_review_and_empty_results():
    at = run(page="predictions", filter_keep=("flag_borderline",), lang="en")
    assert "40 / 300 rows · Filter: borderline" in text(at)
    on(at, "explain")
    assert "40/300 variants ·" in text(at) and "Filter: borderline" in text(at)
    at = upload(csv(sample().assign(Label=0)), filter_keep=("benign", "wrong"), lang="en")
    # Nothing matches: the table is drawn empty rather than replaced, so the page keeps its shape and the
    # filter above it stays the recovery. The review page has no table and no filter of its own, so it says
    # the same sentence in the app's own muted line and keeps the way back. Neither draws a framework notice.
    assert not at.info and "class='tablewrap tall'" in text(at) and "No variants match" in text(at)
    on(at, "explain")
    assert not at.info and "<div class='small'>No variants match this filter.</div>" in text(at)
    assert "Predictions" in text(at)   # the way back is the recovery


def test_the_brand_teal_is_the_interface_colour_never_a_call():
    """One accent, written once: the theme's primaryColor is app.BRAND, the CSS reads it as --brand for the
    selection and focus roles only, and the two call colours stay RED and BLUE on every chart, bar and tag.
    The brand must never read as a third class of call, so no Vega spec draws it."""
    colours = {}
    for node in ast.parse(APP.read_text()).body:   # the palette lines: `NAME = "#hex"` and `A, B = "#hex", "#hex"`
        if isinstance(node, ast.Assign):
            for target in node.targets:
                names = target.elts if isinstance(target, ast.Tuple) else [target]
                values = node.value.elts if isinstance(node.value, ast.Tuple) else [node.value]
                colours.update({n.id: v.value for n, v in zip(names, values) if isinstance(n, ast.Name) and isinstance(v, ast.Constant)})
    colours = {k: colours[k] for k in ("BRAND", "RED", "BLUE", "INK", "SURF")}   # each named once, at module level
    theme = (ROOT / ".streamlit/config.toml").read_text()
    # The three values the framework paints with are the app's own, or the same screen carries two inks and
    # two surfaces: the metric values against the table cells, the filter's hover against every well.
    for key, name in (("primaryColor", "BRAND"), ("textColor", "INK"), ("secondaryBackgroundColor", "SURF")):
        assert re.search(rf'^{key} = "{colours[name]}"', theme, re.M), f"config.toml {key} is not app.{name}"
    assert colours["BRAND"] not in (colours["RED"], colours["BLUE"])
    for page in ("overview", "predictions", "explain"):
        at = run(page=page, panel_keep="cancer", lang="en")
        value = text(at)
        assert f"--brand:{colours['BRAND']}" in value
        for role in ("[aria-current=\"page\"] {background:none; color:var(--ink); font-weight:600; box-shadow:inset 0 -2px var(--brand)}",
                     "button[aria-checked=\"true\"]:is(:hover,:focus,:focus-visible,:active)\n    {background:var(--brand); color:#fff}",
                     "tr.now td:first-child {box-shadow:inset 3px 0 var(--brand)}", ".tablewrap:focus-visible {outline:2px solid var(--brand)"):
            assert role in value, (page, role)
        specs = [json.loads(e.proto.spec) for e in at.get("vega_lite_chart")]
        assert specs, page
        assert not any(colours["BRAND"].lower() in json.dumps(s).lower() for s in specs), page   # no chart mark is teal
        # the calls keep their colours: the score bars and tags in the markup, the histogram's legend in its spec
        if page == "predictions":
            assert f"--bar:{colours['RED']}" in value and f"--bar:{colours['BLUE']}" in value
            hist = next(s for s in specs if "layer" in s and any("rule" in json.dumps(l) for l in s["layer"]))
            assert json.dumps(hist).count(colours["RED"]) and json.dumps(hist).count(colours["BLUE"]), "histogram lost its call colours"
        if page == "explain":
            assert re.search(r"class='tag' style='background:(%s|%s)'" % (colours["RED"], colours["BLUE"]), value)

def test_a_page_update_is_not_split_by_the_post_script_collection():
    """Streamlit's forced generation-2 collection after every script run takes 90-125 ms on this app's
    working set and holds the GIL, so the last elements of a page — the contribution chart, the results
    table — used to reach the browser a stage after the rest. The line that turns it off is invisible in
    every screenshot and would be easy to tidy away, so it is pinned here with the reason."""
    config = (ROOT / ".streamlit/config.toml").read_text()
    assert re.search(r"^postScriptGC = false", config, re.M), "config.toml no longer disables the post-script GC"


def test_canonical_sources_and_chart_math():
    # Extract data/functions without importing the Streamlit entrypoint.
    tree = ast.parse(APP.read_text())
    canon_node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CANON" for t in n.targets))
    canon = eval(compile(ast.Expression(canon_node.value), str(APP), "eval"), {"dict": dict})
    paths = [ROOT / "analysis/output" / n for n in ("PDR_CANONICAL_RESULTS.csv", "panel_cm_rates.csv")]
    if all(p.exists() for p in paths):
        table, rates = [pd.read_csv(p).set_index("panel") for p in paths]
        for panel, values in canon.items():
            for key, column in {"F1":"F1", "SE":"F1_SE", "AUC":"AUC", "MCC":"MCC", "PR":"PR_AUC_at_prior", "thr":"threshold"}.items():
                # Source column names are inspected below; no inferred numerical tolerance beyond rounding.
                assert abs(values[key] - float(table.loc[panel, column])) < .00051
            for key in ("TPR", "FPR"):
                assert abs(values[key] - float(rates.loc[panel, key])) < .000051
    else:
        print("  SKIP canonical CSV equality: private source CSVs absent")
    for panel in PANELS:
        art = load_artifacts(panel)
        assert canon[panel]["train"] == art["training_prior"]
        assert canon[panel]["thr"] == round(art["threshold"], 2) == art["threshold"]
        # ... and the marked row of the Overview table is the loaded artifact's own prior and threshold,
        # so a retrain that moved either cannot silently desync the table from the rest of the app.
        marked = re.search(r"<tr class='now'>(.*?)</tr>", text(run(panel_keep=panel, lang="en")), re.S)[1]
        assert f"{art['training_prior']:.1%}" in marked and f"{art['threshold']:.2f}" in marked, (panel, marked)
        scored = score_frame(pd.read_csv(PANELS[panel], nrows=2), art)
        ex = explain_frame(art, scored)
        row = ex["contribs"].iloc[0]; top = top_contributions(row, 10)
        tail = row.sum() - top.sum()
        expected = sum(w * (base[0] + contrib.iloc[0].sum()) for _, w, contrib, base in ex["members"])
        np.testing.assert_allclose(ex["base"][0] + top.sum() + tail, expected, atol=1e-9)


def main():
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn(); print("  PASS", name, flush=True)
            except Exception:
                failures += 1; print("  FAIL", name, flush=True); traceback.print_exc()
    print("ALL PASS" if not failures else f"{failures} FAILURE(S)")
    return int(bool(failures))


if __name__ == "__main__":
    sys.exit(main())
