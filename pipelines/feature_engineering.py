"""Feature engineering for amino-acid substitutions.

Adds biochemistry-derived columns from AA_1 (reference AA) and AA_2 (alternate
AA). All scales/matrices are public scientific constants from standard
biochemistry references; no external data dependency.

Engineered features added by engineer_aa_features(df):

  AA_blosum62        BLOSUM62 substitution score (int, -4 to +11)
  AA_grantham        Grantham distance (continuous, 5-215, higher = more change)
  AA_hydro_delta     |Kyte-Doolittle hydrophobicity_2 - hydrophobicity_1|
  AA_polarity_delta  |Zimmerman polarity_2 - polarity_1|
  AA_mw_delta        |molecular weight_2 - molecular weight_1| (Daltons)
  AA_volume_delta    |residue volume_2 - volume_1| (cubic angstroms)
  AA_charge_change   charge_2 - charge_1 (signed: -2, -1, 0, +1, +2)
  AA_to_stop         1 if AA_2 == "*" else 0   (nonsense mutation flag)
  AA_glycine_loss    1 if AA_1 == "G" and AA_2 != "G" else 0
  AA_proline_intro   1 if AA_1 != "P" and AA_2 == "P" else 0
  AA_cysteine_loss   1 if AA_1 == "C" and AA_2 != "C" else 0  (disulfide disruption)
  AA_proline_loss    1 if AA_1 == "P" and AA_2 != "P" else 0  (turn loss)
  AA_is_multiletter  1 if either AA_1 or AA_2 is a multi-letter code (AG, SV, etc.)
  AA_same            1 if AA_1 == AA_2 else 0    (synonymous flag — usually 0 for missense)

For rows where either AA_1 or AA_2 is non-standard (multi-letter, NaN), the
numeric scores fall back to NaN. The boolean flags (AA_to_stop, AA_is_multiletter
etc.) always have a defined value.

The function is in-place safe — it adds columns to a copy of the input.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Standard amino-acid property tables
# ---------------------------------------------------------------------------

# Standard one-letter codes in fixed order. "*" is stop.
_STD_AA = "ARNDCQEGHILKMFPSTWYV"
_STD_AA_SET = set(_STD_AA) | {"*"}

# Kyte-Doolittle hydrophobicity (Kyte & Doolittle 1982, J. Mol. Biol.)
HYDROPHOBICITY_KD = {
    "A":  1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C":  2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I":  4.5,
    "L":  3.8, "K": -3.9, "M":  1.9, "F":  2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V":  4.2,
    "*":  0.0,
}

# Zimmerman polarity (Zimmerman, Eliezer, Simha 1968)
POLARITY = {
    "A":   0.00, "R":  52.00, "N":  3.38, "D":  49.70, "C":  1.48,
    "Q":   3.53, "E":  49.90, "G":  0.00, "H":  51.60, "I":  0.13,
    "L":   0.13, "K":  49.50, "M":  1.43, "F":  0.35,  "P": 1.58,
    "S":   1.67, "T":  1.66,  "W":  2.10, "Y":  1.61,  "V":  0.13,
    "*":   0.0,
}

# Molecular weight (Daltons) of free amino acids
MW = {
    "A":  89.09, "R": 174.20, "N": 132.12, "D": 133.10, "C": 121.16,
    "Q": 146.15, "E": 147.13, "G":  75.07, "H": 155.16, "I": 131.17,
    "L": 131.17, "K": 146.19, "M": 149.21, "F": 165.19, "P": 115.13,
    "S": 105.09, "T": 119.12, "W": 204.23, "Y": 181.19, "V": 117.15,
    "*":   0.0,
}

# Residue volume in cubic angstroms (Pontius, Richelle, Wodak 1996)
VOLUME = {
    "A":  88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5,
    "Q": 143.8, "E": 138.4, "G":  60.1, "H": 153.2, "I": 166.7,
    "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9, "P": 112.7,
    "S":  89.0, "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0,
    "*":   0.0,
}

# Net charge at physiological pH (~7.4)
CHARGE = {
    "A":  0, "R": +1, "N":  0, "D": -1, "C":  0,
    "Q":  0, "E": -1, "G":  0, "H":  0, "I":  0,   # H is partial; treat as 0
    "L":  0, "K": +1, "M":  0, "F":  0, "P":  0,
    "S":  0, "T":  0, "W":  0, "Y":  0, "V":  0,
    "*":  0,
}

# BLOSUM62 substitution scores (Henikoff & Henikoff 1992).
# Ordering follows _BLOSUM_AA below.
_BLOSUM_AA = "ARNDCQEGHILKMFPSTWYV"
_BLOSUM_MATRIX = np.array([
    [ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0],
    [-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3],
    [-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3],
    [-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3],
    [ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1],
    [-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2],
    [-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2],
    [ 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3],
    [-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3],
    [-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3],
    [-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1],
    [-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2],
    [-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1],
    [-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1],
    [-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2],
    [ 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2],
    [ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0],
    [-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3],
    [-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1],
    [ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4],
], dtype=np.int8)
_BLOSUM_IDX = {aa: i for i, aa in enumerate(_BLOSUM_AA)}


# Grantham distance matrix (Grantham 1974, Science).
# Symmetric (d[a][b] == d[b][a]); diagonal is 0.
# Standard published table — full 20x20 below.
GRANTHAM = {
    "A": {"A":0,"R":112,"N":111,"D":126,"C":195,"Q":91,"E":107,"G":60,"H":86,"I":94,"L":96,"K":106,"M":84,"F":113,"P":27,"S":99,"T":58,"W":148,"Y":112,"V":64},
    "R": {"A":112,"R":0,"N":86,"D":96,"C":180,"Q":43,"E":54,"G":125,"H":29,"I":97,"L":102,"K":26,"M":91,"F":97,"P":103,"S":110,"T":71,"W":101,"Y":77,"V":96},
    "N": {"A":111,"R":86,"N":0,"D":23,"C":139,"Q":46,"E":42,"G":80,"H":68,"I":149,"L":153,"K":94,"M":142,"F":158,"P":91,"S":46,"T":65,"W":174,"Y":143,"V":133},
    "D": {"A":126,"R":96,"N":23,"D":0,"C":154,"Q":61,"E":45,"G":94,"H":81,"I":168,"L":172,"K":101,"M":160,"F":177,"P":108,"S":65,"T":85,"W":181,"Y":160,"V":152},
    "C": {"A":195,"R":180,"N":139,"D":154,"C":0,"Q":154,"E":170,"G":159,"H":174,"I":198,"L":198,"K":202,"M":196,"F":205,"P":169,"S":112,"T":149,"W":215,"Y":194,"V":192},
    "Q": {"A":91,"R":43,"N":46,"D":61,"C":154,"Q":0,"E":29,"G":87,"H":24,"I":109,"L":113,"K":53,"M":101,"F":116,"P":76,"S":68,"T":42,"W":130,"Y":99,"V":96},
    "E": {"A":107,"R":54,"N":42,"D":45,"C":170,"Q":29,"E":0,"G":98,"H":40,"I":134,"L":138,"K":56,"M":126,"F":140,"P":93,"S":80,"T":65,"W":152,"Y":122,"V":121},
    "G": {"A":60,"R":125,"N":80,"D":94,"C":159,"Q":87,"E":98,"G":0,"H":98,"I":135,"L":138,"K":127,"M":127,"F":153,"P":42,"S":56,"T":59,"W":184,"Y":147,"V":109},
    "H": {"A":86,"R":29,"N":68,"D":81,"C":174,"Q":24,"E":40,"G":98,"H":0,"I":94,"L":99,"K":32,"M":87,"F":100,"P":77,"S":89,"T":47,"W":115,"Y":83,"V":84},
    "I": {"A":94,"R":97,"N":149,"D":168,"C":198,"Q":109,"E":134,"G":135,"H":94,"I":0,"L":5,"K":102,"M":10,"F":21,"P":95,"S":142,"T":89,"W":61,"Y":33,"V":29},
    "L": {"A":96,"R":102,"N":153,"D":172,"C":198,"Q":113,"E":138,"G":138,"H":99,"I":5,"L":0,"K":107,"M":15,"F":22,"P":98,"S":145,"T":92,"W":61,"Y":36,"V":32},
    "K": {"A":106,"R":26,"N":94,"D":101,"C":202,"Q":53,"E":56,"G":127,"H":32,"I":102,"L":107,"K":0,"M":95,"F":102,"P":103,"S":121,"T":78,"W":110,"Y":85,"V":97},
    "M": {"A":84,"R":91,"N":142,"D":160,"C":196,"Q":101,"E":126,"G":127,"H":87,"I":10,"L":15,"K":95,"M":0,"F":28,"P":87,"S":135,"T":81,"W":67,"Y":36,"V":21},
    "F": {"A":113,"R":97,"N":158,"D":177,"C":205,"Q":116,"E":140,"G":153,"H":100,"I":21,"L":22,"K":102,"M":28,"F":0,"P":114,"S":155,"T":103,"W":40,"Y":22,"V":50},
    "P": {"A":27,"R":103,"N":91,"D":108,"C":169,"Q":76,"E":93,"G":42,"H":77,"I":95,"L":98,"K":103,"M":87,"F":114,"P":0,"S":74,"T":38,"W":147,"Y":110,"V":68},
    "S": {"A":99,"R":110,"N":46,"D":65,"C":112,"Q":68,"E":80,"G":56,"H":89,"I":142,"L":145,"K":121,"M":135,"F":155,"P":74,"S":0,"T":58,"W":177,"Y":144,"V":124},
    "T": {"A":58,"R":71,"N":65,"D":85,"C":149,"Q":42,"E":65,"G":59,"H":47,"I":89,"L":92,"K":78,"M":81,"F":103,"P":38,"S":58,"T":0,"W":128,"Y":92,"V":69},
    "W": {"A":148,"R":101,"N":174,"D":181,"C":215,"Q":130,"E":152,"G":184,"H":115,"I":61,"L":61,"K":110,"M":67,"F":40,"P":147,"S":177,"T":128,"W":0,"Y":37,"V":88},
    "Y": {"A":112,"R":77,"N":143,"D":160,"C":194,"Q":99,"E":122,"G":147,"H":83,"I":33,"L":36,"K":85,"M":36,"F":22,"P":110,"S":144,"T":92,"W":37,"Y":0,"V":55},
    "V": {"A":64,"R":96,"N":133,"D":152,"C":192,"Q":96,"E":121,"G":109,"H":84,"I":29,"L":32,"K":97,"M":21,"F":50,"P":68,"S":124,"T":69,"W":88,"Y":55,"V":0},
}


# ---------------------------------------------------------------------------
# Per-pair lookup helpers
# ---------------------------------------------------------------------------

def blosum62(ref: str, alt: str) -> float:
    if not isinstance(ref, str) or not isinstance(alt, str):
        return np.nan
    i = _BLOSUM_IDX.get(ref); j = _BLOSUM_IDX.get(alt)
    if i is None or j is None:
        return np.nan
    return float(_BLOSUM_MATRIX[i, j])


def grantham(ref: str, alt: str) -> float:
    if ref in GRANTHAM and alt in GRANTHAM[ref]:
        return float(GRANTHAM[ref][alt])
    return np.nan


def _delta(ref: str, alt: str, table: dict, signed: bool = False) -> float:
    if ref not in table or alt not in table:
        return np.nan
    d = table[alt] - table[ref]
    return float(d if signed else abs(d))


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def engineer_aa_features(df: pd.DataFrame,
                         ref_col: str = "AA_1",
                         alt_col: str = "AA_2",
                         inplace: bool = False) -> pd.DataFrame:
    """Add biochemistry-derived columns from (ref_col, alt_col).

    Returns the (modified copy of) df with the new columns appended.
    Standard usage:
        df = engineer_aa_features(df)
    Idempotent: if a column already exists it's overwritten.
    """
    if not inplace:
        df = df.copy()

    # Cast categoricals to object so element-wise lookups work
    aa1 = df[ref_col].astype(object) if ref_col in df.columns else pd.Series([np.nan] * len(df))
    aa2 = df[alt_col].astype(object) if alt_col in df.columns else pd.Series([np.nan] * len(df))

    df["AA_blosum62"]      = [blosum62(r, a) for r, a in zip(aa1, aa2)]
    df["AA_grantham"]      = [grantham(r, a) for r, a in zip(aa1, aa2)]
    df["AA_hydro_delta"]   = [_delta(r, a, HYDROPHOBICITY_KD) for r, a in zip(aa1, aa2)]
    df["AA_polarity_delta"]= [_delta(r, a, POLARITY)          for r, a in zip(aa1, aa2)]
    df["AA_mw_delta"]      = [_delta(r, a, MW)                for r, a in zip(aa1, aa2)]
    df["AA_volume_delta"]  = [_delta(r, a, VOLUME)            for r, a in zip(aa1, aa2)]
    df["AA_charge_change"] = [_delta(r, a, CHARGE, signed=True) for r, a in zip(aa1, aa2)]

    # Boolean / flag features (always defined, NaN-safe)
    def _eq(s: pd.Series, value: str) -> pd.Series:
        return (s.astype(object) == value).astype(int)

    def _bothstd(s1: pd.Series, s2: pd.Series) -> pd.Series:
        return s1.astype(object).isin(_STD_AA_SET) & s2.astype(object).isin(_STD_AA_SET)

    df["AA_to_stop"]       = _eq(aa2, "*")
    df["AA_glycine_loss"]  = (_eq(aa1, "G") & ~_eq(aa2, "G")).astype(int)
    df["AA_proline_intro"] = (~_eq(aa1, "P") & _eq(aa2, "P")).astype(int)
    df["AA_proline_loss"]  = (_eq(aa1, "P") & ~_eq(aa2, "P")).astype(int)
    df["AA_cysteine_loss"] = (_eq(aa1, "C") & ~_eq(aa2, "C")).astype(int)
    df["AA_same"]          = (aa1.astype(object) == aa2.astype(object)).astype(int)
    df["AA_is_multiletter"] = (~_bothstd(aa1, aa2)).astype(int)

    # Substitution-radicality flags (derived from BLOSUM62).
    # BLOSUM >= 0 = conservative substitution (protein likely preserved).
    # BLOSUM <= -3 = radical (major chemical disruption).
    # These hard-thresholded flags help tree splits find these regimes faster.
    blos = pd.Series(df["AA_blosum62"], index=df.index)
    df["AA_is_conservative"] = ((blos >= 0) & blos.notna()).astype(int)
    df["AA_is_radical"]      = ((blos <= -3) & blos.notna()).astype(int)
    # Grantham > 100 = moderate-to-large physicochemical change
    gran = pd.Series(df["AA_grantham"], index=df.index)
    df["AA_grantham_high"]   = ((gran >= 100) & gran.notna()).astype(int)
    # Charge flip (e.g., R→D or K→E) = serious electrostatic change
    df["AA_charge_flipped"]  = (df["AA_charge_change"].abs() >= 2).astype(int)

    return df


def engineer_blosum_extras(df: pd.DataFrame,
                           ref_col: str = "AA_1", alt_col: str = "AA_2",
                           inplace: bool = False) -> pd.DataFrame:
    """Extra BLOSUM62-derived columns from Almira's experiment, restricted to the
    pieces engineer_aa_features does NOT already provide.

    engineer_aa_features already has AA_blosum62 (the score), AA_is_conservative
    (>=0), AA_is_radical (<=-3), AA_same and AA_is_multiletter — so the raw score,
    aa_same and the unknown flag would be duplicates and are intentionally omitted.
    What is genuinely new: the |score| magnitude and an explicit sign trichotomy
    (negative / zero / positive), which split the score differently than the
    conservative/radical thresholds.

    Leakage-safe: derived only from AA_1/AA_2 (no labels), so it reproduces on the
    judges' test inputs.

    STATUS: tested and NOT adopted. analysis/feature_ablation.py (honest test-prior,
    repeated 10-fold CV, 2026-06-07) showed no F1* gain on any panel — every delta
    vs the deployed AA features was within noise, and mRMR selected one of these
    flags only once (cancer: AA_blosum_zero_flag, +0.006, noise). The signal is
    already captured by AA_blosum62 in engineer_aa_features. Kept for reference and
    reproducibility; deliberately NOT wired into run_pipeline/predict. If re-enabled
    later, mirror it in predict.apply_preprocessing (it is stateless, so that is the
    only change needed for test purity).
    """
    if not inplace:
        df = df.copy()
    aa1 = df[ref_col].astype(object) if ref_col in df.columns else pd.Series([np.nan] * len(df))
    aa2 = df[alt_col].astype(object) if alt_col in df.columns else pd.Series([np.nan] * len(df))
    b = pd.Series([blosum62(r, a) for r, a in zip(aa1, aa2)], index=df.index)
    df["AA_blosum_abs"]       = b.abs()
    df["AA_blosum_neg_flag"]  = ((b < 0) & b.notna()).astype(int)
    df["AA_blosum_zero_flag"] = ((b == 0) & b.notna()).astype(int)
    df["AA_blosum_pos_flag"]  = ((b > 0) & b.notna()).astype(int)
    return df


def engineer_acmg_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ACMG/AMP-inspired domain features (enabled per-panel via ACMG_PANELS).

    Grounded in the verified column-family dictionary:
      AL_ = gnomAD/AllofUs allele frequencies; EK_4/5/6 = phastCons,
      EK_7 = phyloP100way, EK_2 = GERP++ RS.

    Features (stateless — identical at train and inference):
      acmg_max_af     max minor-allele frequency across population columns
                      (missing -> 0.0 = "absent"); common => benign
      acmg_ba1        1 if max_af > 0.05   (ACMG BA1: stand-alone benign)
      acmg_bs1        1 if max_af > 0.01   (ACMG BS1: strong benign)
      acmg_phastcons  mean of the phastCons tracks (conservation in [0,1])
      acmg_conserved  1 if phyloP100way > 2 or GERP++ RS > 2  (PP3-style)

    The frequency AND conservation groups TOGETHER lift the small PAH panel
    (F1 +0.044, AUC +0.017); either group alone drops PAH below baseline, so the
    full block is kept and gated to PAH (utils.ACMG_PANELS) — each panel is its own
    model and the block is neutral/harmful on the data-rich panels.

    Leakage guard: no "AF-is-missing" flag (missingness is a partial artefact);
    only high OBSERVED frequency drives the benign signal. Minor-AF columns are
    detected by signature (values in (0,1) + low column mean), excluding the
    allele-number column, the binary flags, and major-allele (~1) columns.
    """
    al_cols = [c for c in df.columns if c.startswith("AL_")]
    af_cols = [c for c in al_cols
               if ((df[c] > 0) & (df[c] < 1)).any() and df[c].mean(skipna=True) < 0.1]
    df["acmg_max_af"] = df[af_cols].max(axis=1).fillna(0.0) if af_cols else 0.0
    df["acmg_ba1"] = (df["acmg_max_af"] > 0.05).astype(int)
    df["acmg_bs1"] = (df["acmg_max_af"] > 0.01).astype(int)

    pc = [c for c in ("EK_4", "EK_5", "EK_6") if c in df.columns]
    if pc:
        df["acmg_phastcons"] = df[pc].mean(axis=1)
    conserved = pd.Series(False, index=df.index)
    if "EK_7" in df.columns:          # phyloP100way
        conserved = conserved | (df["EK_7"] > 2.0)
    if "EK_2" in df.columns:          # GERP++ RS
        conserved = conserved | (df["EK_2"] > 2.0)
    df["acmg_conserved"] = conserved.fillna(False).astype(int)

    return df


# ---------------------------------------------------------------------------
# Cross-feature interactions
#
# These mix AA-derived features with conservation columns (EK_*) to capture
# patterns the model would otherwise need depth to learn, e.g.:
#   - high conservation × disruptive substitution = strong pathogenic signal
#   - high conservation × conservative substitution = hypomorphic candidate
#     (the regime the team's PSR cites as ~46% accuracy in literature)
# ---------------------------------------------------------------------------

def engineer_interaction_features(df: pd.DataFrame,
                                  conservation_col: str = "EK_7",
                                  inplace: bool = False) -> pd.DataFrame:
    """Add (conservation × biochemical-disruption) interaction features.

    EK_7 is the strongest single conservation-like feature in MASTER
    (label AUC ~0.74). This routine assumes EK_7 is present; if absent,
    falls back to the first EK_* that has neg+pos values.
    """
    if not inplace:
        df = df.copy()

    cons_col = conservation_col if conservation_col in df.columns else None
    if cons_col is None:
        for c in [f"EK_{i}" for i in (7, 9, 2, 3)]:
            if c in df.columns:
                cons_col = c
                break
    if cons_col is None or "AA_blosum62" not in df.columns:
        return df

    cons = df[cons_col].astype(float)
    blos = df["AA_blosum62"].astype(float)
    # Normalised "disruption": maps BLOSUM [+11, -4] -> [0, 1]
    # High value = more chemically disruptive
    disruption = (4.0 - blos) / 15.0
    disruption = disruption.clip(0.0, 1.0)

    # Both-high signal (strong pathogenic)
    df["FX_cons_x_disrupt"] = cons * disruption
    # Hypomorphic signature: high conservation AND CONSERVATIVE change (low disrupt)
    df["FX_cons_x_conservative"] = cons * (1.0 - disruption)
    return df


# ---------------------------------------------------------------------------
# Score-agreement (in-silico consensus) features
#
# Fitted on training data:
#   - select top-K numeric features by label-AUC as "anchor scores"
#   - normalise each to its observed [0, 1] range on training
#   - threshold = 0.5 (after normalisation)
# Applied at inference:
#   - normalise each anchor by the saved (min, max)
#   - count of anchors above 0.5 -> score_agree_count
#   - mean of normalised anchors  -> score_agree_mean
#   - std of normalised anchors   -> score_agree_std (model uncertainty)
# ---------------------------------------------------------------------------

def fit_score_agreement(X_train: pd.DataFrame,
                        y_train: pd.Series,
                        top_k: int = 7,
                        candidates: list | None = None) -> dict:
    """Pick the top-K label-AUC numeric features as anchors and save their ranges.

    Returns a dict suitable for joblib serialisation: anchor names + per-anchor
    (min, max, sign) where sign=+1 means "higher = more pathogenic" and -1 means
    inverted (SIFT-like, lower = more pathogenic).
    """
    from sklearn.metrics import roc_auc_score
    import numpy as np

    if candidates is None:
        candidates = [c for c in X_train.columns
                      if not isinstance(X_train[c].dtype, pd.CategoricalDtype)
                      and X_train[c].dtype != object]

    y_arr = np.asarray(y_train)
    scored = []
    for c in candidates:
        v = X_train[c].values.astype(float)
        m = ~np.isnan(v)
        if m.sum() < 30 or pd.Series(y_arr[m]).nunique() < 2:
            continue
        if v[m].std() < 1e-9:
            continue
        try:
            auc = float(roc_auc_score(y_arr[m], v[m]))
        except Exception:
            continue
        sign = 1 if auc >= 0.5 else -1
        symmetric_auc = max(auc, 1 - auc)
        scored.append((c, symmetric_auc, sign))
    scored.sort(key=lambda t: -t[1])
    top = scored[:top_k]

    anchors = {}
    for name, _auc, sign in top:
        v = X_train[name].values.astype(float)
        vp = v[~np.isnan(v)]
        anchors[name] = {
            "min":  float(vp.min()),
            "max":  float(vp.max()),
            "sign": int(sign),
        }
    return {"anchors": anchors, "k": len(anchors)}


def fit_quadrants(X_train: pd.DataFrame,
                  cons_col: str = "EK_7",
                  biochem_cols: tuple = ("AA_mw_delta", "AA_polarity_delta"),
                  quantile: float = 0.5) -> dict:
    """Fit median (or other quantile) thresholds that divide variants into 4 regimes.

    Conservation axis: `cons_col` (default EK_7, the strongest single conservation
    feature for PathoSense).
    Biochem axis: max of normalized `biochem_cols` (default MW + polarity deltas —
    the two top discriminators in PAH error analysis).

    Returns dict with thresholds + normalization mins/maxes for inference replay.
    """
    import numpy as np

    fitted: dict = {"cons_col": cons_col, "biochem_cols": list(biochem_cols),
                    "quantile": quantile}
    if cons_col not in X_train.columns:
        fitted["cons_threshold"] = None
        return fitted
    cons_vals = X_train[cons_col].dropna().astype(float)
    fitted["cons_threshold"] = float(cons_vals.quantile(quantile)) if len(cons_vals) > 0 else None

    # Normalise each biochem col to [0,1] using train min/max, then take elementwise max.
    biochem_meta = {}
    for c in biochem_cols:
        if c not in X_train.columns:
            continue
        v = X_train[c].dropna().astype(float)
        if len(v) == 0:
            continue
        biochem_meta[c] = {"min": float(v.min()), "max": float(v.max())}
    fitted["biochem_meta"] = biochem_meta

    # Compute composite biochem score on training to get its threshold.
    biochem_composite = _composite_biochem(X_train, biochem_meta)
    if biochem_composite is not None and biochem_composite.notna().any():
        fitted["biochem_threshold"] = float(biochem_composite.dropna().quantile(quantile))
    else:
        fitted["biochem_threshold"] = None
    return fitted


def _composite_biochem(df: pd.DataFrame, biochem_meta: dict):
    """Elementwise max of normalised biochem cols. NaN-aware."""
    import numpy as np

    if not biochem_meta:
        return None
    cols = []
    for c, meta in biochem_meta.items():
        if c not in df.columns:
            continue
        v = df[c].astype(float).values
        rng = max(meta["max"] - meta["min"], 1e-9)
        normed = (v - meta["min"]) / rng
        cols.append(np.clip(normed, 0.0, 1.0))
    if not cols:
        return None
    stacked = np.vstack(cols)                       # shape (k, n)
    composite = np.nanmax(stacked, axis=0)
    return pd.Series(composite, index=df.index)


def apply_quadrants(df: pd.DataFrame, fitted: dict,
                    inplace: bool = False) -> pd.DataFrame:
    """Add 4 binary quadrant features + the continuous composite biochem score."""
    import numpy as np

    if not inplace:
        df = df.copy()

    cons_col = fitted.get("cons_col", "EK_7")
    cons_t   = fitted.get("cons_threshold")
    bio_meta = fitted.get("biochem_meta", {})
    bio_t    = fitted.get("biochem_threshold")

    if cons_t is None or bio_t is None or cons_col not in df.columns or not bio_meta:
        # Fall back to all-zero flags so downstream code doesn't crash
        for name in ("Q_typical_path", "Q_hypomorphic", "Q_tolerable", "Q_typical_benign",
                     "Q_biochem_composite"):
            df[name] = 0.0
        return df

    cons_vals = df[cons_col].astype(float)
    biochem_composite = _composite_biochem(df, bio_meta)
    if biochem_composite is None:
        biochem_composite = pd.Series(np.full(len(df), np.nan), index=df.index)

    high_cons = (cons_vals > cons_t).astype(int)
    low_cons  = (cons_vals <= cons_t).astype(int)
    high_bio  = (biochem_composite > bio_t).astype(int)
    low_bio   = (biochem_composite <= bio_t).astype(int)

    df["Q_typical_path"]   = high_cons * high_bio     # canonical pathogenic regime
    df["Q_hypomorphic"]    = high_cons * low_bio      # high cons but small change → PAH FN territory
    df["Q_tolerable"]      = low_cons  * high_bio     # low cons but big change → PAH FP territory
    df["Q_typical_benign"] = low_cons  * low_bio      # canonical benign regime
    df["Q_biochem_composite"] = biochem_composite.values
    return df


def apply_score_agreement(df: pd.DataFrame, fitted: dict,
                          threshold: float = 0.5,
                          inplace: bool = False) -> pd.DataFrame:
    """Add 3 new columns derived from the saved anchor set."""
    import numpy as np

    if not inplace:
        df = df.copy()

    anchors = fitted.get("anchors", {})
    if not anchors:
        df["FX_score_agree_count"] = 0
        df["FX_score_agree_mean"]  = np.nan
        df["FX_score_agree_std"]   = np.nan
        return df

    norm_cols = []
    n = len(df)
    for name, meta in anchors.items():
        if name not in df.columns:
            continue
        v = df[name].values.astype(float)
        lo, hi = meta["min"], meta["max"]
        denom = max(hi - lo, 1e-9)
        normed = (v - lo) / denom
        if meta["sign"] < 0:
            normed = 1.0 - normed
        normed = np.clip(normed, 0.0, 1.0)
        norm_cols.append(normed)

    if not norm_cols:
        df["FX_score_agree_count"] = 0
        df["FX_score_agree_mean"]  = np.nan
        df["FX_score_agree_std"]   = np.nan
        return df

    stacked = np.vstack(norm_cols)               # shape (k, n)
    # Mean / std ignoring NaN
    df["FX_score_agree_mean"]  = np.nanmean(stacked, axis=0)
    df["FX_score_agree_std"]   = np.nanstd(stacked, axis=0)
    # Count of anchors with normalised value > threshold (NaN counts as 0)
    above = np.where(np.isnan(stacked), 0, (stacked > threshold).astype(int))
    df["FX_score_agree_count"] = above.sum(axis=0)
    return df


# ---------------------------------------------------------------------------
# Self-test (run as script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np
    test = pd.DataFrame({
        "AA_1": ["A", "C", "G", "R", "AG", None, "P"],
        "AA_2": ["V", "S", "S", "W", "S", "L", "*"],
        "EK_7": [5.0, 8.5, -1.0, 7.2, 3.0, 4.0, 9.0],
        "AL_x": [0.1, 0.8, 0.05, 0.9, np.nan, 0.5, 0.6],
        "Label": [0, 1, 0, 1, 1, 0, 1],
    })
    out = engineer_aa_features(test)
    out = engineer_interaction_features(out)
    print("After AA + interactions:")
    print(out.to_string(index=False))
    aa_cols = [c for c in out.columns if c.startswith("AA_")]
    fx_cols = [c for c in out.columns if c.startswith("FX_")]
    print(f"\nAA columns added: {aa_cols}")
    print(f"FX columns added: {fx_cols}")

    fitted = fit_score_agreement(out[["AL_x", "EK_7", "AA_blosum62"]], out["Label"], top_k=3)
    print(f"\nfitted anchors: {fitted}")
    out2 = apply_score_agreement(out, fitted)
    print(f"\nWith score agreement:")
    print(out2[["AA_1","AA_2","EK_7","AL_x","FX_score_agree_count","FX_score_agree_mean","FX_score_agree_std"]].to_string(index=False))
