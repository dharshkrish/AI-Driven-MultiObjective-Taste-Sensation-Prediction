# app.py
import os
import json
import urllib.parse
import urllib.request
import xmltodict
import pandas as pd
import numpy as np
import streamlit as st
from rdkit import Chem
from rdkit.Chem import MolToSmiles
from rdkit.Chem import AllChem
import zipfile
import io
import joblib
import pickle
import warnings

warnings.filterwarnings("ignore")

# ----------------- CONFIG: exact paths user provided -----------------
BASE_EXTERNAL = r"C:\projom\VirtuousMultiTaste\data\ExternalDBs"

PATH_FOOD_TASTE = os.path.join(BASE_EXTERNAL, "food_taste_database_extended.csv")
PATH_COMPOUND_DB = os.path.join(BASE_EXTERNAL, "food_compounds_db.csv")
PATH_INGREDIENT_DB = os.path.join(BASE_EXTERNAL, "food_ingredient_db.json")
PATH_FOODDB = os.path.join(BASE_EXTERNAL, "FoodDB.csv")

# model candidates (zip / pkl) - search in a few likely places
MODEL_CANDIDATES = [
    os.path.join(BASE_EXTERNAL, "model2_fourtaste.zip"),
    os.path.join(BASE_EXTERNAL, "model2_fourtaste.pkl"),
    os.path.join(os.getcwd(), "model2_fourtaste.zip"),
    os.path.join(os.getcwd(), "model2_fourtaste.pkl"),
    os.path.join(os.getcwd(), "model2_fourtaste.joblib"),
    os.path.join(os.getcwd(), "model2_fourtaste", "model.pkl"),
    os.path.join(os.getcwd(), "model2_fourtaste", "model.joblib"),
]

# Predictions CSV (model outputs with probabilities)
PRED_CANDIDATES = [
    os.path.join(os.getcwd(), "predictions.csv"),
    os.path.join(os.getcwd(), "predicted", "predictions.csv"),
    os.path.join(BASE_EXTERNAL, "predictions.csv"),
    os.path.join(os.getcwd(), "PredictionMultiTaste", "predictions.csv"),
]

# ----------------- DEFAULT FOOD COMPOUND TASTE PROFILES -----------------
# These are NOT chemical industrial compounds — only FOOD-related taste compounds.
# Used when no model prediction exists for a compound.
DEFAULT_FOOD_PROFILES = {
    "Sweet": {
        "Bitter": 0.05,
        "Sweet": 0.75,
        "Umami": 0.05,
        "Other": 0.15
    },

    "Umami": {
        "Bitter": 0.10,
        "Sweet": 0.05,
        "Umami": 0.70,
        "Other": 0.15
    },

    "Bitter": {
        "Bitter": 0.72,
        "Sweet": 0.05,
        "Umami": 0.05,
        "Other": 0.18
    },

    "Other": {
        "Bitter": 0.10,
        "Sweet": 0.10,
        "Umami": 0.10,
        "Other": 0.70
    }
}

# ----------------- HELPERS FOR LOADING FILES (robust) -----------------
@st.cache_data
def safe_read_csv(paths):
    for p in paths if isinstance(paths, (list, tuple)) else [paths]:
        if p and os.path.exists(p):
            try:
                return pd.read_csv(p, encoding="utf-8", on_bad_lines="skip", engine="python")
            except Exception:
                try:
                    return pd.read_csv(p, encoding="utf-8")
                except Exception:
                    continue
    return None

@st.cache_data
def safe_read_json(paths):
    for p in paths if isinstance(paths, (list, tuple)) else [paths]:
        if p and os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                continue
    return {}

# ----------------- Load DBs -----------------
food_taste_db = safe_read_csv([PATH_FOOD_TASTE])   # authoritative Food -> Taste -> Compound
compound_db = safe_read_csv([PATH_COMPOUND_DB])    # compound metadata (may include SMILES)
ingredient_db = safe_read_json([PATH_INGREDIENT_DB])
fooddb_alt = safe_read_csv([PATH_FOODDB])
predictions_df = safe_read_csv(PRED_CANDIDATES)

# If predictions_df found, canonicalize its SMILES column
if predictions_df is not None and "SMILES" in predictions_df.columns:
    def canonicalize_smi(s):
        try:
            m = Chem.MolFromSmiles(str(s))
            if m:
                return MolToSmiles(m, canonical=True)
        except Exception:
            pass
        return None
    predictions_df["SMILES_CANON"] = predictions_df["SMILES"].astype(str).apply(canonicalize_smi)

# ----------------- PubChem SMILES lookup (fallback) -----------------
def pubchem_smiles(name):
    """Return Canonical SMILES from PubChem for a compound name, or None."""
    try:
        base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        url = base + urllib.parse.quote(name) + "/property/CanonicalSMILES/XML"
        data = urllib.request.urlopen(url, timeout=10).read()
        info = xmltodict.parse(data)
        smiles = info["PropertyTable"]["Properties"]["CanonicalSMILES"]
        # canonicalize
        try:
            m = Chem.MolFromSmiles(smiles)
            return MolToSmiles(m, canonical=True) if m else smiles
        except Exception:
            return smiles
    except Exception:
        return None

# ----------------- Resolution helpers -----------------
def is_smiles(s):
    try:
        m = Chem.MolFromSmiles(str(s))
        return m is not None
    except Exception:
        return False

def canonicalize(s):
    try:
        m = Chem.MolFromSmiles(str(s))
        if m:
            return MolToSmiles(m, canonical=True)
    except Exception:
        pass
    return None

def find_smiles_for_compound_local(compound_name):
    """
    Try to find SMILES for a compound name in local compound DBs.
    This looks for typical SMILES column names (SMILES, SMILES_CHECK, Std_SMILES etc.)
    """
    if compound_db is None:
        return None
    # lower-case column names
    cols = [c.lower() for c in compound_db.columns]
    # possible SMILES columns
    candidates = [c for c in compound_db.columns if "smile" in c.lower() or "smi" in c.lower()]
    # direct name match: try columns that may have compound names
    name_cols = [c for c in compound_db.columns if "compound" in c.lower() or "name" in c.lower()]
    # Try direct lookups in name columns
    for name_col in name_cols:
        try:
            mask = compound_db[name_col].astype(str).str.strip().str.lower() == str(compound_name).strip().lower()
            if mask.any():
                row = compound_db[mask].iloc[0]
                # get SMILES candidate
                for s_col in candidates:
                    val = row.get(s_col)
                    if pd.notna(val) and str(val).strip():
                        can = canonicalize(val)
                        if can:
                            return can
                        else:
                            return str(val).strip()
        except Exception:
            continue
    # If no direct name match, try substring search in 'Compound' column
    if "compound" in cols:
        try:
            colname = compound_db.columns[cols.index("compound")]
            mask = compound_db[colname].astype(str).str.lower().str.contains(str(compound_name).strip().lower(), na=False)
            if mask.any():
                row = compound_db[mask].iloc[0]
                for s_col in candidates:
                    val = row.get(s_col)
                    if pd.notna(val) and str(val).strip():
                        can = canonicalize(val)
                        if can:
                            return can
                        else:
                            return str(val).strip()
        except Exception:
            pass
    return None

def resolve_compound_to_smiles(compound_name):
    """
    Resolve a compound name to SMILES using:
    1. local compound DB
    2. predictions_df (if it contains a 'Compound' column or 'Name' column)
    3. PubChem fallback
    """
    # 1) local compound DB
    smi = find_smiles_for_compound_local(compound_name)
    if smi:
        return smi

    # 2) check predictions_df for a 'Compound' or 'Name' column (some DBs include name->SMILES mapping)
    if predictions_df is not None:
        for col in predictions_df.columns:
            if col.lower() in ("compound", "name"):
                # exact match
                mask = predictions_df[col].astype(str).str.strip().str.lower() == compound_name.strip().lower()
                if mask.any():
                    row = predictions_df[mask].iloc[0]
                    if "SMILES" in row and pd.notna(row["SMILES"]):
                        can = canonicalize(row["SMILES"])
                        return can or row["SMILES"]

    # 3) PubChem fallback
    smi = pubchem_smiles(compound_name)
    if smi:
        return smi

    return None

# ----------------- Lookup prediction by SMILES in predictions_df -----------------
def lookup_prediction_by_smiles(smi):
    """
    Accepts smi (any form). Canonicalizes and looks up in predictions_df["SMILES_CANON"].
    Returns dict with Bitter,Sweet,Umami,Other,Dominant or None.
    """
    if predictions_df is None or "SMILES_CANON" not in predictions_df.columns:
        return None

    can = canonicalize(smi)
    if can is None:
        return None

    matches = predictions_df[predictions_df["SMILES_CANON"] == can]
    if not matches.empty:
        row = matches.iloc[0]
        # Expecting columns like "Bitter","Sweet","Umami","Other","Dominant"
        out = {}
        for col in ["Bitter", "Sweet", "Umami", "Other", "Dominant"]:
            if col in row.index:
                val = row[col]
                if col != "Dominant":
                    try:
                        out[col] = float(val)
                    except Exception:
                        out[col] = np.nan
                else:
                    out[col] = str(val)
        return out
    return None

# ----------------- ML model loader & descriptor generator -----------------
@st.cache_resource
def load_model_from_candidates(candidates):
    """
    Try to locate and load a ML model. Supports:
     - direct .pkl or .joblib files (sklearn pipelines)
     - .zip archives that contain a .pkl/.joblib inside
    Returns loaded model or None.
    """
    for p in candidates:
        if not p:
            continue
        if os.path.exists(p):
            # direct pkl or joblib
            if p.endswith((".pkl", ".joblib")):
                try:
                    model = joblib.load(p)
                    st.info(f"Loaded model from {p}")
                    return model
                except Exception:
                    try:
                        with open(p, "rb") as fh:
                            model = pickle.load(fh)
                            st.info(f"Loaded pickled model from {p}")
                            return model
                    except Exception:
                        st.warning(f"Found model file {p} but could not load it.")
                        continue
            # zip - try to open and find a pkl inside
            if p.endswith(".zip") or zipfile.is_zipfile(p):
                try:
                    with zipfile.ZipFile(p, "r") as zf:
                        # search for typical model filenames
                        candidates_inside = [name for name in zf.namelist() if name.endswith((".pkl", ".joblib"))]
                        if not candidates_inside:
                            # if zip itself is a directory zip, maybe a single file; try first file
                            if zf.namelist():
                                candidates_inside = [zf.namelist()[0]]
                        for inside in candidates_inside:
                            try:
                                with zf.open(inside) as fh:
                                    data = fh.read()
                                    bio = io.BytesIO(data)
                                    try:
                                        model = joblib.load(bio)
                                        st.info(f"Loaded model {inside} from zip {p}")
                                        return model
                                    except Exception:
                                        try:
                                            bio.seek(0)
                                            model = pickle.load(bio)
                                            st.info(f"Loaded pickled model {inside} from zip {p}")
                                            return model
                                        except Exception:
                                            continue
                            except Exception:
                                continue
                except Exception:
                    st.warning(f"Could not read zip model file: {p}")
                    continue
    return None

# simple Morgan fingerprint descriptor generator (2048 bits, radius 2)
def mol_to_morgan_array(smiles, n_bits=2048, radius=2):
    try:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return None
        arr = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
        # convert ExplicitBitVect to numpy array
        a = np.zeros((1, n_bits), dtype=np.int8)
        AllChem.DataStructs.ConvertToNumpyArray(arr, a[0])
        return a.flatten()
    except Exception:
        return None

# Prediction wrapper that accepts model and SMILES and returns dict with probs
def model_predict_for_smiles(model, smiles):
    """
    Given a sklearn-like model and SMILES, return a dict: {"Bitter":..,"Sweet":..,"Umami":..,"Other":..,"Dominant":..}
    Model is expected to either:
     - accept fingerprint vectors and implement predict_proba returning Nx4 array in consistent column order
     - OR accept DataFrame with named features (less common) — we pass numpy vector first
    We will attempt using predict_proba; if not available, attempt predict and map.
    """
    if model is None:
        return None

    fp = mol_to_morgan_array(smiles)
    if fp is None:
        return None

    # Some pipelines expect 2D array
    X = np.asarray(fp).reshape(1, -1)

    # Try predict_proba
    try:
        # predict_proba could return class-prob order; check model.classes_ if available
        probs = None
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(X)  # If multiclass, might return shape (n_samples, n_classes)
            except Exception:
                # some multioutput classifiers return list of arrays
                try:
                    # If it's a list of arrays, combine
                    proba_list = model.predict_proba(X)
                    # attempt to build 2D array from list elements
                    if isinstance(proba_list, list):
                        # each element: (n_samples, n_classes_for_label) - not trivial to combine; so skip
                        probs = None
                except Exception:
                    probs = None
        # If we got proper probs as 2D array
        if probs is not None and probs.ndim == 2:
            # Determine class order
            # We expect classes to be something like ['Bitter','Other','Sweet','Umami'] or numeric labels
            class_order = None
            if hasattr(model, "classes_"):
                class_order = list(model.classes_)
            # If class_order contains strings like 'Bitter' etc, map accordingly.
            # Otherwise, assume order is [Bitter, Sweet, Umami, Other] (common in our pipeline).
            labels = ["Bitter", "Sweet", "Umami", "Other"]
            probs_row = probs[0]
            mapped = {}
            if class_order and all(isinstance(c, str) for c in class_order):
                # map based on class_order
                # If class_order matches labels (case-insensitive), map accordingly; else fallback to index order mapping
                lower_co = [str(c).lower() for c in class_order]
                mapped = {}
                for i, lab in enumerate(labels):
                    if lab.lower() in lower_co:
                        idx = lower_co.index(lab.lower())
                        mapped[lab] = float(probs_row[idx])
                    else:
                        # fallback: if labels length equals probs length, try index-based mapping
                        if len(probs_row) >= 4 and i < len(probs_row):
                            mapped[lab] = float(probs_row[i])
                        else:
                            mapped[lab] = 0.0
            else:
                # fallback index mapping
                if len(probs_row) >= 4:
                    mapped = {
                        "Bitter": float(probs_row[0]),
                        "Sweet": float(probs_row[1]),
                        "Umami": float(probs_row[2]),
                        "Other": float(probs_row[3])
                    }
                else:
                    # if odd shape, can't interpret
                    return None
            # ensure sum to ~1 (normalize)
            vals = np.array([mapped["Bitter"], mapped["Sweet"], mapped["Umami"], mapped["Other"]], dtype=float)
            s = vals.sum()
            if s > 0:
                vals = vals / s
            mapped = {"Bitter": float(vals[0]), "Sweet": float(vals[1]), "Umami": float(vals[2]), "Other": float(vals[3])}
            dom = ["Bitter", "Sweet", "Umami", "Other"][int(np.nanargmax(vals))]
            mapped["Dominant"] = dom
            return mapped
    except Exception:
        pass

    # If predict_proba not available or failed, try predict (class label)
    try:
        preds = None
        if hasattr(model, "predict"):
            preds = model.predict(X)
        if preds is not None:
            # If predict returns class probabilities as array-like (rare), try to coerce; else if returns label, we can't provide probabilities
            if isinstance(preds, np.ndarray) and preds.ndim == 2 and preds.shape[1] >= 4:
                probs_row = preds[0]
                mapped = {
                    "Bitter": float(probs_row[0]),
                    "Sweet": float(probs_row[1]),
                    "Umami": float(probs_row[2]),
                    "Other": float(probs_row[3])
                }
                vals = np.array([mapped["Bitter"], mapped["Sweet"], mapped["Umami"], mapped["Other"]], dtype=float)
                s = vals.sum()
                if s > 0:
                    vals = vals / s
                mapped = {"Bitter": float(vals[0]), "Sweet": float(vals[1]), "Umami": float(vals[2]), "Other": float(vals[3])}
                dom = ["Bitter", "Sweet", "Umami", "Other"][int(np.nanargmax(vals))]
                mapped["Dominant"] = dom
                return mapped
            else:
                # treat label as dominant
                label = preds[0]
                mapped = {"Bitter": 0.0, "Sweet": 0.0, "Umami": 0.0, "Other": 0.0}
                # if label is index 0-3
                try:
                    if isinstance(label, (int, np.integer)) and 0 <= int(label) < 4:
                        mapped[["Bitter", "Sweet", "Umami", "Other"][int(label)]] = 1.0
                    else:
                        # string label attempt
                        lab_str = str(label).strip().lower()
                        for k in mapped.keys():
                            if k.lower() == lab_str:
                                mapped[k] = 1.0
                                break
                except Exception:
                    pass
                mapped["Dominant"] = max(mapped.keys(), key=lambda k: mapped[k])
                return mapped
    except Exception:
        pass

    return None

# ----------------- Utility: default profile selection by name -----------------
def default_profile_for_compound(compound_name):
    """
    Map compound category to a default taste profile.
    Assumes FOOD-BASED compounds (sugars, peptides, amino acids).
    """
    if not compound_name:
        return DEFAULT_FOOD_PROFILES["Other"]

    name = str(compound_name).lower()

    # Sweet compounds (sugars etc.)
    if any(x in name for x in ["sugar", "glucose", "fructose", "sucrose", "maltose", "sweet", "mannitol", "xylitol"]):
        return DEFAULT_FOOD_PROFILES["Sweet"]

    # Umami compounds (amino acids, nucleotides)
    if any(x in name for x in ["glutamate", "glutamic", "inosinate", "guanylate", "umami", "msg", "monosodium glutamate"]):
        return DEFAULT_FOOD_PROFILES["Umami"]

    # Bitter compounds (alkaloids, polyphenols)
    if any(x in name for x in ["caffeine", "quinine", "tannin", "alkaloid", "bitter", "theobromine", "catechin"]):
        return DEFAULT_FOOD_PROFILES["Bitter"]

    # Everything else → Other
    return DEFAULT_FOOD_PROFILES["Other"]

# ----------------- Load model once (cached) -----------------
model = load_model_from_candidates(MODEL_CANDIDATES)

# ----------------- Food prediction orchestration -----------------
def predict_for_food(food_name):
    """
    For a given food name:
      - look up authoritative row in food_taste_db (Food_Item)
      - if found, use row["Compound"] (or multiple) -> resolve to SMILES -> lookup predictions.csv -> build output rows
      - if not in CSV, try ML model
      - otherwise fallback to DEFAULT_FOOD_PROFILES
    Returns list of result dicts (one per matched compound) and aggregated mean probabilities row.
    """
    if food_taste_db is None:
        return {"error": "Food taste DB not found on disk (looked at {}).".format(PATH_FOOD_TASTE)}

    # try exact or fuzzy match: we do case-insensitive exact or substring match
    df = food_taste_db.copy()
    # normalize column name
    if "Food_Item" not in df.columns and "Food Item" in df.columns:
        df = df.rename(columns={"Food Item": "Food_Item"})

    # Find rows where Food_Item matches (case-insensitive)
    mask_exact = df["Food_Item"].astype(str).str.strip().str.lower() == food_name.strip().lower()
    mask_contain = df["Food_Item"].astype(str).str.strip().str.lower().str.contains(food_name.strip().lower(), na=False)
    use_mask = mask_exact if mask_exact.any() else (mask_contain if mask_contain.any() else None)
    if use_mask is None:
        return {"error": f"{food_name} not found in food taste DB."}

    matched_rows = df[use_mask]
    results = []
    probs_accum = np.zeros(4)
    hit_count = 0

    for _, row in matched_rows.iterrows():
        compound = row.get("Compound")
        notes = row.get("Notes") if "Notes" in row else None
        alt = row.get("Healthy_Alternative") if "Healthy_Alternative" in row else row.get("Healthy Alternative", None)

        # Resolve compound to SMILES
        smi = None
        if pd.notna(compound) and str(compound).strip():
            smi = resolve_compound_to_smiles(compound)

        # 1) Try predictions.csv
        pred = None
        if smi:
            pred = lookup_prediction_by_smiles(smi)

        # 2) If no predictions.csv result, try to find by substring in predictions_df (older heuristic)
        if pred is None and predictions_df is not None and compound:
            found = None
            for col in predictions_df.columns:
                try:
                    mask = predictions_df[col].astype(str).str.lower().str.contains(str(compound).strip().lower(), na=False)
                    if mask.any():
                        found = predictions_df[mask].iloc[0]
                        break
                except Exception:
                    continue
            if found is not None:
                pred = {}
                for col in ["Bitter", "Sweet", "Umami", "Other", "Dominant"]:
                    if col in found.index:
                        try:
                            pred[col] = float(found[col]) if col != "Dominant" else str(found[col])
                        except Exception:
                            pred[col] = found[col]

        # 3) If still no prediction, try ML model fallback
        model_used = None
        if pred is None and model is not None and smi is not None:
            try:
                mp = model_predict_for_smiles(model, smi)
                if mp is not None:
                    pred = mp
                    model_used = "MODEL"
            except Exception:
                pred = None

        # 4) If still no prediction, use DEFAULT FOOD PROFILE (food-only)
        if pred is None:
            default_prof = default_profile_for_compound(compound)
            pred = {
                "Bitter": default_prof["Bitter"],
                "Sweet": default_prof["Sweet"],
                "Umami": default_prof["Umami"],
                "Other": default_prof["Other"],
                "Dominant": max(default_prof, key=lambda k: default_prof[k])
            }
            model_used = "DEFAULT"

        # accumulate numeric probabilities if present
        try:
            arr = np.array([float(pred.get("Bitter", np.nan)), float(pred.get("Sweet", np.nan)),
                            float(pred.get("Umami", np.nan)), float(pred.get("Other", np.nan))], dtype=float)
            if not np.isnan(arr).all():
                probs_accum += np.nan_to_num(arr)
                hit_count += 1
        except Exception:
            pass

        status = "OK"
        if model_used == "MODEL":
            status = "Model Prediction Used"
        elif model_used == "DEFAULT":
            status = "Default Profile Used"
        else:
            status = "CSV Prediction Used"

        results.append({
            "Food_Item": row.get("Food_Item"),
            "Compound": compound,
            "SMILES": smi or None,
            "Bitter": float(pred.get("Bitter")) if pred.get("Bitter") is not None else None,
            "Sweet": float(pred.get("Sweet")) if pred.get("Sweet") is not None else None,
            "Umami": float(pred.get("Umami")) if pred.get("Umami") is not None else None,
            "Other": float(pred.get("Other")) if pred.get("Other") is not None else None,
            "Dominant": pred.get("Dominant"),
            "Healthy_Alternative": alt,
            "Notes": notes or "",
            "Status": status
        })

    # aggregate mean if any numeric preds
    aggregated = None
    if hit_count > 0:
        mean_probs = probs_accum / float(hit_count)
        aggregated = {
            "Food_Item": food_name,
            "Compound": None,
            "SMILES": None,
            "Bitter": float(mean_probs[0]),
            "Sweet": float(mean_probs[1]),
            "Umami": float(mean_probs[2]),
            "Other": float(mean_probs[3]),
            "Dominant": ["Bitter", "Sweet", "Umami", "Other"][int(np.nanargmax(mean_probs))],
            "Healthy_Alternative": None,
            "Notes": f"Aggregated mean from {hit_count} compound predictions",
            "Status": "Aggregated"
        }

    return {"rows": results, "aggregated": aggregated}

# ----------------- UI -----------------
st.set_page_config(page_title="AI Multi-Taste Predictor (CSV + Model fallback)", layout="wide")
st.title("🍽️ AI Driven Multi-Taste Sensation Predictor — CSV + ML Fallback")

st.sidebar.header("Mode & Input")
mode = st.sidebar.radio("Mode", ["Compound / SMILES", "Food Lookup (DB-first)"])
page = st.sidebar.radio("Page", ["Input & Table", "Detailed Row"])

if mode == "Compound / SMILES":
    user_input = st.sidebar.text_input("Enter SMILES or Compound name (e.g., 'CCO', 'Glutamic acid')", value="")
else:
    user_input = st.sidebar.text_input("Enter food name (e.g., Idli, Pizza)", value="Idli")

run_btn = st.sidebar.button("🔮 Predict / Lookup")

st.markdown("---")

def format_float(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return ""
        return f"{float(x):.2f}"
    except Exception:
        return str(x)

if run_btn and user_input.strip():
    q = user_input.strip()
    if mode == "Compound / SMILES":
        # If input is SMILES, canonicalize and lookup
        if is_smiles(q):
            can = canonicalize(q)
            pred = lookup_prediction_by_smiles(can)
            used = "CSV"
            model_pred = None
            if pred is None and model is not None:
                model_pred = model_predict_for_smiles(model, can)
                if model_pred:
                    pred = model_pred
                    used = "MODEL"
            if pred is None:
                # use default profile based on input string
                default_prof = default_profile_for_compound(q)
                pred = {
                    "Bitter": default_prof["Bitter"],
                    "Sweet": default_prof["Sweet"],
                    "Umami": default_prof["Umami"],
                    "Other": default_prof["Other"],
                    "Dominant": max(default_prof, key=lambda k: default_prof[k])
                }
                used = "DEFAULT"

            row = {
                "SMILES": can,
                "Bitter": format_float(pred.get("Bitter")),
                "Sweet": format_float(pred.get("Sweet")),
                "Umami": format_float(pred.get("Umami")),
                "Other": format_float(pred.get("Other")),
                "Dominant": pred.get("Dominant"),
                "Source": used
            }
            st.subheader("Prediction (Compound / SMILES)")
            st.table(pd.DataFrame([row]))
        else:
            # treat input as compound name; resolve to SMILES, then lookup
            smi = resolve_compound_to_smiles(q)
            if smi is None:
                st.error(f"Could not resolve compound name '{q}' to SMILES (checked local DBs and PubChem).")
            else:
                pred = lookup_prediction_by_smiles(smi)
                used = "CSV"
                if pred is None and model is not None:
                    model_pred = model_predict_for_smiles(model, smi)
                    if model_pred:
                        pred = model_pred
                        used = "MODEL"
                if pred is None:
                    default_prof = default_profile_for_compound(q)
                    pred = {
                        "Bitter": default_prof["Bitter"],
                        "Sweet": default_prof["Sweet"],
                        "Umami": default_prof["Umami"],
                        "Other": default_prof["Other"],
                        "Dominant": max(default_prof, key=lambda k: default_prof[k])
                    }
                    used = "DEFAULT"

                row = {
                    "Compound": q,
                    "SMILES": smi,
                    "Bitter": format_float(pred.get("Bitter")),
                    "Sweet": format_float(pred.get("Sweet")),
                    "Umami": format_float(pred.get("Umami")),
                    "Other": format_float(pred.get("Other")),
                    "Dominant": pred.get("Dominant"),
                    "Source": used
                }
                st.subheader("Compound → SMILES → Prediction")
                st.table(pd.DataFrame([row]))

    else:  # Food Lookup
        out = predict_for_food(q)
        if "error" in out:
            st.error(out["error"])
        else:
            rows = out.get("rows", [])
            agg = out.get("aggregated")
            # Build dataframe for rows
            if rows:
                df_rows = pd.DataFrame([
                    {
                        "Food_Item": r.get("Food_Item"),
                        "Compound": r.get("Compound"),
                        "SMILES": r.get("SMILES"),
                        "Bitter": format_float(r.get("Bitter")),
                        "Sweet": format_float(r.get("Sweet")),
                        "Umami": format_float(r.get("Umami")),
                        "Other": format_float(r.get("Other")),
                        "Dominant": r.get("Dominant"),
                        "Healthy_Alternative": r.get("Healthy_Alternative"),
                        "Notes": r.get("Notes"),
                        "Status": r.get("Status")
                    } for r in rows
                ])
                st.subheader("Per-compound results")
                st.dataframe(df_rows, use_container_width=True)
            else:
                st.info("No compound rows found for this food in the authoritative DB.")

            if agg:
                st.subheader("Aggregated mean probabilities across compounds")
                st.table(pd.DataFrame([{
                    "Food_Item": agg["Food_Item"],
                    "Bitter": format_float(agg["Bitter"]),
                    "Sweet": format_float(agg["Sweet"]),
                    "Umami": format_float(agg["Umami"]),
                    "Other": format_float(agg["Other"]),
                    "Dominant": agg["Dominant"],
                    "Notes": agg["Notes"]
                }]))
            else:
                st.info("No numeric probability predictions were available to aggregate for this food.")

else:
    st.info("Choose a Mode and enter input on the left, then click 'Predict / Lookup'.")

# ----------------- Footer / diagnostics -----------------
st.markdown("---")
st.markdown("**Diagnostics / Files loaded:**")
st.write({
    "food_taste_database_extended.csv": PATH_FOOD_TASTE if os.path.exists(PATH_FOOD_TASTE) else "NOT FOUND",
    "food_compounds_db.csv": PATH_COMPOUND_DB if os.path.exists(PATH_COMPOUND_DB) else "NOT FOUND",
    "food_ingredient_db.json": PATH_INGREDIENT_DB if os.path.exists(PATH_INGREDIENT_DB) else "NOT FOUND",
    "FoodDB.csv": PATH_FOODDB if os.path.exists(PATH_FOODDB) else "NOT FOUND",
    "predictions.csv": next((p for p in PRED_CANDIDATES if os.path.exists(p)), "NOT FOUND"),
    "model": "Loaded" if model is not None else "NOT FOUND"
})
