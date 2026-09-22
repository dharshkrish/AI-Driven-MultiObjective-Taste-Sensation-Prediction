import os
import pandas as pd
import numpy as np
import random
import streamlit as st
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
st.set_page_config(page_title="AI Multi-Taste Predictor", layout="wide")

# ---------------------------------------------------------------------
# FOOD DB PATHS
# ---------------------------------------------------------------------
FOOD_DB_PATHS = [
    os.path.join(os.getcwd(), "food_taste_database_extended.csv"),
    os.path.join(os.getcwd(), "data", "ExternalDBs", "food_taste_database_extended.csv"),
    r"C:\projom\VirtuousMultiTaste\data\ExternalDBs\food_taste_database_extended.csv",
]

# ---------------------------------------------------------------------
# FIXED: SAFE FOOD CSV LOADING
# ---------------------------------------------------------------------
@st.cache_data
def load_food_db():
    for path in FOOD_DB_PATHS:
        if os.path.exists(path):
            try:
                return pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
            except:
                return pd.read_csv(path, encoding="utf-8", on_bad_lines="skip", engine="python")
    return None

# ---------------------------------------------------------------------
# DEFAULT FOOD PROFILES
# ---------------------------------------------------------------------
DEFAULT_FOOD_PROFILES = {
    "Sweet":  {"Bitter": 0.05, "Sweet": 0.75, "Umami": 0.05, "Other": 0.15},
    "Umami":  {"Bitter": 0.10, "Sweet": 0.05, "Umami": 0.70, "Other": 0.15},
    "Bitter": {"Bitter": 0.72, "Sweet": 0.05, "Umami": 0.05, "Other": 0.18},
    "Other":  {"Bitter": 0.10, "Sweet": 0.10, "Umami": 0.10, "Other": 0.70}
}

# ---------------------------------------------------------------------
# FOOD MODE PROCESSING
# ---------------------------------------------------------------------
def predict_food_item(food_name, db):
    if db is None or db.empty:
        return None, "Food taste database missing."

    mask = db["Food_Item"].str.lower() == food_name.lower().strip()
    if not mask.any():
        mask = db["Food_Item"].str.lower().str.contains(food_name.lower().strip())

    if not mask.any():
        return None, f"'{food_name}' not found in food database."

    row = db[mask].iloc[0]

    taste = row["Taste"]
    compound = row["Compound"]
    alt = row["Healthy_Alternative"]
    notes = row["Notes"]

    profile = DEFAULT_FOOD_PROFILES.get(taste, DEFAULT_FOOD_PROFILES["Other"])

    result = {
        "Food_Item": food_name,
        "Compound": compound,
        "Bitter": profile["Bitter"],
        "Sweet": profile["Sweet"],
        "Umami": profile["Umami"],
        "Other": profile["Other"],
        "Dominant": taste,
        "Healthy_Alternative": alt,
        "Notes": notes,
    }
    return result, None

# ---------------------------------------------------------------------
# ---------------------- CHEMICAL MODE (UNTOUCHED) --------------------
# ---------------------------------------------------------------------
DB_PATHS = [
    os.path.join(os.getcwd(), "food_compounds_db.csv"),
    os.path.join(os.getcwd(), "data", "ExternalDBs", "food_compounds_db.csv"),
    r"C:\projom\VirtuousMultiTaste\data\ExternalDBs\food_compounds_db.csv",
]

@st.cache_data
def load_compound_db():
    for path in DB_PATHS:
        if os.path.exists(path):
            return pd.read_csv(path)
    return None

def dummy_predict(smiles: str):
    np.random.seed(len(smiles))
    return {
        "Bitter": round(np.random.rand(), 2),
        "Sweet": round(np.random.rand(), 2),
        "Umami": round(np.random.rand(), 2),
        "Other": round(np.random.rand(), 2)
    }, max(["Bitter", "Sweet", "Umami", "Other"], key=lambda x: random.random())

def get_ai_suggestion(dominant: str, db: pd.DataFrame, detailed=False):
    if db is None or db.empty:
        return "No database available."
    matches = db[db["Taste"].str.lower() == dominant.lower()]
    if matches.empty:
        return "No suggestions available."
    row = matches.sample(1).iloc[0]
    return f"{row.get('Compound','Unknown')} ({row.get('Notes','No notes')})"

# ---------------------------------------------------------------------
# MAIN UI
# ---------------------------------------------------------------------
def main():

    st.title("🍽️ AI Driven Multi-Taste Sensation Predictor")
    st.markdown("---")

    mode = st.sidebar.radio("Mode:", ["Compound (SMILES)", "Food Lookup (DB-first)"])
    page = st.sidebar.radio("Choose Page:", ["Input & Table", "Graph Visualization", "AI Suggestions"])

    food_db = load_food_db()
    chem_db = load_compound_db()

    results = []

    # ---------------------------------------------------------------------
    # FOOD MODE
    # ---------------------------------------------------------------------
    if mode == "Food Lookup (DB-first)":
        st.subheader("🥗 Food Lookup Mode")

        food_input = st.sidebar.text_area("Enter Food Items", height=180)
        food_list = [x.strip() for x in food_input.split("\n") if x.strip()] if food_input else []

        run_button = st.sidebar.button("🔍 Lookup Taste")

        if run_button and food_list:
            for f in food_list:
                res, err = predict_food_item(f, food_db)
                if err:
                    st.error(err)
                else:
                    results.append(res)
            st.session_state["food_results"] = pd.DataFrame(results)

        df_results = st.session_state.get("food_results", pd.DataFrame())

    # ---------------------------------------------------------------------
    # CHEMICAL MODE
    # ---------------------------------------------------------------------
    else:
        st.subheader("🧪 Chemical Compound Mode")

        smiles_input = st.sidebar.text_area("Enter SMILES", height=180)
        smiles_list = [x.strip() for x in smiles_input.split("\n") if x.strip()] if smiles_input else []

        run_button = st.sidebar.button("🔮 Predict Taste")

        if run_button and smiles_list:
            for smi in smiles_list:
                scores, dom = dummy_predict(smi)
                results.append({
                    "SMILES": smi,
                    "Bitter": scores["Bitter"],
                    "Sweet": scores["Sweet"],
                    "Umami": scores["Umami"],
                    "Other": scores["Other"],
                    "Dominant": dom,
                    "Suggestion": get_ai_suggestion(dom, chem_db)
                })
            st.session_state["chem_results"] = pd.DataFrame(results)

        df_results = st.session_state.get("chem_results", pd.DataFrame())

    # ---------------------------------------------------------------------
    # INPUT & TABLE VIEW
    # ---------------------------------------------------------------------
    if page == "Input & Table":
        st.subheader("📋 Results")
        st.dataframe(df_results, use_container_width=True)

    # ---------------------------------------------------------------------
    # GRAPH VIEW
    # ---------------------------------------------------------------------
    elif page == "Graph Visualization":
        if df_results.empty:
            st.info("Run prediction first.")
        else:
            tastes = ["Bitter", "Sweet", "Umami", "Other"]
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(df_results))
            width = 0.18
            for i, t in enumerate(tastes):
                ax.bar(x + i*width, df_results[t], width, label=t)
            ax.set_xticks(x)
            ax.set_xticklabels(df_results.iloc[:, 0])
            ax.legend()
            st.pyplot(fig)

    # ---------------------------------------------------------------------
    # AI SUGGESTIONS PAGE (CLASSIC UI)
    # ---------------------------------------------------------------------
    elif page == "AI Suggestions":
        st.subheader("✨ AI Suggestions (Classic View)")

        if df_results.empty:
            st.info("Run prediction first.")
        else:
            colors = {
                "Bitter": "#E74C3C",
                "Sweet": "#E91E63",
                "Umami": "#27AE60",
                "Other": "#5C6BC0"
            }

            for _, row in df_results.iterrows():

                if "Suggestion" in df_results.columns:
                    # Chemical mode
                    title = row["SMILES"]
                    taste = row["Dominant"]
                    suggestion_text = row["Suggestion"]
                else:
                    # Food mode
                    title = row["Food_Item"]
                    taste = row["Dominant"]
                    suggestion_text = f"{row['Healthy_Alternative']} — {row['Notes']}"

                color = colors.get(taste, "#333")

                st.markdown(
                    f"""
                    <div style="
                        background:#fff;
                        padding:18px 22px;
                        border-radius:12px;
                        margin-bottom:18px;
                        box-shadow:0 4px 12px rgba(0,0,0,0.08);
                        border-left:6px solid {color};
                        font-family:'Segoe UI',sans-serif;
                    ">
                        <div style="font-size:19px; font-weight:600; color:{color};">
                            {title}
                        </div>
                        <div style="font-size:15px; margin-top:6px;">
                            <b>Taste:</b>
                            <span style="color:{color}; font-weight:600;">{taste}</span>
                        </div>
                        <div style="margin-top:10px; font-size:14px; color:#444;">
                            {suggestion_text}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
