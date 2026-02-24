import joblib
import pandas as pd
import streamlit as st
from pathlib import Path
import numpy as np  # <-- TOEGEVOEGD: nodig voor berekeningen

# App titel
st.title("Hartziekte voorspeller")
st.write("Vul patiëntwaarden in. De app geeft een kans en een advies (doorverwijzen ja/nee).")

# Laad model
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "heart_model.joblib"
clf = joblib.load(MODEL_PATH)

# Instelbare drempel (threshold)
threshold = st.slider("Drempel voor doorverwijzen (kans >= drempel)", 0.0, 1.0, 0.50, 0.01)

st.subheader("Patiëntgegevens")

# Invoer velden (numeriek)
age = st.number_input("Age (jaren)", min_value=0, max_value=120, value=55)
restingbp = st.number_input("RestingBP (mm Hg)", min_value=0, max_value=300, value=145)
chol = st.number_input("Cholesterol (mg/dl)", min_value=0, max_value=1000, value=220)
maxhr = st.number_input("MaxHR", min_value=60, max_value=202, value=140)
oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.2, step=0.1)

fastingbs = st.selectbox("FastingBS > 120 mg/dl?", options=[0, 1], index=0)
restingecg = st.selectbox("RestingECG", options=[0, 1, 2], index=0)

# Invoer velden (categorisch)
sex = st.selectbox("Sex", options=["M", "F"], index=0)
cpt = st.selectbox("ChestPainType", options=["TA", "ATA", "NAP", "ASY"], index=1)
exang = st.selectbox("ExerciseAngina", options=["N", "Y"], index=0)
stslope = st.selectbox("ST_Slope", options=["Up", "Flat", "Down"], index=1)


# ============================================================
# TOEGEVOEGD: helperfunctie voor uitleg (top-5 bijdragen)
# ============================================================
def explain_top_factors(pipeline, X_one_row: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Maak uitleg op basis van LogisticRegression-coëfficiënten.
    We berekenen de bijdrage per (getransformeerde) feature en aggregeren dit terug
    naar de originele input-velden (Age, RestingBP, Sex, etc.).
    Output: DataFrame met top N factoren (grootste absolute bijdrage).
    """
    # Verwacht: pipeline met stappen 'preprocessor' en 'model'
    pre = pipeline.named_steps.get("preprocessor", None)
    model = pipeline.named_steps.get("model", None)

    if pre is None or model is None:
        raise ValueError("Pipeline mist 'preprocessor' of 'model' step.")

    # Alleen geschikt voor LogisticRegression met coef_
    if not hasattr(model, "coef_"):
        raise ValueError("Model heeft geen coef_. Deze uitleg werkt voor LogisticRegression-achtige modellen.")

    # 1) Transformeer de patiënt net zoals het model dat doet
    Xt = pre.transform(X_one_row)

    # 2) Feature-namen na preprocessing ophalen
    #    - numeric: blijft 1-op-1
    #    - categorical: wordt one-hot kolommen
    feature_names = pre.get_feature_names_out()

    # 3) Bepaal bijdragen per getransformeerde feature: x_i * w_i (log-odds bijdrage)
    coef = model.coef_.ravel()
    contrib = Xt.toarray().ravel() * coef if hasattr(Xt, "toarray") else np.asarray(Xt).ravel() * coef

    # 4) Zet in DataFrame op “getransformeerd feature” niveau
    df_feat = pd.DataFrame({
        "transformed_feature": feature_names,
        "contribution_logodds": contrib
    })

    # 5) Map terug naar originele kolomnaam:
    #    scikit-learn feature names zien er vaak uit als:
    #    - "num__Age"
    #    - "cat__Sex_M"
    # We nemen het stuk na "__", en dan voor categorieën het eerste deel vóór "_" als basisnaam.
    def to_base_feature(name: str) -> str:
        after = name.split("__", 1)[-1]  # bv "Sex_M" of "Age"
        # Als het exact een numeric kolom is: "Age" etc.
        # Als het one-hot is: "Sex_M" -> basis = "Sex"
        if after in X_one_row.columns:
            return after
        # anders: pak tot eerste "_" (one-hot patroon)
        return after.split("_", 1)[0]

    df_feat["base_feature"] = df_feat["transformed_feature"].apply(to_base_feature)

    # 6) Aggegreer bijdragen per originele feature (som van one-hot bijdragen)
    df_base = (
        df_feat.groupby("base_feature", as_index=False)["contribution_logodds"]
        .sum()
    )
    df_base["abs_contribution"] = df_base["contribution_logodds"].abs()

    # 7) Sorteer op grootste impact en pak top N
    df_top = df_base.sort_values("abs_contribution", ascending=False).head(top_n).copy()

    # 8) Voeg patiëntwaarde toe voor leesbaarheid
    # (als feature niet direct kolom is — bijv. one-hot basis — dan pakken we de ruwe inputkolom)
    def get_patient_value(base_feature: str):
        return X_one_row.iloc[0].get(base_feature, None)

    df_top["patient_value"] = df_top["base_feature"].apply(get_patient_value)

    # 9) Richting voor uitleg (positief -> verhoogt risico, negatief -> verlaagt)
    df_top["richting"] = df_top["contribution_logodds"].apply(lambda v: "verhoogt risico" if v > 0 else "verlaagt risico")

    # Netjes ordenen
    df_top = df_top[["base_feature", "patient_value", "richting", "contribution_logodds"]]
    df_top.rename(columns={
        "base_feature": "Factor",
        "patient_value": "Waarde patiënt",
        "contribution_logodds": "Bijdrage (log-odds)"
    }, inplace=True)

    return df_top


# Knop
if st.button("Voorspel"):
    new_patient = pd.DataFrame([{
        "Age": age,
        "RestingBP": restingbp,
        "Cholesterol": chol,
        "MaxHR": maxhr,
        "Oldpeak": oldpeak,
        "FastingBS": fastingbs,
        "RestingECG": restingecg,
        "Sex": sex,
        "ChestPainType": cpt,
        "ExerciseAngina": exang,
        "ST_Slope": stslope
    }])

    prob = clf.predict_proba(new_patient)[0, 1]
    pred = int(prob >= threshold)

    st.markdown(f"### Kans op hartziekte: **{prob:.3f}**")
    if pred == 1:
        st.error("Advies: **Doorverwijzen** (hoge kans op hartziekte).")
    else:
        st.success("Advies: **Geen doorverwijzing** (lage kans op hartziekte).")

    # ============================================================
    # TOEGEVOEGD: Uitleg / top 5 factoren + hoe de kans ontstaat
    # ============================================================
    try:
        df_top5 = explain_top_factors(clf, new_patient, top_n=5)

        st.subheader("Uitleg: top 5 factoren die het meest meetellen")
        st.write(
            "Dit zijn de 5 input-velden die bij deze patiënt de grootste bijdrage leveren aan de uitkomst "
            "(positieve bijdrage = richting ‘meer risico’, negatieve bijdrage = richting ‘minder risico’)."
        )
        st.dataframe(df_top5, use_container_width=True)

        # Extra: laat zien hoe de kans wordt opgebouwd (logit -> sigmoid -> kans)
        pre = clf.named_steps["preprocessor"]
        model = clf.named_steps["model"]

        Xt = pre.transform(new_patient)
        x_vec = Xt.toarray().ravel() if hasattr(Xt, "toarray") else np.asarray(Xt).ravel()
        coef = model.coef_.ravel()
        intercept = float(model.intercept_.ravel()[0])

        logit = intercept + float(np.dot(x_vec, coef))
        prob_check = 1.0 / (1.0 + np.exp(-logit))

        st.subheader("Hoe komt het model tot deze kans?")
        st.write(
            "Het model maakt eerst van jouw invoer een interne rekensom (log-odds). "
            "Die rekensom wordt daarna omgezet naar een kans (0–1)."
        )
        st.markdown(
            f"- **Basiswaarde (intercept)**: `{intercept:.3f}`\n"
            f"- **Som van alle bijdragen**: `{(logit - intercept):.3f}`\n"
            f"- **Totaal (log-odds)**: `{logit:.3f}`\n"
            f"- **Omgezet naar kans**: `{prob_check:.3f}`\n"
            f"- **Drempel voor doorverwijzen**: `{threshold:.2f}`"
        )

    except Exception as e:
        st.warning(f"Uitleg kon niet worden berekend: {e}")