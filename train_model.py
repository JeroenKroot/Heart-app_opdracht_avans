# Bibliotheken importeren
# - `pandas` voor data-manipulatie
# - `joblib` om het getrainde model op te slaan en later te laden
import pandas as pd
import joblib

# Scikit-learn onderdelen:
# - `train_test_split` om data te splitsen in train/test
# - `ColumnTransformer`, `Pipeline` voor nette preprocessing + model
# - `SimpleImputer` voor ontbrekende waarden
# - `OneHotEncoder`, `StandardScaler` voor encoderen/schalen
# - `LogisticRegression` als model
# - `roc_auc_score`, `accuracy_score` om prestaties te meten
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score


# 1) Data laden (vanuit het opgeschoonde csv-bestand)
# - Lees de CSV met de dataset in een DataFrame
# - `y` is de doelvariabele (HeartDisease)
# - `X` bevat alle features, we droppen irrelevante of identificerende kolommen
df = pd.read_csv(r"C:\Users\jkroot\Python code JK\Heartfaillure app\1. opschoon file\Opgeschoonde_Heart faillure prediction data.csv")

y = df["HeartDisease"]
X = df.drop(columns=["HeartDisease"]) # hij neemt nu alle kolommen mee m.u.v. heart disease, want die is onze target variabele

# 2) Train/test split (80% train, 20% test)
# - `stratify=y` zorgt dat de verhouding van klassen behouden blijft in beide sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 3) Preprocessing pipelines voor numerieke en categorische features
# - We splitsen features op type zodat elke groep passende transformaties krijgt
numeric_features = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak", "FastingBS", "RestingECG"]
categorical_features = ["Sex", "ChestPainType", "ExerciseAngina", "ST_Slope"]

# Numerieke pipeline:
# 1) Impute ontbrekende waarden met de mediaan
# 2) Schaal features (mean=0, std=1)
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorische pipeline:
# 1) Vul ontbrekende waarden met de meest voorkomende waarde
# 2) One-hot encodeer categorische kolommen; onbekende categorieën negeren bij predictie
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# ColumnTransformer combineert beide pipelines en past ze alleen toe op de aangegeven kolommen
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# 4) Volledige pipeline: preprocessing gevolgd door het model
# - Door preprocessing in de pipeline te plaatsen, worden dezelfde stappen automatisch toegepast bij predictie
clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

# 5) Model trainen op de trainingsset
clf.fit(X_train, y_train)

# 6) Evaluatie op de testset (sanity check)
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]   # kans op klasse 1 (HeartDisease=1)

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("\n--- Model evaluatie ---")
print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"ROC-AUC  : {roc_auc:.4f}")
print("----------------------\n")
# we meten zowel de accuracy (hoe vaak we het goed hebben), recall ( hoe vaak we de positieve klasse goed herkennen), als de ROC-AUC (hoe goed de kans-voorspellingen zijn).

# 7) Model opslaan
# - Sla de hele pipeline op (preprocessing + model) zodat je later direct `predict` kunt draaien
joblib.dump(clf, "heart_model.joblib")
print("Model opgeslagen als: heart_model.joblib")
print("Script succesvol uitgevoerd!")
