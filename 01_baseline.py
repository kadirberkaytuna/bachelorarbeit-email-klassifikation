"""Schritt 1: Klassische Baseline (TF-IDF + Naive Bayes / logistische Regression)."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

SEED = 42
PRO_KATEGORIE = 300  # max. Tickets pro Kategorie, damit die Klassen ausgeglichen sind

# 1. Daten laden und ausbalancieren
df = pd.read_csv("data/tickets_de.csv")
df["Eingabe"] = df["Betreff"].fillna("") + " " + df["Text"].fillna("")
df = df.groupby("Kategorie").sample(n=PRO_KATEGORIE, random_state=SEED)

# 2. Train/Test-Split (80/20, Kategorien gleich verteilt)
train, test = train_test_split(df, test_size=0.2, stratify=df["Kategorie"], random_state=SEED)
test.to_csv("data/testset.csv", index=False)  # dasselbe Testset nutzt später das LLM
train.to_csv("data/trainset.csv", index=False)
print(f"Training: {len(train)} Tickets, Test: {len(test)} Tickets\n")

# 3. Modelle trainieren und auswerten
modelle = {
    "Naive Bayes": MultinomialNB(),
    "Logistische Regression": LogisticRegression(max_iter=1000),
}
for name, modell in modelle.items():
    pipeline = make_pipeline(TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2), modell)
    pipeline.fit(train["Eingabe"], train["Kategorie"])
    vorhersage = pipeline.predict(test["Eingabe"])
    print(f"===== {name} =====")
    print(classification_report(test["Kategorie"], vorhersage, digits=3, zero_division=0))

# Einzelne E-Mail klassifizieren (letztes Modell = logistische Regression)
beispiel = "Hallo, auf meiner Rechnung Nr. 4711 ist der Betrag zu hoch. Bitte prüfen Sie das."
print("Vorhersage:", pipeline.predict([beispiel])[0])

wahrscheinlichkeiten = pipeline.predict_proba([beispiel])[0]
for kategorie, p in zip(pipeline.classes_, wahrscheinlichkeiten):
    print(f"  {kategorie}: {p:.0%}")