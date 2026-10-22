"""Schritt 1: Klassische Baseline (TF-IDF + Naive Bayes / logistische Regression)."""
import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

# Startwert (Seed) für den Zufallsgenerator: Mit demselben Seed werden bei jeder Ausführung
# dieselben Tickets gezogen und gleich aufgeteilt. Dadurch sind die Ergebnisse reproduzierbar.
SEED = 42
PRO_KATEGORIE = 300  # max. Tickets pro Kategorie, damit die Klassen ausgeglichen sind

# 1. Daten laden und ausbalancieren
df = pd.read_csv("data/tickets_de.csv")
df["Eingabe"] = df["Betreff"].fillna("") + " " + df["Text"].fillna("")
# Je Kategorie zufällig 300 Tickets ziehen, random_state=SEED sorgt für immer dieselbe Auswahl
df = df.groupby("Kategorie").sample(n=PRO_KATEGORIE, random_state=SEED)

# 2. Train/Test-Split (80/20, Kategorien gleich verteilt)
# stratify: jede Kategorie ist in Training und Test im gleichen Verhältnis vertreten
# random_state=SEED: bei jeder Ausführung dieselbe Aufteilung
train, test = train_test_split(df, test_size=0.2, stratify=df["Kategorie"], random_state=SEED)
test.to_csv("data/testset.csv", index=False, lineterminator="\n")  # dasselbe Testset nutzt später das LLM
train.to_csv("data/trainset.csv", index=False, lineterminator="\n")
print(f"Training: {len(train)} Tickets, Test: {len(test)} Tickets\n")

# 3. Modelle trainieren, auswerten und Vorhersagen speichern
modelle = {
    "Naive Bayes": MultinomialNB(),
    "Logistische Regression": LogisticRegression(max_iter=1000),
}
ergebnisse = test[["ID", "Kategorie"]].rename(columns={"Kategorie": "Label"})
for name, modell in modelle.items():
    pipeline = make_pipeline(TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2), modell)
    start = time.perf_counter()
    pipeline.fit(train["Eingabe"], train["Kategorie"])
    trainingszeit = time.perf_counter() - start

    # Jedes Testticket einzeln klassifizieren und die Zeit messen (wie beim LLM)
    vorhersage, sekunden = [], []
    for text in test["Eingabe"]:
        start = time.perf_counter()
        vorhersage.append(pipeline.predict([text])[0])
        sekunden.append(round(time.perf_counter() - start, 6))
    ergebnisse[name] = vorhersage
    ergebnisse[f"{name} Sekunden"] = sekunden

    print(f"===== {name} =====")
    print(classification_report(test["Kategorie"], vorhersage, digits=3, zero_division=0))
    print(f"Trainingszeit: {trainingszeit:.2f} s")
    print(f"Ø Vorhersagezeit: {sum(sekunden) / len(sekunden) * 1000:.2f} ms pro E-Mail\n")

os.makedirs("results", exist_ok=True)
ergebnisse.to_csv("results/baseline.csv", index=False, lineterminator="\n")
print("Vorhersagen gespeichert in results/baseline.csv\n")

# Einzelne E-Mail klassifizieren (letztes Modell = logistische Regression)
beispiel = "Hallo, auf meiner Rechnung Nr. 4711 ist der Betrag zu hoch. Bitte prüfen Sie das."
print("Vorhersage:", pipeline.predict([beispiel])[0])

wahrscheinlichkeiten = pipeline.predict_proba([beispiel])[0]
for kategorie, p in zip(pipeline.classes_, wahrscheinlichkeiten):
    print(f"  {kategorie}: {p:.0%}")