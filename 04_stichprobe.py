"""Schritt 4: Zusatzauswertung auf den 100 manuell geprüften Tickets (Stichprobe).

Die Kennzahlen aus Schritt 1 und 2 messen die Übereinstimmung mit den Labels des Datensatzes.
Hier wird zusätzlich gegen die manuell geprüfte Kategorie ("Manuell") ausgewertet.
- Naive Bayes und logistische Regression werden wie in 01_baseline.py trainiert, aber ohne die
  geprüften Tickets, damit kein geprüftes Ticket im Training vorkommt.
- Das LLM klassifiziert die 100 Tickets mit demselben Prompt und denselben Einstellungen wie in
  02_llm_zero_shot.py. Die Antworten werden in results/stichprobe_llm.csv gespeichert. Existiert
  die Datei bereits, wird die API nicht erneut aufgerufen.
"""
import json, os, time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix

KATEGORIEN = ["Rechnung/Zahlung", "Reklamation/Rückgabe", "Vertriebsanfrage",
              "Kundenservice", "Technischer Support"]
LLM_DATEI = "results/stichprobe_llm.csv"

# 1. Daten laden: geprüfte Tickets mit Betreff und Text verbinden
stichprobe = pd.read_csv("data/stichprobe_100.csv")
tickets = pd.read_csv("data/tickets_de.csv")[["ID", "Betreff", "Text"]]
stichprobe = stichprobe.merge(tickets, on="ID", how="left")
stichprobe[["Betreff", "Text"]] = stichprobe[["Betreff", "Text"]].fillna("")
stichprobe["Eingabe"] = stichprobe["Betreff"] + " " + stichprobe["Text"]  # wie in 01_baseline.py

train = pd.read_csv("data/trainset.csv")
entfernt = train["ID"].isin(stichprobe["ID"]).sum()
train = train[~train["ID"].isin(stichprobe["ID"])]
test = pd.read_csv("data/testset.csv")
print(f"Geprüfte Tickets: {len(stichprobe)}, davon im Testset: {stichprobe['ID'].isin(test['ID']).sum()}, "
      f"im Trainingsset: {entfernt}")
print(f"Training ohne geprüfte Tickets: {len(train)} Tickets ({entfernt} entfernt)\n")

# 2. Klassische Verfahren (gleiche Einstellungen wie in 01_baseline.py)
modelle = {
    "Naive Bayes": MultinomialNB(),
    "Logistische Regression": LogisticRegression(max_iter=1000),
}
for name, modell in modelle.items():
    pipeline = make_pipeline(TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2), modell)
    pipeline.fit(train["Eingabe"], train["Kategorie"])
    stichprobe[name] = pipeline.predict(stichprobe["Eingabe"])

# 3. LLM (gleicher Prompt und gleiche Einstellungen wie in 02_llm_zero_shot.py)
if os.path.exists(LLM_DATEI):
    print(f"LLM-Antworten aus {LLM_DATEI} geladen (kein neuer API-Aufruf)\n")
    llm = pd.read_csv(LLM_DATEI)
else:
    from openai import OpenAI
    MODELL = "gpt-6-luna"
    SYSTEM_PROMPT = (
        "Du bist ein System zur Klassifikation von Kunden-E-Mails eines Unternehmens. "
        f"Ordne die E-Mail genau einer dieser Kategorien zu: {', '.join(KATEGORIEN)}. "
        'Antworte ausschließlich als JSON im Format {"kategorie": "<Kategorie>"}.'
    )
    client = OpenAI()  # liest den API-Schlüssel aus der Umgebungsvariable OPENAI_API_KEY
    zeilen = []
    for _, ticket in stichprobe.iterrows():
        start = time.perf_counter()
        antwort = client.chat.completions.create(
            model=MODELL,
            reasoning_effort="none",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Betreff: {ticket['Betreff']}\n\n{ticket['Text']}"},
            ],
        )
        sekunden = time.perf_counter() - start
        rohantwort = antwort.choices[0].message.content
        try:
            kategorie = json.loads(rohantwort)["kategorie"]
        except (json.JSONDecodeError, KeyError, TypeError):
            kategorie = "UNGÜLTIG"
        if kategorie not in KATEGORIEN:
            kategorie = "UNGÜLTIG"
        zeilen.append({
            "ID": ticket["ID"],
            "Vorhersage": kategorie,
            "Rohantwort": rohantwort,
            "Modell": antwort.model,
            "Input_Tokens": antwort.usage.prompt_tokens,
            "Output_Tokens": antwort.usage.completion_tokens,
            "Sekunden": round(sekunden, 2),
        })
        print(f"{ticket['ID']}: {kategorie}")
    llm = pd.DataFrame(zeilen)
    os.makedirs("results", exist_ok=True)
    llm.to_csv(LLM_DATEI, index=False, lineterminator="\n")
    print(f"\nLLM-Antworten gespeichert in {LLM_DATEI} (Modell: {', '.join(llm['Modell'].unique())})\n")

stichprobe = stichprobe.merge(llm[["ID", "Vorhersage"]].rename(columns={"Vorhersage": "LLM"}), on="ID")
verfahren = ["Naive Bayes", "Logistische Regression", "LLM"]

# 4. Auswertung
label_falsch = stichprobe["Label"] != stichprobe["Manuell"]
print("===== Genauigkeit auf den 100 geprüften Tickets =====")
print(f"{'Verfahren':24s} {'gegen Label':>12s} {'gegen Manuell':>14s}")
for name in verfahren:
    print(f"{name:24s} {accuracy_score(stichprobe['Label'], stichprobe[name]):12.1%} "
          f"{accuracy_score(stichprobe['Manuell'], stichprobe[name]):14.1%}")
print(f"Ungültige LLM-Antworten: {(stichprobe['LLM'] == 'UNGÜLTIG').sum()} von {len(stichprobe)}")
# Vergleichswert: Die manuellen Kategorien sind nicht gleich verteilt. Wer immer die häufigste wählt,
# erreicht bereits diesen Anteil.
haeufigste = stichprobe["Manuell"].value_counts()
print(f"Vergleich: immer '{haeufigste.index[0]}' wählen = {haeufigste.iloc[0] / len(stichprobe):.1%} gegen Manuell")
print(f"Verteilung Manuell: {haeufigste.to_dict()}")

print(f"\n===== Tickets mit falschem Label ({label_falsch.sum()}) =====")
print("Folgt die Vorhersage dem Label des Datensatzes oder dem Inhalt (manuelle Kategorie)?")
falsch = stichprobe[label_falsch]
for name in verfahren:
    print(f"{name:24s} wie Label: {(falsch[name] == falsch['Label']).sum():3d}   "
          f"wie Inhalt: {(falsch[name] == falsch['Manuell']).sum():3d}   "
          f"weder noch: {((falsch[name] != falsch['Label']) & (falsch[name] != falsch['Manuell'])).sum():3d}")

richtig = stichprobe[~label_falsch]
print(f"\n===== Tickets mit korrektem Label ({len(richtig)}) =====")
for name in verfahren:
    print(f"{name:24s} richtig: {(richtig[name] == richtig['Label']).sum():3d} von {len(richtig)}")

print("\n===== Konfusionsmatrizen gegen die manuelle Kategorie =====")
print("Zeilen = manuelle Kategorie, Spalten = Vorhersage")
for name in verfahren:
    matrix = pd.DataFrame(confusion_matrix(stichprobe["Manuell"], stichprobe[name], labels=KATEGORIEN),
                          index=KATEGORIEN, columns=KATEGORIEN)
    print(f"\n{name}\n{matrix.to_string()}")

# Wiederholbarkeit LLM: Tickets, die auch im Lauf auf den 300 Testtickets (Schritt 2) enthalten waren
if os.path.exists("results/zero_shot.csv"):
    alt = pd.read_csv("results/zero_shot.csv")[["ID", "Vorhersage"]]
    vergleich = stichprobe[["ID", "LLM"]].merge(alt, on="ID")
    if len(vergleich):
        gleich = (vergleich["LLM"] == vergleich["Vorhersage"]).sum()
        print(f"\nLLM-Wiederholung: {gleich} von {len(vergleich)} Tickets erhielten dieselbe Kategorie "
              f"wie im Lauf aus Schritt 2")

stichprobe[["Nr", "ID", "Label", "Manuell"] + verfahren].to_csv(
    "results/stichprobe.csv", index=False, lineterminator="\n")
print("\nVorhersagen gespeichert in results/stichprobe.csv")
