"""Schritt 2: Zero-Shot-Klassifikation mit der OpenAI API auf demselben Testset."""
import json, os, time
import pandas as pd
from openai import OpenAI
from sklearn.metrics import accuracy_score, classification_report

MODELL = "gpt-6-luna"    # Modellwahl siehe Kapitel 4.3
ANZAHL = 300             # alle Testtickets; zum Ausprobieren kleiner setzen, z. B. 50
KATEGORIEN = ["Rechnung/Zahlung", "Reklamation/Rückgabe", "Vertriebsanfrage",
              "Kundenservice", "Technischer Support"]

SYSTEM_PROMPT = (
    "Du bist ein System zur Klassifikation von Kunden-E-Mails eines Unternehmens. "
    f"Ordne die E-Mail genau einer dieser Kategorien zu: {', '.join(KATEGORIEN)}. "
    'Antworte ausschließlich als JSON im Format {"kategorie": "<Kategorie>"}.'
)

client = OpenAI()  # liest den API-Schlüssel aus der Umgebungsvariable OPENAI_API_KEY
test = pd.read_csv("data/testset.csv").head(ANZAHL)
# Leere Felder als leeren Text senden, nicht als "nan" (wie in 01_baseline.py)
test[["Betreff", "Text"]] = test[["Betreff", "Text"]].fillna("")

ergebnisse = []
for _, ticket in test.iterrows():
    start = time.perf_counter()
    antwort = client.chat.completions.create(
        model=MODELL,
        reasoning_effort="none",  # kein Reasoning: schneller, günstiger, besser reproduzierbar
        temperature=0,            # falls das Modell den Parameter ablehnt: Zeile entfernen
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
    if kategorie not in KATEGORIEN:  # gültiges JSON, aber keine der fünf Kategorien
        kategorie = "UNGÜLTIG"
    ergebnisse.append({
        "ID": ticket["ID"],
        "Label": ticket["Kategorie"],
        "Vorhersage": kategorie,
        "Rohantwort": rohantwort,
        "Modell": antwort.model,
        "Input_Tokens": antwort.usage.prompt_tokens,
        "Output_Tokens": antwort.usage.completion_tokens,
        "Sekunden": round(sekunden, 2),
    })
    print(f"{ticket['ID']}: {kategorie}")

df = pd.DataFrame(ergebnisse)
os.makedirs("results", exist_ok=True)
df.to_csv("results/zero_shot.csv", index=False, lineterminator="\n")

print("\n===== Zero-Shot =====")
print(f"Modell: {', '.join(df['Modell'].unique())}")
print(f"Accuracy: {accuracy_score(df['Label'], df['Vorhersage']):.3f}")
print(f"Ungültige Antworten: {(df['Vorhersage'] == 'UNGÜLTIG').sum()} von {len(df)}")
# labels=KATEGORIEN: "UNGÜLTIG" zählt als Fehler, nicht als sechste Kategorie
print(classification_report(df["Label"], df["Vorhersage"], labels=KATEGORIEN, digits=3, zero_division=0))
print(f"Tokens gesamt: {df['Input_Tokens'].sum()} Input, {df['Output_Tokens'].sum()} Output")
print(f"Ø Antwortzeit: {df['Sekunden'].mean():.2f} s pro E-Mail")