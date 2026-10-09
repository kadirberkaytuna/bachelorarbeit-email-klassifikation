"""Schritt 2: Zero-Shot-Klassifikation mit der OpenAI API auf demselben Testset."""
import json, os, time
import pandas as pd
from openai import OpenAI
from sklearn.metrics import classification_report

MODELL = "gpt-4o-mini"   # vor dem Start ein aktuelles, günstiges Modell in der OpenAI-Doku prüfen
ANZAHL = 50              # erst mit wenigen Tickets testen, später auf len(test) erhöhen
KATEGORIEN = ["Rechnung/Zahlung", "Reklamation/Rückgabe", "Vertriebsanfrage",
              "Kundenservice", "Technischer Support"]

SYSTEM_PROMPT = (
    "Du bist ein System zur Klassifikation von Kunden-E-Mails eines Unternehmens. "
    f"Ordne die E-Mail genau einer dieser Kategorien zu: {', '.join(KATEGORIEN)}. "
    'Antworte ausschließlich als JSON im Format {"kategorie": "<Kategorie>"}.'
)

client = OpenAI()  # liest den API-Schlüssel aus der Umgebungsvariable OPENAI_API_KEY
test = pd.read_csv("data/testset.csv").head(ANZAHL)

ergebnisse = []
for _, ticket in test.iterrows():
    start = time.time()
    antwort = client.chat.completions.create(
        model=MODELL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Betreff: {ticket['Betreff']}\n\n{ticket['Text']}"},
        ],
    )
    try:
        kategorie = json.loads(antwort.choices[0].message.content)["kategorie"]
    except (json.JSONDecodeError, KeyError):
        kategorie = "UNGÜLTIG"
    ergebnisse.append({
        "ID": ticket["ID"],
        "Wahr": ticket["Kategorie"],
        "Vorhersage": kategorie,
        "Input_Tokens": antwort.usage.prompt_tokens,
        "Output_Tokens": antwort.usage.completion_tokens,
        "Sekunden": round(time.time() - start, 2),
    })
    print(f"{ticket['ID']}: {kategorie}")

df = pd.DataFrame(ergebnisse)
os.makedirs("results", exist_ok=True)
df.to_csv("results/zero_shot.csv", index=False)

print("\n===== Zero-Shot =====")
print(classification_report(df["Wahr"], df["Vorhersage"], digits=3, zero_division=0))
print(f"Tokens gesamt: {df['Input_Tokens'].sum()} Input, {df['Output_Tokens'].sum()} Output")
print(f"Ø Antwortzeit: {df['Sekunden'].mean():.2f} s pro E-Mail")
