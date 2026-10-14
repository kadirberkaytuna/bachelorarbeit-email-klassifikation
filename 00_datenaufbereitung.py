"""Schritt 0: Datenaufbereitung - aus dem Hugging-Face-Datensatz die Datei data/tickets_de.csv erzeugen.

Quelle: Tobi-Bueck/customer-support-tickets (Hugging Face), DOI 10.57967/hf/6184, Lizenz CC BY-NC 4.0.
Verwendet wird die Datei aa_dataset-tickets-multi-lang-5-2-50-version.csv in einer festen Revision,
damit spätere Änderungen am Datensatz das Ergebnis nicht verändern.
"""
import html
import re
import pandas as pd

REVISION = "ddf1c81a5475992c4fa6752bf1e8b4e31f07bbeb"  # Stand des Datensatzes vom 28.06.2026
QUELLE = ("https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets/resolve/"
          f"{REVISION}/aa_dataset-tickets-multi-lang-5-2-50-version.csv")

# Zuordnung der Original-Queues zu den fünf Kategorien dieser Arbeit.
# Die übrigen Queues (Service Outages, Human Resources, General Inquiry) werden nicht verwendet.
ZUORDNUNG = {
    "Billing and Payments": "Rechnung/Zahlung",
    "Returns and Exchanges": "Reklamation/Rückgabe",
    "Sales and Pre-Sales": "Vertriebsanfrage",
    "Customer Service": "Kundenservice",
    "Technical Support": "Technischer Support",
    "Product Support": "Technischer Support",
    "IT Support": "Technischer Support",
}

# Kleine Stoppwortlisten, um als "de" markierte, aber englische Tickets zu erkennen
DEUTSCH = set("und der die das ich wir sie ist nicht mit für bitte ein eine zu von den auf ihre "
              "unsere können".split())
ENGLISCH = set("the and is to we our please you for with of a in are have this".split())


def ist_deutsch(text):
    """Ein Text gilt als deutsch, wenn er mehr deutsche als englische Stoppwörter enthält."""
    woerter = re.findall(r"[a-zäöüß]+", str(text).lower())
    return sum(w in DEUTSCH for w in woerter) > sum(w in ENGLISCH for w in woerter)


def bereinigen(text):
    """Entfernt HTML-Zeilenumbrüche und HTML-Zeichen und vereinheitlicht Zeilenumbrüche."""
    text = "" if pd.isna(text) else str(text)
    text = re.sub(r"<br\s*/?>", "\n", text)
    return html.unescape(text).replace("\\n", "\n").strip()


# 1. Rohdaten laden (28.587 Tickets in mehreren Sprachen)
df = pd.read_csv(QUELLE)
df["ID"] = df.index  # Zeilennummer in der Quelldatei als eindeutige ID
print(f"Quelldatei: {len(df)} Tickets")

# 2. Nur deutschsprachige Tickets: Sprachfeld "de" und tatsächlich deutscher Text
de = df[df["language"] == "de"]
print(f"Als deutsch markiert: {len(de)}")
de = de[de["body"].map(ist_deutsch)]
print(f"Tatsächlich deutsch: {len(de)}")

# 3. Nur die sieben verwendeten Queues, abgebildet auf fünf Kategorien
de = de[de["queue"].isin(ZUORDNUNG)].copy()
de["Kategorie"] = de["queue"].map(ZUORDNUNG)

# 4. Texte bereinigen und speichern
de["Betreff"] = de["subject"].map(bereinigen)
de["Text"] = de["body"].map(bereinigen)
de = de.sort_values(["Kategorie", "ID"])
de[["ID", "Kategorie", "Betreff", "Text"]].to_csv("data/tickets_de.csv", index=False)

print(f"Gespeichert: {len(de)} Tickets in data/tickets_de.csv\n")
print(de["Kategorie"].value_counts().to_string())