"""Schritt 5: Auswertung - berechnet die Kennzahlen für Kapitel 6 aus den gespeicherten Ergebnisdateien.

Es wird nichts neu trainiert und keine API aufgerufen. Grundlage sind:
- results/baseline.csv und results/zero_shot.csv (Schritte 1 und 2, 300 Testtickets)
- data/stichprobe_100.csv und results/stichprobe.csv (manuelle Prüfung und Schritt 4)
Optional: Prüfung des Sprachfilters mit lingua (pip install lingua-language-detector==2.2.0).
Dafür wird die Quelldatei erneut von Hugging Face geladen.
"""
import math
import re
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

KATEGORIEN = ["Rechnung/Zahlung", "Reklamation/Rückgabe", "Vertriebsanfrage",
              "Kundenservice", "Technischer Support"]
KURZ = {"Rechnung/Zahlung": "RZ", "Reklamation/Rückgabe": "RR", "Vertriebsanfrage": "VA",
        "Kundenservice": "KS", "Technischer Support": "TS"}
PREIS_EINGABE = 0.10  # US-Dollar pro 1 Mio. Eingabetokens, GPT-6 Luna, Stand 14.10.2026
PREIS_AUSGABE = 0.50  # US-Dollar pro 1 Mio. Ausgabetokens


def matrix(label, vorhersage):
    """Konfusionsmatrix als Tabelle: Zeilen = Label, Spalten = Vorhersage."""
    werte = confusion_matrix(label, vorhersage, labels=KATEGORIEN)
    namen = [KURZ[k] for k in KATEGORIEN]
    return pd.DataFrame(werte, index=namen, columns=namen)


def mcnemar(a_richtig, b_richtig):
    """Exakter McNemar-Test: vergleicht zwei Verfahren auf denselben Tickets."""
    nur_a = int((a_richtig & ~b_richtig).sum())
    nur_b = int((~a_richtig & b_richtig).sum())
    n, k = nur_a + nur_b, min(nur_a, nur_b)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return nur_a, nur_b, p


def wilson(richtig, n, z=1.96):
    """95%-Konfidenzintervall für einen Anteil (Wilson)."""
    p = richtig / n
    mitte = (p + z * z / (2 * n)) / (1 + z * z / n)
    breite = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return mitte - breite, mitte + breite


baseline = pd.read_csv("results/baseline.csv")
llm = pd.read_csv("results/zero_shot.csv")
verfahren = {
    "Naive Bayes": (baseline["Label"], baseline["Naive Bayes"]),
    "Logistische Regression": (baseline["Label"], baseline["Logistische Regression"]),
    "GPT-6 Luna": (llm["Label"], llm["Vorhersage"]),
}

# ---------------------------------------------------------------- 6.1
print("===== 6.1 Klassifikationsgüte (300 Testtickets) =====\n")
print("Tabelle 6: Genauigkeit und Makro-Durchschnitt")
for name, (label, vorhersage) in verfahren.items():
    p, r, f, _ = precision_recall_fscore_support(label, vorhersage, labels=KATEGORIEN,
                                                 average="macro", zero_division=0)
    richtig = (label == vorhersage).sum()
    print(f"  {name:24s} Genauigkeit {accuracy_score(label, vorhersage):.1%} ({richtig} von {len(label)}), "
          f"Precision {p:.1%}, Recall {r:.1%}, F1 {f:.1%}")

print("\nTabelle 7 und Text: Precision / Recall / F1 je Kategorie")
for name, (label, vorhersage) in verfahren.items():
    p, r, f, _ = precision_recall_fscore_support(label, vorhersage, labels=KATEGORIEN, zero_division=0)
    print(f"  {name}")
    for k, pk, rk, fk in zip(KATEGORIEN, p, r, f):
        print(f"    {k:22s} {pk:6.1%} {rk:6.1%} {fk:6.1%}")

print("\nAnzahl der Vorhersagen je Kategorie (je Kategorie 60 Testtickets)")
for name, (_, vorhersage) in verfahren.items():
    anzahl = vorhersage.value_counts().reindex(KATEGORIEN, fill_value=0)
    print(f"  {name:24s} " + ", ".join(f"{KURZ[k]} {n}" for k, n in anzahl.items()))

print("\nZusatz: McNemar-Test (Tickets, die nur das erste bzw. nur das zweite Verfahren richtig hat)")
richtig = {name: (label == vorhersage) for name, (label, vorhersage) in verfahren.items()}
for a, b in [("Logistische Regression", "GPT-6 Luna"), ("Naive Bayes", "GPT-6 Luna"),
             ("Logistische Regression", "Naive Bayes")]:
    nur_a, nur_b, p = mcnemar(richtig[a], richtig[b])
    print(f"  {a} vs. {b}: {nur_a} zu {nur_b}, p = {p:.4f}")
for name, r in richtig.items():
    unten, oben = wilson(r.sum(), len(r))
    print(f"  95%-Konfidenzintervall Genauigkeit {name}: {unten:.1%} bis {oben:.1%}")

# ---------------------------------------------------------------- 6.2.1
print("\n===== 6.2.1 Konfusionsmatrizen (Zeilen = Label, Spalten = Vorhersage) =====")
for name, (label, vorhersage) in verfahren.items():
    abweichung = label != vorhersage
    print(f"\n{name}: {abweichung.sum()} Abweichungen vom Label")
    print(matrix(label, vorhersage).to_string())
    nach_vorhersage = vorhersage[abweichung].value_counts().reindex(KATEGORIEN, fill_value=0)
    print("  Abweichungen nach Vorhersage: " + ", ".join(f"{KURZ[k]} {n}" for k, n in nach_vorhersage.items()))

# ---------------------------------------------------------------- 6.2.2
print("\n===== 6.2.2 Qualität der Labels (100 manuell geprüfte Tickets) =====\n")
stichprobe = pd.read_csv("data/stichprobe_100.csv")
stichprobe["Korrekt"] = stichprobe["Label"] == stichprobe["Manuell"]
print("Tabelle 9")
for k in KATEGORIEN:
    teil = stichprobe[stichprobe["Label"] == k]
    stattdessen = teil.loc[~teil["Korrekt"], "Manuell"].value_counts()
    print(f"  {k:22s} {teil['Korrekt'].sum():2d} von {len(teil)} passen   "
          f"stattdessen: {', '.join(f'{m} {n}' for m, n in stattdessen.items())}")
print(f"  Summe: {stichprobe['Korrekt'].sum()} von {len(stichprobe)}")

falsch = stichprobe[~stichprobe["Korrekt"]]
grenzfaelle = (falsch["Grenzfall"] == "Ja").sum()
print(f"\nFalsche Labels: {len(falsch)}, davon Grenzfälle: {grenzfaelle}")

# Hochrechnung auf alle 8.110 Tickets: Anteil korrekter Labels je Kategorie,
# gewichtet mit dem Anteil der Kategorie im Datensatz
anteil = pd.read_csv("data/tickets_de.csv")["Kategorie"].value_counts(normalize=True)
gewichtet, mit_grenzfaellen = 0, 0
for k in KATEGORIEN:
    teil = stichprobe[stichprobe["Label"] == k]
    korrekt = teil["Korrekt"].sum()
    grenz = ((~teil["Korrekt"]) & (teil["Grenzfall"] == "Ja")).sum()
    gewichtet += anteil[k] * korrekt / len(teil)
    mit_grenzfaellen += anteil[k] * (korrekt + grenz) / len(teil)
print(f"Geschätzter Anteil korrekter Labels im Datensatz: {gewichtet:.0%} "
      f"(Grenzfälle als korrekt gezählt: {mit_grenzfaellen:.0%})")

try:
    from lingua import Language, LanguageDetectorBuilder
except ImportError:
    print("\nSprachfilter-Prüfung übersprungen (lingua nicht installiert)")
else:
    # Quelle und Stoppwortlisten wie in 00_datenaufbereitung.py
    QUELLE = ("https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets/resolve/"
              "ddf1c81a5475992c4fa6752bf1e8b4e31f07bbeb/aa_dataset-tickets-multi-lang-5-2-50-version.csv")
    DEUTSCH = set("und der die das ich wir sie ist nicht mit für bitte ein eine zu von den auf ihre "
                  "unsere können".split())
    ENGLISCH = set("the and is to we our please you for with of a in are have this".split())

    def ist_deutsch(text):
        woerter = re.findall(r"[a-zäöüß]+", str(text).lower())
        return sum(w in DEUTSCH for w in woerter) > sum(w in ENGLISCH for w in woerter)

    detektor = LanguageDetectorBuilder.from_languages(Language.GERMAN, Language.ENGLISH).build()
    quelle = pd.read_csv(QUELLE)
    de = quelle[quelle["language"] == "de"]
    heuristik = de["body"].map(ist_deutsch)
    lingua = de["body"].map(lambda t: detektor.detect_language_of(str(t)) == Language.GERMAN)
    print(f"\nSprachfilter: {len(de)} Tickets mit Sprachangabe 'de'")
    print(f"  laut lingua englisch: {(~lingua).sum()} ({(~lingua).mean():.0%})")
    print(f"  von der Heuristik behalten: {heuristik.sum()}, davon laut lingua deutsch: {(heuristik & lingua).sum()}")
    verworfen = de.loc[~heuristik & lingua, "body"]
    print(f"  deutsche Tickets, die die Heuristik verworfen hat: {len(verworfen)} "
          f"(Länge im Median {verworfen.str.split().str.len().median():.0f} Wörter, "
          f"alle Tickets: {de['body'].str.split().str.len().median():.0f} Wörter)")

# ---------------------------------------------------------------- 6.2.3
print("\n===== 6.2.3 Zusatzauswertung: Abweichungen des LLM vom Label (100 geprüfte Tickets) =====")
print("Ergebnisse von 04_stichprobe.py, hier aufgeschlüsselt nach Label und Vorhersage des LLM.")
print("'davon geprüft gleich' = die manuelle Prüfung ergab dieselbe Kategorie wie das LLM\n")
geprueft = pd.read_csv("results/stichprobe.csv")
abweichung = geprueft[geprueft["LLM"] != geprueft["Label"]]
for (label, vorhersage), teil in abweichung.groupby(["Label", "LLM"]):
    print(f"  Label {KURZ[label]} → LLM {KURZ[vorhersage]}: {len(teil):2d}, "
          f"davon geprüft gleich: {(teil['Manuell'] == vorhersage).sum()}")
print("\nZusatz: McNemar-Test und 95%-Konfidenzintervall gegen die geprüfte Kategorie")
richtig = {name: (geprueft[name] == geprueft["Manuell"])
           for name in ["Naive Bayes", "Logistische Regression", "LLM"]}
for a, b in [("LLM", "Logistische Regression"), ("LLM", "Naive Bayes"),
             ("Naive Bayes", "Logistische Regression")]:
    nur_a, nur_b, p = mcnemar(richtig[a], richtig[b])
    print(f"  {a} vs. {b}: {nur_a} zu {nur_b}, p = {p:.4f}")
for name, r in richtig.items():
    unten, oben = wilson(r.sum(), len(r))
    print(f"  {name}: {r.mean():.0%}, Konfidenzintervall {unten:.0%} bis {oben:.0%}")

# ---------------------------------------------------------------- 6.3
print("\n===== 6.3 Kosten und Antwortzeit (300 Testtickets) =====\n")
eingabe, ausgabe = llm["Input_Tokens"].sum(), llm["Output_Tokens"].sum()
kosten = eingabe * PREIS_EINGABE / 1e6 + ausgabe * PREIS_AUSGABE / 1e6
print(f"Tokens: {eingabe} Eingabe, {ausgabe} Ausgabe")
print(f"  pro E-Mail im Mittel: {eingabe / len(llm):.0f} Eingabe, {ausgabe / len(llm):.0f} Ausgabe "
      f"(Ausgabe zwischen {llm['Output_Tokens'].min()} und {llm['Output_Tokens'].max()})")
print(f"Kosten: {kosten:.4f} US-Dollar für {len(llm)} E-Mails, "
      f"{kosten / len(llm) * 1000:.3f} pro 1.000, {kosten / len(llm) * 10000:.2f} pro 10.000")

zeit = llm["Sekunden"]
print(f"\nGPT-6 Luna: Mittel {zeit.mean():.2f} s, Median {zeit.median():.2f} s, "
      f"Minimum {zeit.min():.2f} s, Maximum {zeit.max():.2f} s (ID {llm.loc[zeit.idxmax(), 'ID']}, "
      f"Position {zeit.idxmax() + 1} im Lauf)")
print(f"  Antworten über 5 Sekunden (NFA4): {(zeit > 5).sum()}")
for name in ["Naive Bayes", "Logistische Regression"]:
    ms = baseline[f"{name} Sekunden"] * 1000
    print(f"{name}: Mittel {ms.mean():.2f} ms, Maximum {ms.max():.2f} ms")
lr = baseline["Logistische Regression Sekunden"].mean()
print(f"GPT-6 Luna braucht im Mittel {zeit.mean() / lr:.0f}-mal so lange wie die logistische Regression")
print("Trainingszeiten: siehe Konsolenausgabe von 01_baseline.py (results/ausgabe_01_baseline.txt)")
