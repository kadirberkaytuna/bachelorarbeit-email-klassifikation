"""Schritt 5: Auswertung - berechnet die Kennzahlen für Kapitel 6 aus den gespeicherten Ergebnisdateien.

Es wird nichts neu trainiert und keine API aufgerufen. Grundlage sind:
- results/baseline.csv und results/zero_shot.csv (Schritte 1 und 2, 300 Testtickets)
- data/stichprobe_100.csv und results/stichprobe.csv (manuelle Prüfung und Schritt 4)
Optional: Prüfung des Sprachfilters mit lingua (pip install lingua-language-detector==2.2.0).
Dafür wird die Quelldatei erneut von Hugging Face geladen.

Aufbau der Ausgabe (Abschnittsnummern wie in der Ausgabe):
- 6.1   Klassifikationsgüte: Tabelle 6 und 7, Anzahl der Vorhersagen je Kategorie
- 6.2.1 Konfusionsmatrizen der drei Verfahren (Tabelle 8 und Anhang)
- 6.2.2 Qualität der Labels: Tabelle 9, Hochrechnung auf den Datensatz, Prüfung des Sprachfilters
- 6.2.3 Abweichungen des LLM vom Label auf den 100 geprüften Tickets
- 6.3   Kosten und Antwortzeit (Tabelle 11)
Die mit "Zusatz" überschriebenen Teile (McNemar-Test, Konfidenzintervalle) dienen nur der
Einordnung der Ergebnisse und werden in der Arbeit nicht verwendet.
"""
import math  # Mathematische Funktionen (Binomialkoeffizient, Wurzel) für die Zusatzstatistik
import re  # Reguläre Ausdrücke, hier zum Zerlegen eines Textes in Wörter (Sprachfilter)
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

# Reihenfolge der Kategorien in allen Tabellen (wie in 02_llm_zero_shot.py)
KATEGORIEN = ["Rechnung/Zahlung", "Reklamation/Rückgabe", "Vertriebsanfrage",
              "Kundenservice", "Technischer Support"]
# Kurzformen für die Zeilen und Spalten der Konfusionsmatrizen
KURZ = {"Rechnung/Zahlung": "RZ", "Reklamation/Rückgabe": "RR", "Vertriebsanfrage": "VA",
        "Kundenservice": "KS", "Technischer Support": "TS"}
PREIS_EINGABE = 0.10  # US-Dollar pro 1 Mio. Eingabetokens, GPT-6 Luna, Stand 14.10.2026
PREIS_AUSGABE = 0.50  # US-Dollar pro 1 Mio. Ausgabetokens


def matrix(label, vorhersage):
    """Konfusionsmatrix als Tabelle: Zeilen = Label, Spalten = Vorhersage."""
    # confusion_matrix zählt, wie oft jedes Paar aus Label und Vorhersage vorkommt.
    # labels=KATEGORIEN legt die Reihenfolge der Zeilen und Spalten fest.
    werte = confusion_matrix(label, vorhersage, labels=KATEGORIEN)
    namen = [KURZ[k] for k in KATEGORIEN]
    return pd.DataFrame(werte, index=namen, columns=namen)


def mcnemar(a_richtig, b_richtig):
    """Exakter McNemar-Test: vergleicht zwei Verfahren auf denselben Tickets."""
    # a_richtig und b_richtig enthalten je Ticket True (richtig) oder False (falsch).
    # Gezählt werden nur Tickets, bei denen genau eines der beiden Verfahren richtig liegt
    # (& bedeutet "und", ~ bedeutet "nicht").
    nur_a = int((a_richtig & ~b_richtig).sum())
    nur_b = int((~a_richtig & b_richtig).sum())
    # Wären beide Verfahren gleich gut, müssten sich diese Tickets wie bei einem Münzwurf etwa
    # zur Hälfte aufteilen. p ist die Wahrscheinlichkeit, rein zufällig eine mindestens so
    # ungleiche Aufteilung zu erhalten (Binomialverteilung mit 0,5, beidseitig).
    n, k = nur_a + nur_b, min(nur_a, nur_b)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return nur_a, nur_b, p


def wilson(richtig, n, z=1.96):
    """95%-Konfidenzintervall für einen Anteil (Wilson)."""
    # Gibt den Bereich an, in dem die tatsächliche Genauigkeit mit hoher Wahrscheinlichkeit
    # liegt. Je weniger Tickets ausgewertet werden, desto breiter ist dieser Bereich.
    # z = 1,96 entspricht einer Sicherheit von 95 %.
    p = richtig / n
    mitte = (p + z * z / (2 * n)) / (1 + z * z / n)
    breite = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return mitte - breite, mitte + breite


# Ergebnisse der Schritte 1 und 2 laden (je eine Zeile pro Testticket)
baseline = pd.read_csv("results/baseline.csv")
llm = pd.read_csv("results/zero_shot.csv")
# Für jedes Verfahren ein Paar aus Label des Datensatzes und Vorhersage. Alle folgenden Schleifen
# laufen über diese drei Paare, dadurch werden alle Verfahren auf genau dieselbe Weise ausgewertet.
verfahren = {
    "Naive Bayes": (baseline["Label"], baseline["Naive Bayes"]),
    "Logistische Regression": (baseline["Label"], baseline["Logistische Regression"]),
    "GPT-6 Luna": (llm["Label"], llm["Vorhersage"]),
}

# ---------------------------------------------------------------- 6.1
# Klassifikationsgüte auf den 300 Testtickets, gemessen an den Labels des Datensatzes
print("===== 6.1 Klassifikationsgüte (300 Testtickets) =====\n")
print("Tabelle 6: Genauigkeit und Makro-Durchschnitt")
for name, (label, vorhersage) in verfahren.items():
    # average="macro": ungewichteter Mittelwert über die fünf Kategorien.
    # zero_division=0: Eine Kategorie, die nie vorhergesagt wird, erhält 0 statt einer Fehlermeldung.
    p, r, f, _ = precision_recall_fscore_support(label, vorhersage, labels=KATEGORIEN,
                                                 average="macro", zero_division=0)
    # Anzahl der Tickets, bei denen Vorhersage und Label übereinstimmen (True zählt als 1)
    richtig = (label == vorhersage).sum()
    print(f"  {name:24s} Genauigkeit {accuracy_score(label, vorhersage):.1%} ({richtig} von {len(label)}), "
          f"Precision {p:.1%}, Recall {r:.1%}, F1 {f:.1%}")

print("\nTabelle 7 und Text: Precision / Recall / F1 je Kategorie")
for name, (label, vorhersage) in verfahren.items():
    # Ohne average liefert die Funktion je einen Wert pro Kategorie, in der Reihenfolge von KATEGORIEN
    p, r, f, _ = precision_recall_fscore_support(label, vorhersage, labels=KATEGORIEN, zero_division=0)
    print(f"  {name}")
    # zip durchläuft die Kategorie und ihre drei Werte gemeinsam
    for k, pk, rk, fk in zip(KATEGORIEN, p, r, f):
        print(f"    {k:22s} {pk:6.1%} {rk:6.1%} {fk:6.1%}")

print("\nAnzahl der Vorhersagen je Kategorie (je Kategorie 60 Testtickets)")
for name, (_, vorhersage) in verfahren.items():  # _ : das Label wird hier nicht gebraucht
    # value_counts zählt, wie oft jede Kategorie vorhergesagt wurde. reindex ergänzt Kategorien,
    # die nie vorhergesagt wurden, mit 0 und bringt alle in die Reihenfolge von KATEGORIEN.
    anzahl = vorhersage.value_counts().reindex(KATEGORIEN, fill_value=0)
    print(f"  {name:24s} " + ", ".join(f"{KURZ[k]} {n}" for k, n in anzahl.items()))

# Zusatz, nicht in der Arbeit: Könnte der Abstand zwischen zwei Verfahren zufällig entstanden sein?
print("\nZusatz: McNemar-Test (Tickets, die nur das erste bzw. nur das zweite Verfahren richtig hat)")
# Je Verfahren eine Liste mit True/False: Wurde das Ticket richtig zugeordnet?
richtig = {name: (label == vorhersage) for name, (label, vorhersage) in verfahren.items()}
for a, b in [("Logistische Regression", "GPT-6 Luna"), ("Naive Bayes", "GPT-6 Luna"),
             ("Logistische Regression", "Naive Bayes")]:
    nur_a, nur_b, p = mcnemar(richtig[a], richtig[b])
    print(f"  {a} vs. {b}: {nur_a} zu {nur_b}, p = {p:.4f}")
for name, r in richtig.items():
    unten, oben = wilson(r.sum(), len(r))
    print(f"  95%-Konfidenzintervall Genauigkeit {name}: {unten:.1%} bis {oben:.1%}")

# ---------------------------------------------------------------- 6.2.1
# Welche Kategorien werden miteinander verwechselt? Tabelle 8 (LLM) und Anhang (NB, LR)
print("\n===== 6.2.1 Konfusionsmatrizen (Zeilen = Label, Spalten = Vorhersage) =====")
for name, (label, vorhersage) in verfahren.items():
    abweichung = label != vorhersage  # True, wenn die Vorhersage vom Label abweicht
    print(f"\n{name}: {abweichung.sum()} Abweichungen vom Label")
    print(matrix(label, vorhersage).to_string())
    # Welcher Kategorie ordnet das Verfahren die abweichenden Tickets zu?
    # (z. B. das LLM 106-mal Technischer Support)
    nach_vorhersage = vorhersage[abweichung].value_counts().reindex(KATEGORIEN, fill_value=0)
    print("  Abweichungen nach Vorhersage: " + ", ".join(f"{KURZ[k]} {n}" for k, n in nach_vorhersage.items()))

# ---------------------------------------------------------------- 6.2.2
# Qualität der Labels: Ergebnis der manuellen Prüfung von 100 Tickets (Tabelle 9)
print("\n===== 6.2.2 Qualität der Labels (100 manuell geprüfte Tickets) =====\n")
stichprobe = pd.read_csv("data/stichprobe_100.csv")
# Ein Label gilt als korrekt, wenn es mit der manuell geprüften Kategorie übereinstimmt
stichprobe["Korrekt"] = stichprobe["Label"] == stichprobe["Manuell"]
print("Tabelle 9")
for k in KATEGORIEN:
    teil = stichprobe[stichprobe["Label"] == k]  # die 20 Tickets mit diesem Label
    # Welche geprüfte Kategorie haben die Tickets mit falschem Label stattdessen?
    stattdessen = teil.loc[~teil["Korrekt"], "Manuell"].value_counts()
    print(f"  {k:22s} {teil['Korrekt'].sum():2d} von {len(teil)} passen   "
          f"stattdessen: {', '.join(f'{m} {n}' for m, n in stattdessen.items())}")
print(f"  Summe: {stichprobe['Korrekt'].sum()} von {len(stichprobe)}")

falsch = stichprobe[~stichprobe["Korrekt"]]
# Grenzfall: Auch eine andere Kategorie wäre vertretbar gewesen (Spalte Grenzfall der Stichprobe)
grenzfaelle = (falsch["Grenzfall"] == "Ja").sum()
print(f"\nFalsche Labels: {len(falsch)}, davon Grenzfälle: {grenzfaelle}")

# Hochrechnung auf alle 8.110 Tickets: Anteil korrekter Labels je Kategorie,
# gewichtet mit dem Anteil der Kategorie im Datensatz.
# Grund: Die Stichprobe enthält aus jeder Kategorie gleich viele Tickets (20), im Datensatz sind
# die Kategorien aber sehr unterschiedlich groß (Technischer Support ca. 64 %).
# normalize=True liefert Anteile statt Anzahlen.
anteil = pd.read_csv("data/tickets_de.csv")["Kategorie"].value_counts(normalize=True)
gewichtet, mit_grenzfaellen = 0, 0
for k in KATEGORIEN:
    teil = stichprobe[stichprobe["Label"] == k]
    korrekt = teil["Korrekt"].sum()
    # falsche Labels, die als Grenzfall markiert sind
    grenz = ((~teil["Korrekt"]) & (teil["Grenzfall"] == "Ja")).sum()
    # z. B. Technischer Support: 0,638 × 9/20
    gewichtet += anteil[k] * korrekt / len(teil)
    # obere Schätzung: Grenzfälle zusätzlich als korrekt gezählt
    mit_grenzfaellen += anteil[k] * (korrekt + grenz) / len(teil)
print(f"Geschätzter Anteil korrekter Labels im Datensatz: {gewichtet:.0%} "
      f"(Grenzfälle als korrekt gezählt: {mit_grenzfaellen:.0%})")

# Prüfung des Sprachfilters aus 00_datenaufbereitung.py mit einem zweiten Verfahren (lingua).
# try/except: Ist lingua nicht installiert, wird dieser Teil übersprungen statt abzubrechen.
try:
    from lingua import Language, LanguageDetectorBuilder
except ImportError:
    print("\nSprachfilter-Prüfung übersprungen (lingua nicht installiert)")
else:  # wird nur ausgeführt, wenn der Import geklappt hat
    # Quelle und Stoppwortlisten wie in 00_datenaufbereitung.py
    QUELLE = ("https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets/resolve/"
              "ddf1c81a5475992c4fa6752bf1e8b4e31f07bbeb/aa_dataset-tickets-multi-lang-5-2-50-version.csv")
    DEUTSCH = set("und der die das ich wir sie ist nicht mit für bitte ein eine zu von den auf ihre "
                  "unsere können".split())
    ENGLISCH = set("the and is to we our please you for with of a in are have this".split())

    def ist_deutsch(text):
        # Text in Kleinbuchstaben in Wörter zerlegen, dann deutsche und englische Stoppwörter zählen
        woerter = re.findall(r"[a-zäöüß]+", str(text).lower())
        return sum(w in DEUTSCH for w in woerter) > sum(w in ENGLISCH for w in woerter)

    # Detektor nur mit Deutsch und Englisch, da genau diese beiden Sprachen unterschieden werden sollen
    detektor = LanguageDetectorBuilder.from_languages(Language.GERMAN, Language.ENGLISH).build()
    quelle = pd.read_csv(QUELLE)  # lädt die Quelldatei erneut aus dem Internet
    de = quelle[quelle["language"] == "de"]  # Tickets mit Sprachangabe "de" (12.249)
    heuristik = de["body"].map(ist_deutsch)  # True = laut eigener Heuristik deutsch
    lingua = de["body"].map(lambda t: detektor.detect_language_of(str(t)) == Language.GERMAN)  # True = laut lingua deutsch
    print(f"\nSprachfilter: {len(de)} Tickets mit Sprachangabe 'de'")
    print(f"  laut lingua englisch: {(~lingua).sum()} ({(~lingua).mean():.0%})")
    print(f"  von der Heuristik behalten: {heuristik.sum()}, davon laut lingua deutsch: {(heuristik & lingua).sum()}")
    # Tickets, die die Heuristik verworfen hat, obwohl sie laut lingua deutsch sind.
    # str.split().str.len() zählt die Wörter je Text.
    verworfen = de.loc[~heuristik & lingua, "body"]
    print(f"  deutsche Tickets, die die Heuristik verworfen hat: {len(verworfen)} "
          f"(Länge im Median {verworfen.str.split().str.len().median():.0f} Wörter, "
          f"alle Tickets: {de['body'].str.split().str.len().median():.0f} Wörter)")

# ---------------------------------------------------------------- 6.2.3
# Zusatzauswertung auf den 100 geprüften Tickets: Wenn das LLM vom Label abweicht,
# bestätigt die manuelle Prüfung dann eher das LLM oder das Label?
print("\n===== 6.2.3 Zusatzauswertung: Abweichungen des LLM vom Label (100 geprüfte Tickets) =====")
print("Ergebnisse von 04_stichprobe.py, hier aufgeschlüsselt nach Label und Vorhersage des LLM.")
print("'davon geprüft gleich' = die manuelle Prüfung ergab dieselbe Kategorie wie das LLM\n")
geprueft = pd.read_csv("results/stichprobe.csv")
abweichung = geprueft[geprueft["LLM"] != geprueft["Label"]]  # nur Tickets, bei denen das LLM abweicht
# groupby bildet eine Gruppe je Kombination aus Label und Vorhersage des LLM (z. B. RR → TS)
for (label, vorhersage), teil in abweichung.groupby(["Label", "LLM"]):
    print(f"  Label {KURZ[label]} → LLM {KURZ[vorhersage]}: {len(teil):2d}, "
          f"davon geprüft gleich: {(teil['Manuell'] == vorhersage).sum()}")
# Zusatz, nicht in der Arbeit: dieselben Tests wie in 6.1, aber gegen die geprüfte Kategorie
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
# Kosten und Antwortzeiten auf den 300 Testtickets (Tabelle 11)
print("\n===== 6.3 Kosten und Antwortzeit (300 Testtickets) =====\n")
# Summe der Tokens über alle 300 Anfragen an das LLM
eingabe, ausgabe = llm["Input_Tokens"].sum(), llm["Output_Tokens"].sum()
# Kosten = Tokens × Preis pro 1 Mio. Tokens / 1.000.000 (1e6 = 1.000.000)
kosten = eingabe * PREIS_EINGABE / 1e6 + ausgabe * PREIS_AUSGABE / 1e6
print(f"Tokens: {eingabe} Eingabe, {ausgabe} Ausgabe")
print(f"  pro E-Mail im Mittel: {eingabe / len(llm):.0f} Eingabe, {ausgabe / len(llm):.0f} Ausgabe "
      f"(Ausgabe zwischen {llm['Output_Tokens'].min()} und {llm['Output_Tokens'].max()})")
# Hochrechnung: Kosten pro E-Mail × 1.000 bzw. × 10.000
print(f"Kosten: {kosten:.4f} US-Dollar für {len(llm)} E-Mails, "
      f"{kosten / len(llm) * 1000:.3f} pro 1.000, {kosten / len(llm) * 10000:.2f} pro 10.000")

zeit = llm["Sekunden"]  # Antwortzeit je Ticket, gemessen in 02_llm_zero_shot.py
# idxmax liefert die Zeile mit der längsten Antwortzeit, daraus werden ID und Position im Lauf gelesen
print(f"\nGPT-6 Luna: Mittel {zeit.mean():.2f} s, Median {zeit.median():.2f} s, "
      f"Minimum {zeit.min():.2f} s, Maximum {zeit.max():.2f} s (ID {llm.loc[zeit.idxmax(), 'ID']}, "
      f"Position {zeit.idxmax() + 1} im Lauf)")
# NFA4: höchstens fünf Sekunden pro E-Mail
print(f"  Antworten über 5 Sekunden (NFA4): {(zeit > 5).sum()}")
for name in ["Naive Bayes", "Logistische Regression"]:
    ms = baseline[f"{name} Sekunden"] * 1000  # Sekunden in Millisekunden umrechnen
    print(f"{name}: Mittel {ms.mean():.2f} ms, Maximum {ms.max():.2f} ms")
# Verhältnis der mittleren Antwortzeiten von LLM und logistischer Regression
lr = baseline["Logistische Regression Sekunden"].mean()
print(f"GPT-6 Luna braucht im Mittel {zeit.mean() / lr:.0f}-mal so lange wie die logistische Regression")
print("Trainingszeiten: siehe Konsolenausgabe von 01_baseline.py (results/ausgabe_01_baseline.txt)")