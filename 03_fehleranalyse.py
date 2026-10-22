"""Schritt 3: Fehleranalyse - Tickets, bei denen LLM und Label nicht übereinstimmen."""
import pandas as pd

test = pd.read_csv("data/testset.csv")
llm = pd.read_csv("results/zero_shot.csv")

# Betreff und Text ergänzen: merge verbindet beide Tabellen über die ID (wie SVERWEIS in Excel)
df = llm.merge(test[["ID", "Betreff", "Text"]], on="ID")
# Nur Tickets behalten, bei denen die Vorhersage des LLM vom Label des Datensatzes abweicht
fehler = df[df["Label"] != df["Vorhersage"]]

print(f"{len(fehler)} von {len(df)} Tickets falsch klassifiziert\n")
print(pd.crosstab(fehler["Label"], fehler["Vorhersage"]))  # wer wird mit wem verwechselt

fehler[["ID", "Label", "Vorhersage", "Betreff", "Text"]].to_excel(
    "results/fehleranalyse.xlsx", index=False
)
print("\nGespeichert in results/fehleranalyse.xlsx")