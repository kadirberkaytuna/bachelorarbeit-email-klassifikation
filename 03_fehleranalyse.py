"""Schritt 3: Fehleranalyse - Tickets, bei denen LLM und Label nicht übereinstimmen."""
import pandas as pd

test = pd.read_csv("data/testset.csv")
llm = pd.read_csv("results/zero_shot.csv")

df = llm.merge(test[["ID", "Betreff", "Text"]], on="ID")
fehler = df[df["Wahr"] != df["Vorhersage"]]

print(f"{len(fehler)} von {len(df)} Tickets falsch klassifiziert\n")
print(pd.crosstab(fehler["Wahr"], fehler["Vorhersage"]))  # wer wird mit wem verwechselt

fehler[["ID", "Wahr", "Vorhersage", "Betreff", "Text"]].to_excel(
    "results/fehleranalyse.xlsx", index=False
)
print("\nGespeichert in results/fehleranalyse.xlsx")