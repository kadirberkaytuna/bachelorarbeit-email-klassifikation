# Prototyp: E-Mail-Klassifikation für KMU

## Einrichtung
```
pip install -r requirements.txt
```

## Ausführen
0. Datenaufbereitung (optional, `data/tickets_de.csv` liegt bereits im Repository):
```
   python 00_datenaufbereitung.py
```
   Lädt den Datensatz in einer festen Revision von Hugging Face, filtert die deutschsprachigen
   Tickets und ordnet sieben Original-Queues den fünf Kategorien zu.
1. Baseline (kein API-Schlüssel nötig, erzeugt auch das Testset):
```
   python 01_baseline.py
```
2. Zero-Shot mit OpenAI (API-Schlüssel nötig):
   - Mac/Linux: `export OPENAI_API_KEY="sk-..."`
   - Windows (PowerShell): `$env:OPENAI_API_KEY="sk-..."`
   - Windows (cmd): `set OPENAI_API_KEY=sk-...`
```
   python 02_llm_zero_shot.py
```
3. Fehleranalyse (nach Schritt 2):
```
   python 03_fehleranalyse.py
```

## Umgebung
Python 3.13.6, Paketversionen siehe `requirements.txt`. LLM: `gpt-6-luna` über die OpenAI API
(Chat Completions, `reasoning_effort="none"`, `temperature=0`). Die verwendete Modellversion wird
pro Ticket in `results/zero_shot.csv` gespeichert.

## Daten
`data/tickets_de.csv`: 8.110 deutschsprachige Tickets aus Tobi-Bueck/customer-support-tickets
(Hugging Face, DOI 10.57967/hf/6184, CC BY-NC 4.0), gefiltert und auf 5 Kategorien abgebildet.
Erzeugt mit `00_datenaufbereitung.py`.

| Kategorie | Original-Queue(s) | Tickets |
|---|---|---|
| Rechnung/Zahlung | Billing and Payments | 856 |
| Reklamation/Rückgabe | Returns and Exchanges | 414 |
| Vertriebsanfrage | Sales and Pre-Sales | 310 |
| Kundenservice | Customer Service | 1.354 |
| Technischer Support | Technical Support, Product Support, IT Support | 5.176 |

## Ergebnisse
- `results/baseline.csv`: Vorhersagen und Vorhersagezeiten der klassischen Verfahren
- `results/zero_shot.csv`: Vorhersagen, Rohantworten, Tokens und Antwortzeiten des LLM (Lauf vom 16.10.2026)
- `results/fehleranalyse.xlsx`: falsch klassifizierte Tickets des LLM
- `results/ausgabe_01_baseline.txt`, `results/ausgabe_02_llm.txt`: Konsolenausgaben

## Lizenz
- Code: MIT, siehe `LICENSE`
- Daten in `data/`: abgeleitet aus Tobi-Bueck/customer-support-tickets (Bück 2025, DOI 10.57967/hf/6184), CC BY-NC 4.0