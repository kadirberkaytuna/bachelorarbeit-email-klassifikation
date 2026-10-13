# Prototyp: E-Mail-Klassifikation für KMU

## Einrichtung
```
pip install -r requirements.txt
```

## Ausführen
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