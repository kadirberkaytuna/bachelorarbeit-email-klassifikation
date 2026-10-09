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
   - Windows: `set OPENAI_API_KEY=sk-...`
   ```
   python 02_llm_zero_shot.py
   ```

## Daten
`data/tickets_de.csv`: 8.110 deutschsprachige Tickets aus Tobi-Bueck/customer-support-tickets
(Hugging Face, DOI 10.57967/hf/6184, CC BY-NC 4.0), gefiltert und auf 5 Kategorien abgebildet.
