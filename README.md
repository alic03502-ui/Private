# Football Banker AI — Professional v7

## New in v7
- Backtesting from historical odds/results CSV
- Hit rate, ROI/unit, net units and equity curve
- Bankroll tracking
- Quarter-Kelly stake suggestion with 2% bankroll cap
- Risk grades
- Local SQLite odds snapshot database
- Odds capture so you can build your own historical odds dataset
- Live/top-10 banker scanner retained from v6

## Important API limitation
API-Sports states that pre-match odds history is retained for only 7 days and live odds are not historically stored. Therefore, the app does **not pretend** it can backtest years of bookmaker ROI from API-Sports alone. Use the Odds Capture module to build a local dataset or import a historical CSV.

## Run
pip install -r requirements.txt
streamlit run app.py

Set the key:
Windows PowerShell:
$env:API_FOOTBALL_KEY='YOUR_KEY'
streamlit run app.py
