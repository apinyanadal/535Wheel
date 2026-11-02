# Wheel Strategy Backtest (FINTECH 535)

Sell cash-secured puts. If assigned, sell covered calls against the shares. If called away, start over.

Live site: served by GitHub Pages from the `docs/` folder.

## Layout

- `docs/index.html`: the site. It reads everything from `docs/data/`, so the page never needs editing.
- `docs/data/`: written by the backtest:
  - `summary.json`: metadata, metrics table, commentary text
  - `nav.csv`: daily NAV for the wheel and benchmarks (`date` column plus one column per series)
  - `blotter.csv`: every trade (opens, closes, assignments, call-aways, rolls)
  - `ledger.csv`: end-of-day cash, positions, NAV
  - `positions.json`: full portfolio snapshot per date, keyed `YYYY-MM-DD`
- Backtest code: added in the next step.

Data from LSEG Workspace. Raw LSEG pulls and credentials are not committed.
