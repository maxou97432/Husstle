# Architecture (multi-stratégies)

```
statarb/
├── core/                        # Infrastructure partagée
│   ├── config.py                # INTERVAL, BARS_PER_YEAR, LOOKBACK
│   ├── data/                    # Candles & funding (DuckDB + HL fetch)
│   ├── execution/               # HLClient, place_two_legs
│   ├── risk/                    # sizing.leg_sizes, safeguards
│   ├── alerts/                  # JSONL journal + Telegram
│   └── backtest/                # engine, metrics (shared)
│
├── strategies/                  # Logique par stratégie
│   ├── base.py                  # Strategy ABC, Decision, Action, Leg
│   └── statarb/                 # Implémentation StatArb BTC/ETH
│       ├── config.py            # UNIVERSE, fenêtres, seuils
│       ├── hedge.py             # OLS causal
│       ├── spread.py            # spread + z-score
│       ├── cointegration.py     # ADF p-value rolling
│       ├── robustness.py        # bootstrap, OOS, shuffle
│       └── strategy.py          # StatArbStrategy(Strategy)
│
├── orchestrator.py              # Boucle générique, exécute N Strategy
├── live_bot.py                  # Wrapper : crée StatArb + lance orchestrator
├── run_backtest.py              # Backtest CLI
└── tools/, tests/               # Sweeps, sanity tests
```

## Ajouter une nouvelle stratégie

1. Créer `strategies/<nom>/` avec :
   - `config.py` : params spécifiques
   - `signal.py` : calculs du signal
   - `strategy.py` : `class MyStrategy(Strategy)`
2. Dans `live_bot.py`, instancier et ajouter à la liste passée à `Orchestrator`.

L'orchestrateur découvre :
- Quels coins fetcher : `Strategy.required_coins()`
- Quoi faire à chaque barre : `Strategy.compute_signal(market_data, position) → Decision`
- L'action à exécuter : `Decision.action ∈ {HOLD, ENTER, EXIT}`

## Contrat `Strategy`

```python
class Strategy(ABC):
    name: str
    def required_coins(self) -> list[str]: ...
    def compute_signal(self, market_data, position) -> Decision: ...
    def on_filled(self, decision, fill_info): ...  # hook optionnel
```

`Decision` porte :
- `action` : HOLD / ENTER / EXIT
- `legs` : list[Leg] (coin, is_buy, size, price)
- `telemetry` : dict logué dans le journal (z, ADF, funding…)
- `state` : opaque, repassé à `compute_signal()` la fois suivante
- `safety_stop_pct` : stop natif HL à poser après entrée

## Garanties préservées par le refactor

- Logique de signal **identique** au backtest (même code, même fenêtres)
- Safety stops natifs HL toujours posés à -5%
- Partial-fill handler (orphan-close IOC) inchangé
- Circuit breaker drawdown / erreurs inchangé
- Tests de causalité passent toujours sur la nouvelle archi
