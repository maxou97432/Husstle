# Live bot — testnet operational checklist

## What this is
- Single-pair (BTC/ETH) live bot, **testnet only** (mainnet is hard-blocked).
- Validates operational conformity per brief §0 — NOT the strategy's edge.

## What's wired
- `execution/hl_client.py` — HL SDK wrapper, EIP-712 signing, SSL retry, dry-run fallback.
- `execution/orders.py` — two-leg post-only entry + emergency orphan-leg close on partial fill.
- `risk/sizing.py` — 2× per leg, dollar-neutral via β.
- `risk/safeguards.py` — circuit breaker, drawdown halt, consecutive-error halt.
- `alerts/notify.py` — JSONL journal (`live_journal.jsonl`) + optional Telegram.
- `live_bot.py` — every 1h: fetch recent candles → compute signal (same code path as backtest) → enter/exit per identical rules.

## Run dry (no creds)
```bash
cd statarb
../env/bin/python live_bot.py --once --dry-run     # single iteration
../env/bin/python live_bot.py --dry-run            # loop forever
```

## To go LIVE on testnet
1. Fund a testnet wallet at https://app.hyperliquid-testnet.xyz/drip
2. Export environment variables (NEVER commit these):
   ```bash
   export HL_PRIVATE_KEY="0x..."            # testnet key only
   export HL_ACCOUNT_ADDRESS="0x..."        # wallet that signed
   export TG_BOT_TOKEN="..."                # optional
   export TG_CHAT_ID="..."                  # optional
   ```
3. Smoke test:
   ```bash
   ../env/bin/python live_bot.py --once --capital 50
   ```
4. Long run:
   ```bash
   ../env/bin/python live_bot.py --capital 50 --entry 2.0 --exit 0.5 --stop 3.5
   ```

## What the bot WILL do
- Place post-only limit orders (maker) on both legs simultaneously at the best touch.
- If only one leg fills within 5s: cancel the other, emergency-close the orphan with IOC (taker, slippage capped at 0.2%).
- Halt all new entries if: account drawdown ≥ 10% from peak, OR 3 consecutive broker errors.
- Use the brief's exact entry/exit/stop/kill rules — same code as the backtest.

## What it will NOT do
- Touch mainnet (raises RuntimeError on init).
- Trade pairs other than BTC/ETH in this phase.
- Adjust position size mid-trade.
- Re-enter immediately after a kill-switch (manual restart required).
