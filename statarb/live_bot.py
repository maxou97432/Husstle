"""
Thin entry-point: instantiate StatArb strategy and hand it to the orchestrator.

This file existed before as a 250-line monolithic loop; it now just wires up
the new abstraction. Add other strategies by appending them to the list below.

Usage:
    python live_bot.py [--once] [--dry-run]
                       [--entry 2.0 --exit 0.5 --stop 3.5]
                       [--capital 200] [--safety-stop-pct 0.05]
"""
from __future__ import annotations

import argparse

from orchestrator import Orchestrator, build_default_client
from core.risk.safeguards import Safeguards
from strategies.statarb.strategy import StatArbStrategy, StatArbParams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--entry", type=float, default=2.0)
    ap.add_argument("--exit",  type=float, default=0.5)
    ap.add_argument("--stop",  type=float, default=3.5)
    ap.add_argument("--capital", type=float, default=100.0)
    ap.add_argument("--safety-stop-pct", type=float, default=0.05)
    args = ap.parse_args()

    client = build_default_client()
    if args.dry_run:
        client.dry_run = True

    statarb = StatArbStrategy(StatArbParams(
        entry_z=args.entry,
        exit_z=args.exit,
        stop_z=args.stop,
        capital_usd=args.capital,
        safety_stop_pct=args.safety_stop_pct,
    ))
    sg = Safeguards(max_open_pairs=1)
    orch = Orchestrator(strategies=[statarb], client=client, safeguards=sg)
    orch.loop_forever(once=args.once)


if __name__ == "__main__":
    main()
