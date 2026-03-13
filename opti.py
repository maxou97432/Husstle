#!/usr/bin/env python3
"""
opti.py — Straddle Breakout Optimizer
======================================
Outil CLI d'aide à la décision pour stratégie Delta-Neutral Breakout
(Long + Short simultanés à x50 sur Paradex + Variational.io).

Modules :
  1. Connexion données temps réel (API Binance publique, sans clé)
  2. Détection de compression de volatilité (ATR + Bollinger Band Squeeze)
  3. Calculateur de spread dynamique (TP/SL basé sur l'ATR réel)
"""

import requests
import numpy as np
from datetime import datetime

# ══════════════════════════════════════════════════════════
#  CONFIGURATION — modifie UNIQUEMENT cette section
# ══════════════════════════════════════════════════════════




SYMBOL                  = "ETHUSDT"   # BTCUSDT | SOLUSDT | HYPEUSDT | ETHUSDT
POSITION_SIZE_USDC      = 4000        # Taille de position USDC (identique sur les 2 DEX)
LEVERAGE                = 50          # Levier utilisé
TARGET_ROI_COLLATERAL   = 0.15        # ROI cible sur la marge par trade (0.15 = 15%)
MAX_LOSS_PCT_COLLATERAL = 0.40        # Part max du collatéral risquée au SL (0.40 = 40%)
ATR_MULTIPLIER          = 1.5         # × ATR pour placer le SL hors du bruit
ATR_PERIOD              = 14          # Périodes pour le calcul de l'ATR
COMPRESSION_PERCENTILE  = 30          # Seuil de compression (percentile bas = compression)
KLINE_INTERVAL          = "15m"       # Intervalle : "15m" | "1h" | "4h"
KLINE_LIMIT             = 150         # Nombre de bougies historiques chargées
MIN_RR_RATIO            = 1.5         # Ratio TP/SL min (1.5 = ton gain paie 1.5x ton risque max)

# ══════════════════════════════════════════════════════════
#  COULEURS ANSI (terminal)
# ══════════════════════════════════════════════════════════

R   = "\033[91m"   # Rouge
G   = "\033[92m"   # Vert
Y   = "\033[93m"   # Jaune
B   = "\033[94m"   # Bleu
C   = "\033[96m"   # Cyan
W   = "\033[97m"   # Blanc vif
DIM = "\033[2m"    # Gris atténué
RST = "\033[0m"    # Reset
BLD = "\033[1m"    # Gras

# ══════════════════════════════════════════════════════════
#  MODULE 1 — DONNÉES TEMPS RÉEL (Binance API publique)
# ══════════════════════════════════════════════════════════

BASE_URL = "https://api.binance.com/api/v3"


def get_live_price(symbol: str) -> float:
    """Récupère le dernier prix spot de l'actif."""
    r = requests.get(f"{BASE_URL}/ticker/price", params={"symbol": symbol}, timeout=5)
    r.raise_for_status()
    return float(r.json()["price"])


def get_klines(symbol: str, interval: str, limit: int) -> list[dict]:
    """
    Récupère les bougies OHLCV depuis Binance.
    Retourne une liste de dicts {open, high, low, close, volume}.
    """
    r = requests.get(f"{BASE_URL}/klines", params={
        "symbol":   symbol,
        "interval": interval,
        "limit":    limit,
    }, timeout=10)
    r.raise_for_status()
    candles = [
        {"open": float(c[1]), "high": float(c[2]),
         "low":  float(c[3]), "close": float(c[4])}
        for c in r.json()
    ]
    # Retirer la dernière bougie car elle est "en cours" (non clôturée).
    # Cela évite les faux signaux de volatilité.
    return candles[:-1]


def get_price_at_time(symbol: str, ts: datetime) -> float:
    """
    Récupère le prix d'ouverture de la bougie 1m qui contenait le timestamp ts.
    Utilise l'API klines Binance avec startTime / endTime.

    ⚠️  Binance retourne une bougie uniquement si son heure d'OUVERTURE est ≥ startTime.
    Les bougies 1m ouvrent à des minutes rondes (ex: 15:09:00).
    On tronque donc le timestamp à la minute ronde pour ne pas "rater" la bougie.
    Exemple : 15:09:43 → startTime = 15:09:00 → Binance renvoie bien la bougie de 15:09.
    """
    # Tronquer à la minute ronde (ex: 15:09:43 → 15:09:00)
    ts_floor = ts.replace(second=0, microsecond=0)
    ts_ms    = int(ts_floor.timestamp() * 1000)
    end_ms   = ts_ms + 60_000   # +1 minute pour inclure la bougie entière
    r = requests.get(f"{BASE_URL}/klines", params={
        "symbol":    symbol,
        "interval":  "1m",
        "startTime": ts_ms,
        "endTime":   end_ms,
        "limit":     1,
    }, timeout=10)
    r.raise_for_status()
    data = r.json()
    if not data:
        raise ValueError(f"Aucune bougie trouvée pour {ts.strftime('%H:%M:%S')} sur Binance.")
    # On prend le prix d'ouverture de la bougie (premier prix de la minute)
    return float(data[0][1])


# ══════════════════════════════════════════════════════════
#  MODULE 2 — DÉTECTION DE COMPRESSION
# ══════════════════════════════════════════════════════════

def _true_ranges(candles: list[dict]) -> np.ndarray:
    """Calcule le True Range pour chaque bougie (High-Low, |High-Cprev|, |Low-Cprev|)."""
    highs  = np.array([c["high"]  for c in candles])
    lows   = np.array([c["low"]   for c in candles])
    closes = np.array([c["close"] for c in candles])
    prev   = np.roll(closes, 1)
    prev[0] = closes[0]
    return np.maximum(highs - lows,
           np.maximum(np.abs(highs - prev),
                      np.abs(lows  - prev)))


def _rolling_atr(trs: np.ndarray, period: int) -> np.ndarray:
    """ATR glissant (moyenne simple)."""
    return np.array([np.mean(trs[i - period:i]) for i in range(period, len(trs) + 1)])


def _rolling_bbw(closes: np.ndarray, period: int = 20, k: float = 2.0) -> np.ndarray:
    """Bollinger Band Width = (2k*σ) / SMA, glissant."""
    bbws = []
    for i in range(period, len(closes) + 1):
        w   = closes[i - period:i]
        sma = np.mean(w)
        std = np.std(w)
        bbws.append((2 * k * std) / sma if sma != 0 else 0.0)
    return np.array(bbws)


def detect_compression(candles: list[dict]) -> dict:
    """
    Analyse ATR + BB Width pour détecter si le marché est en compression.
    Retourne un dict avec 'go' (bool), métriques détaillées, et le verdict coloré.
    """
    trs    = _true_ranges(candles)
    atrs   = _rolling_atr(trs, ATR_PERIOD)
    closes = np.array([c["close"] for c in candles])
    bbws   = _rolling_bbw(closes)

    atr_now  = atrs[-1]
    bbw_now  = bbws[-1]

    atr_thresh = np.percentile(atrs, COMPRESSION_PERCENTILE)
    bbw_thresh = np.percentile(bbws, COMPRESSION_PERCENTILE)

    atr_rank = float(np.sum(atrs <= atr_now) / len(atrs) * 100)
    bbw_rank = float(np.sum(bbws <= bbw_now) / len(bbws) * 100)

    atr_compressed = bool(atr_now <= atr_thresh)
    bbw_compressed = bool(bbw_now <= bbw_thresh)
    score = int(atr_compressed) + int(bbw_compressed)

    if score == 2:
        verdict, color, go = "COMPRESSION FORTE  ✅  GO TRADE", G, True
    elif score == 1:
        verdict, color, go = "COMPRESSION MODÉRÉE  ⚠️  PRUDENCE", Y, True
    else:
        verdict, color, go = "EXPANSION / BRUIT  ⛔  NE PAS TRADER", R, False

    # Extract history for charting (last 100 points or less if fewer exist)
    hist_len = min(100, len(atrs))
    history_atr = atrs[-hist_len:].tolist()
    history_bbw = bbws[-hist_len:].tolist()

    return {
        "go": go, "score": score,
        "verdict": verdict, "verdict_color": color,
        "atr_now": atr_now,  "atr_thresh": atr_thresh, "atr_rank": atr_rank,
        "bbw_now": bbw_now,  "bbw_thresh": bbw_thresh, "bbw_rank": bbw_rank,
        "atr_compressed": atr_compressed,
        "bbw_compressed": bbw_compressed,
        "history_atr": history_atr,
        "history_bbw": history_bbw,
    }


# ══════════════════════════════════════════════════════════
#  MODULE 3 — CALCULATEUR DE SPREAD DYNAMIQUE
# ══════════════════════════════════════════════════════════

def calculate_trade(entry: float, atr: float) -> dict:
    """
    Calcule SL et TP dynamiques basés sur l'ATR réel du marché.
    Position identique sur les deux DEX (même notionnel USDC).
    Le profit cible est calculé dynamiquement depuis TARGET_ROI_COLLATERAL.
    """
    notional       = POSITION_SIZE_USDC
    collateral     = notional / LEVERAGE
    # Profit cible en $ = % ROI voulu × marge réelle
    profit_target  = collateral * TARGET_ROI_COLLATERAL

    # SL dynamique : hors du bruit détecté = ATR × multiplicateur
    sl_dist = atr * ATR_MULTIPLIER
    sl_pct  = sl_dist / entry

    # Plafond anti-liquidation (95% de la distance de liquidation théorique)
    max_sl_pct = (1 / LEVERAGE) * 0.95
    if sl_pct > max_sl_pct:
        sl_pct  = max_sl_pct
        sl_dist = entry * sl_pct

    loss_amount = notional * sl_pct

    # TP : perte + objectif de profit
    gross_needed  = loss_amount + profit_target
    tp_pct_target = gross_needed / notional

    # Garantie du ratio minimum TP/SL
    tp_pct = max(tp_pct_target, MIN_RR_RATIO * sl_pct)

    win_amount = notional * tp_pct
    net_profit = win_amount - loss_amount
    rr_ratio   = tp_pct / sl_pct

    # ROI réel sur la marge (ce que le DEX affiche)
    roi_tp_pct = win_amount / collateral * 100   # % gain si TP touché
    roi_sl_pct = loss_amount / collateral * 100  # % perte si SL touché
    roi_net    = net_profit / collateral * 100   # % net après les deux jambes

    return {
        "notional":      notional,
        "collateral":    collateral,
        "profit_target": profit_target,
        "entry":         entry,
        "sl_pct":        sl_pct,
        "tp_pct":        tp_pct,
        "loss_amount":   loss_amount,
        "win_amount":    win_amount,
        "net_profit":    net_profit,
        "rr_ratio":      rr_ratio,
        "roi_tp_pct":    roi_tp_pct,
        "roi_sl_pct":    roi_sl_pct,
        "roi_net":       roi_net,
        "long_sl":       entry * (1 - sl_pct),
        "long_tp":       entry * (1 + tp_pct),
        "short_sl":      entry * (1 + sl_pct),
        "short_tp":      entry * (1 - tp_pct),
    }


# ══════════════════════════════════════════════════════════
#  AFFICHAGE TERMINAL
# ══════════════════════════════════════════════════════════

def _sep(char="─", w=58):
    print(f"{DIM}{char * w}{RST}")


def _header():
    print()
    print(f"{BLD}{B}{'━' * 58}{RST}")
    print(f"{BLD}{W}   ⚡  STRADDLE OPTIMIZER  —  Paradex × Variational.io{RST}")
    print(f"{BLD}{B}{'━' * 58}{RST}")
    now = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    print(f"{DIM}   {now}  |  {SYMBOL}  |  x{LEVERAGE}  |  {KLINE_INTERVAL.upper()}{RST}")
    print()


def _print_compression(c: dict):
    ok   = lambda v: f"{G}🟢{RST}" if v else f"{R}🔴{RST}"
    print(f"{BLD}📊  ANALYSE VOLATILITÉ  ({KLINE_INTERVAL.upper()} — {KLINE_LIMIT} bougies){RST}")
    _sep()
    print(f"  {ok(c['atr_compressed'])}  ATR actuel  : {W}{c['atr_now']:.5f}{RST}   "
          f"{DIM}seuil ≤ {c['atr_thresh']:.5f}  |  rang {c['atr_rank']:.0f}ème %{RST}")
    print(f"  {ok(c['bbw_compressed'])}  BB Width    : {W}{c['bbw_now']*100:.3f}%{RST}   "
          f"{DIM}seuil ≤ {c['bbw_thresh']*100:.3f}%  |  rang {c['bbw_rank']:.0f}ème %{RST}")
    print()
    vc = c["verdict_color"]
    print(f"  {vc}{BLD}  ▶  {c['verdict']}{RST}")
    print()


def _print_trade(t: dict):
    rr_c = G if t["rr_ratio"] >= MIN_RR_RATIO - 0.01 else Y

    print(f"{BLD}📐  PARAMÈTRES DU TRADE{RST}")
    _sep()
    print(f"  Taille position   : {W}{BLD}${POSITION_SIZE_USDC:,.0f} USDC / DEX{RST}"
          f"   {DIM}(Total : ${POSITION_SIZE_USDC * 2:,.0f} USDC){RST}")
    print(f"  Marge / DEX       : {W}~${t['collateral']:,.2f}{RST}  "
          f"{DIM}(x{LEVERAGE} levier){RST}")
    print(f"  ROI cible / trade : {W}{BLD}{TARGET_ROI_COLLATERAL*100:.0f}%{RST} de la marge  "
          f"{DIM}= ${t['profit_target']:.2f} visés{RST}")
    print(f"  Prix d'entrée    : {W}{BLD}${t['entry']:,.4f}{RST}")
    print()

    print(f"  {B}{BLD}▶  LONG  (Variational){RST}")
    print(f"     {G}🟢 TAKE PROFIT  (+{t['tp_pct']*100:.3f}% prix)  [{G}{BLD}+{t['roi_tp_pct']:.1f}% ROI{RST}{G}]{RST}")
    print(f"         PNL : {G}{BLD}+${t['win_amount']:,.2f}{RST}  |  Prix → {W}{BLD}${t['long_tp']:,.4f}{RST}")
    print(f"     {R}🔴 STOP LOSS    (-{t['sl_pct']*100:.3f}% prix)  [{R}{BLD}-{t['roi_sl_pct']:.1f}% ROI{RST}{R}]{RST}")
    print(f"         PNL : {R}{BLD}-${t['loss_amount']:,.2f}{RST}  |  Prix → {W}{BLD}${t['long_sl']:,.4f}{RST}")
    print()

    print(f"  {Y}{BLD}▶  SHORT (Paradex){RST}")
    print(f"     {G}🟢 TAKE PROFIT  (+{t['tp_pct']*100:.3f}% prix)  [{G}{BLD}+{t['roi_tp_pct']:.1f}% ROI{RST}{G}]{RST}")
    print(f"         PNL : {G}{BLD}+${t['win_amount']:,.2f}{RST}  |  Prix → {W}{BLD}${t['short_tp']:,.4f}{RST}")
    print(f"     {R}🔴 STOP LOSS    (-{t['sl_pct']*100:.3f}% prix)  [{R}{BLD}-{t['roi_sl_pct']:.1f}% ROI{RST}{R}]{RST}")
    print(f"         PNL : {R}{BLD}-${t['loss_amount']:,.2f}{RST}  |  Prix → {W}{BLD}${t['short_sl']:,.4f}{RST}")
    print()

    _sep()
    liq_dist = (1 / LEVERAGE) * t["entry"]
    print(f"  {rr_c}⚖️  Ratio TP/SL  : {BLD}{t['rr_ratio']:.2f}:1{RST}   "
          f"{DIM}({'✅  Bon ratio' if t['rr_ratio'] >= MIN_RR_RATIO - 0.01 else '⚠️  Ratio trop faible'}){RST}")
    print(f"  {G}💰 Profit net   : {BLD}+${t['net_profit']:.2f}  (+{t['roi_net']:.1f}% ROI marge){RST}  "
          f"{DIM}(+${t['win_amount']:.2f} TP  −  ${t['loss_amount']:.2f} SL){RST}")
    print(f"  {DIM}⚠️  Liquidation ≈ ±${liq_dist:.2f} du prix d'entrée (x{LEVERAGE}){RST}")
    print()



def _print_warning_banner():
    """Bannière d'avertissement affichée quand le marché n'est pas en compression."""
    print(f"{R}{BLD}")
    print("  ╔══════════════════════════════════════════════╗")
    print("  ║  ⚠️  ATTENTION : MARCHÉ PAS EN COMPRESSION   ║")
    print("  ║  Risque de double SL élevé si tu trades !    ║")
    print("  ║  Les niveaux ci-dessous sont indicatifs.     ║")
    print("  ╚══════════════════════════════════════════════╝")
    print(f"{RST}")


def _ask_entry_time() -> tuple:
    """
    Mode interactif : demande à l'utilisateur l'heure d'entrée en trade.
    Retourne (prix, source_label) où prix est le prix Binance à cet instant.
    Si entrée vide → prix live.
    """
    print(f"  {C}{BLD}Mode Post-Trade{RST} {DIM}— Entrez l'heure du trade pour obtenir vos TP/SL exacts.{RST}")
    print(f"  {DIM}Format : HH:MM:SS  (ex: 10:27:38) — ou appuyez sur [ENTRÉE] pour le prix live{RST}")
    print()
    raw = input(f"  {W}{BLD}  ⏰  Heure d'entrée en trade ? {RST}").strip()
    print()

    if not raw:
        # Mode live : on récupère le prix actuel
        price = get_live_price(SYMBOL)
        return price, f"Prix LIVE ({datetime.now().strftime('%H:%M:%S')})"

    # Mode historique : on parse l'heure et on cherche sur Binance
    try:
        now = datetime.now()
        t   = datetime.strptime(raw, "%H:%M:%S").replace(
            year=now.year, month=now.month, day=now.day
        )
        price = get_price_at_time(SYMBOL, t)
        return price, f"Prix Historique ({t.strftime('%H:%M:%S')})"
    except ValueError:
        print(f"  {R}⚠️  Format invalide. Utilisation du prix live.{RST}\n")
        price = get_live_price(SYMBOL)
        return price, f"Prix LIVE ({datetime.now().strftime('%H:%M:%S')})"


# ══════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════

def main():
    _header()
    try:
        # ── 1. Demande de l'heure d'entrée (mode interactif) ───────────────
        entry_price, price_label = _ask_entry_time()

        # ── 2. Analyse de compression (toujours sur le marché actuel) ───────
        print(f"  {DIM}Chargement de l'analyse de volatilité...{RST}\n")
        candles = get_klines(SYMBOL, KLINE_INTERVAL, KLINE_LIMIT)
        comp    = detect_compression(candles)
        trade   = calculate_trade(entry_price, comp["atr_now"])

        # ── 3. Affichages ────────────────────────────────────────────
        _print_compression(comp)

        if not comp["go"]:
            _print_warning_banner()

        # Remplacement du label "live" par la source réelle du prix
        print(f"  {DIM}ℹ️  Référence prix : {W}{BLD}{price_label}{RST}  ({SYMBOL})───────────────────────{RST}")
        print()
        _print_trade(trade)

    except requests.exceptions.RequestException as e:
        print(f"\n{R}❌  Erreur API Binance : {e}{RST}")
    except Exception as e:
        print(f"\n{R}❌  Erreur : {e}{RST}")
        raise


if __name__ == "__main__":
    main()
