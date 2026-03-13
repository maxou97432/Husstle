# Straddle Optimizer — Paradex + Variational.io

Outil d'aide à la décision pour la stratégie **Delta-Neutral Breakout** (Long + Short simultanés à x50).  
Disponible avec une **Interface Web** (nouveau !) ou en ligne de commande (CLI).

---

## 🌐 Dashboard Web (Recommandé)

Interface graphique se rafraîchissant toute seule toutes les 30 secondes.

```bash
# 1. Aller dans le dossier du projet
cd /Users/maximethiong-ly/Documents/crypto

# 2. Activer l'environnement virtuel
source env/bin/activate

# 3. Installer les dépendances (première fois uniquement)
pip install flask flask-cors requests numpy

# 4. Lancer le serveur local
python app.py
```

👉 Ouvre ensuite **http://localhost:5001** dans ton navigateur !  
*(Laisse la fenêtre de terminal ouverte pendant que tu utilises le site)*

---

## 💻 Ligne de commande (Terminal)

Si tu préfères utiliser l'ancienne version texte :

```bash
cd /Users/maximethiong-ly/Documents/crypto
source env/bin/activate
python opti.py
```

> **Pour quitter l'env virtuel :** `deactivate`  
> **Pour relancer l'outil plus tard :** `source env/bin/activate` puis `python app.py` (web) ou `python opti.py` (CLI)

---

## Configuration

Modifie **uniquement** le bloc `CONFIGURATION` en haut du fichier `opti.py` :

| Variable | Description | Défaut |
|---|---|---|
| `SYMBOL` | Actif cible | `"ETHUSDT"` |
| `COLLATERAL_PER_DEX` | Capital par plateforme ($) | `50` |
| `LEVERAGE` | Levier | `50` |
| `TARGET_NET_PROFIT` | Profit net visé ($) | `30` |
| `MAX_LOSS_PCT_COLLATERAL` | % de perte max acceptée sur le collatéral | `0.40` |
| `ATR_MULTIPLIER` | Multiplicateur ATR pour le SL dynamique | `1.5` |
| `ATR_PERIOD` | Période de calcul de l'ATR | `14` |
| `COMPRESSION_PERCENTILE` | Seuil bas de volatilité (30 = 30ème percentile) | `30` |
| `KLINE_INTERVAL` | Intervalle des bougies | `"1h"` |

---

## Logique de Signal

| Signal | Condition | Action |
|---|---|---|
| 🟢 **COMPRESSION FORTE** | ATR + BBW tous les deux en percentile bas | Entrer le trade |
| 🟡 **COMPRESSION MODÉRÉE** | Un seul indicateur en compression | Entrer avec prudence |
| 🔴 **NE PAS TRADER** | Marché volatil ou en range | Attendre |

---

## Comment utiliser le résultat

1. Lance `python opti.py`
2. Si le signal est 🟢 ou 🟡 :
   - Copie le **prix SL Long** → entre-le comme Stop Loss sur **Paradex (LONG)**
   - Copie le **prix TP Long** → entre-le comme Take Profit sur **Paradex (LONG)**
   - Copie le **prix SL Short** → entre-le comme Stop Loss sur **Variational.io (SHORT)**
   - Copie le **prix TP Short** → entre-le comme Take Profit sur **Variational.io (SHORT)**
3. Si le signal est 🔴 : ne prends aucune position, relance le script plus tard.

---

## Pourquoi les 2 SL se touchent (et comment c'est évité)

Le "Double SL" arrive quand le marché est en **range + wicks** (pas de tendance).  
L'outil détecte la **compression de volatilité** (ATR + Bollinger Band Width anormalement bas) — les marchés en compression génèrent ensuite des breakouts directionnels forts, ce qui est le seul contexte où la stratégie est rentable mathématiquement.

**Règle d'or maintenue :** `Distance TP ≥ 2 × Distance SL` garantit qu'un trade gagnant couvre toujours une perte potentielle.
