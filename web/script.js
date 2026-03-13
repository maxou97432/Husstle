/* ═══════════════════════════════════════════
   Straddle Optimizer — Frontend Logic
   ═══════════════════════════════════════════ */

const API_BASE = "http://localhost:5001";
const REFRESH_SECS = 30;

let countdownTimer = null;
let countdownVal = REFRESH_SECS;
let activeParams = null;
let volChart = null;
let lastScore = null;

/* ── Helpers ── */
const $ = id => document.getElementById(id);
const fmt = (n, d = 2) => Number(n).toLocaleString("fr-FR", { minimumFractionDigits: d, maximumFractionDigits: d });

function showLoading(on) {
    $("loading-overlay").classList.toggle("hidden", !on);
}

function showError(msg) {
    $("error-msg").textContent = msg;
    $("error-toast").classList.remove("hidden");
    setTimeout(() => $("error-toast").classList.add("hidden"), 6000);
}

/* ── Mode detection ── */
function isLiveMode() {
    return $("time-input").value.trim() === ""
        && $("price-long").value.trim() === ""
        && $("price-short").value.trim() === "";
}

function onModeChange() {
    const live = isLiveMode();

    // Header badge
    const badge = $("refresh-badge");
    badge.textContent = live ? "↻ AUTO" : "⏸ PAUSE";
    badge.className = "status-pill " + (live ? "status-live" : "status-paused");

    // Card badge
    const badge2 = $("refresh-badge2");
    badge2.textContent = live ? "↻ AUTO" : "⏸ PAUSE";
    badge2.className = "card-label-badge" + (live ? "" : " card-label-badge-paused");

    // Countdown
    $("countdown-label").textContent = live ? "Prochain refresh dans" : "Auto-refresh en pause";
    $("countdown").textContent = live ? (countdownVal + "s") : "–";

    // Spread preview
    const pL = parseFloat($("price-long").value);
    const pS = parseFloat($("price-short").value);
    if (!isNaN(pL) && !isNaN(pS) && pL > 0 && pS > 0) {
        const spread = Math.abs(pL - pS);
        const spreadPct = (spread / Math.min(pL, pS) * 100).toFixed(4);
        $("spread-val").textContent = `$${spread.toFixed(4)}`;
        $("spread-pct").textContent = spreadPct;
        $("spread-row").classList.remove("hidden");
    } else {
        $("spread-row").classList.add("hidden");
    }
}

/* ── Countdown ── */
function startCountdown() {
    clearInterval(countdownTimer);
    if (!isLiveMode()) return;

    countdownVal = REFRESH_SECS;
    $("countdown").textContent = countdownVal + "s";

    countdownTimer = setInterval(() => {
        if (!isLiveMode()) { clearInterval(countdownTimer); return; }
        countdownVal--;
        $("countdown").textContent = countdownVal + "s";
        if (countdownVal <= 0) { clearInterval(countdownTimer); fetchAnalysis(); }
    }, 1000);
}

/* ── Settings panel ── */
function toggleSettings() {
    const body = $("settings-body");
    const chevron = $("settings-chevron");
    const isOpen = body.classList.contains("open");
    body.classList.toggle("open", !isOpen);
    chevron.classList.toggle("open", !isOpen);
}

function collectParams() {
    return {
        symbol: $("p-symbol").value,
        position_size_usdc: parseFloat($("p-position").value),
        leverage: parseInt($("p-leverage").value),
        target_roi_collateral: parseFloat($("p-roi").value) / 100,
        max_loss_pct_collateral: parseFloat($("p-max-loss").value) / 100,
        kline_interval: $("p-interval").value,
        kline_limit: parseInt($("p-kline-limit").value),
        atr_multiplier: parseFloat($("p-atr-mult").value),
        atr_period: parseInt($("p-atr-period").value),
        compression_percentile: parseInt($("p-compression").value),
        min_rr_ratio: parseFloat($("p-min-rr").value),
    };
}

function applySettings() { activeParams = collectParams(); fetchAnalysis(); }

const DEFAULTS = {
    symbol: "ETHUSDT", position: "4000", leverage: "50", roi: "15",
    maxLoss: "40", minRr: "1.5", interval: "15m", klineLimit: "150",
    atrMult: "1.5", atrPeriod: "14", compression: "30",
};

function resetSettings() {
    $("p-symbol").value = DEFAULTS.symbol;
    $("p-position").value = DEFAULTS.position;
    $("p-leverage").value = DEFAULTS.leverage;
    $("p-roi").value = DEFAULTS.roi;
    $("p-max-loss").value = DEFAULTS.maxLoss;
    $("p-min-rr").value = DEFAULTS.minRr;
    $("p-interval").value = DEFAULTS.interval;
    $("p-kline-limit").value = DEFAULTS.klineLimit;
    $("p-atr-mult").value = DEFAULTS.atrMult;
    $("p-atr-period").value = DEFAULTS.atrPeriod;
    $("p-compression").value = DEFAULTS.compression;
    activeParams = null;
    fetchAnalysis();
}

function syncSettingsFromData(params) {
    if (activeParams !== null) return;
    $("p-symbol").value = params.symbol;
    $("p-position").value = params.position_size_usdc;
    $("p-leverage").value = params.leverage;
    $("p-roi").value = Math.round(params.target_roi_collateral * 100);
    $("p-max-loss").value = Math.round(params.max_loss_pct_collateral * 100);
    $("p-interval").value = params.kline_interval;
    $("p-kline-limit").value = params.kline_limit;
    $("p-atr-mult").value = params.atr_multiplier;
    $("p-atr-period").value = params.atr_period;
    $("p-compression").value = params.compression_percentile;
    $("p-min-rr").value = params.min_rr_ratio;
}

/* ── Main fetch ── */
async function fetchAnalysis() {
    showLoading(true);
    clearInterval(countdownTimer);

    const body = {};
    const priceLong = parseFloat($("price-long").value);
    const priceShort = parseFloat($("price-short").value);
    const timeInput = $("time-input").value.trim();

    if (!isNaN(priceLong) && priceLong > 0) body.price_long = priceLong;
    if (!isNaN(priceShort) && priceShort > 0) body.price_short = priceShort;
    if (timeInput && !body.price_long && !body.price_short) body.time = timeInput;
    if (activeParams) body.params = activeParams;

    try {
        const response = await fetch(`${API_BASE}/api/analyze`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({ error: `HTTP ${response.status}` }));
            throw new Error(err.error || `HTTP ${response.status}`);
        }

        const json = await response.json();
        if (!json.ok) throw new Error(json.error);
        renderDashboard(json.data);

    } catch (e) {
        showError(`Erreur : ${e.message}`);
    } finally {
        showLoading(false);
        startCountdown();
    }
}

/* ── Render ── */
function renderDashboard(d) {
    const { meta, compression: c, trade: t } = d;

    /* Header */
    $("meta-symbol").textContent = meta.symbol;
    $("meta-leverage").textContent = `×${meta.leverage}`;
    $("meta-interval").textContent = meta.interval.toUpperCase();
    $("meta-timestamp").textContent = meta.timestamp;
    $("meta-price-label").textContent = isLiveMode() ? "LIVE" : "POST-TRADE";

    if (meta.params) syncSettingsFromData(meta.params);

    /* Compression */
    $("kline-info").textContent = `${meta.kline_limit} bougies`;

    setDot("atr-dot", c.atr_compressed);
    setDot("bbw-dot", c.bbw_compressed);

    $("atr-now").textContent = c.atr_now;
    $("atr-detail").textContent = `Seuil ≤ ${c.atr_thresh}  ·  rang ${c.atr_rank}%`;
    $("bbw-now").textContent = `${c.bbw_now}%`;
    $("bbw-detail").textContent = `Seuil ≤ ${c.bbw_thresh}%  ·  rang ${c.bbw_rank}%`;

    /* Verdict — bar inside card */
    const vBox = $("verdict-box");
    vBox.className = "verdict-bar " + verdictClass(c.score);
    $("verdict-bar-text").textContent = c.verdict;

    /* Verdict — hero pill */
    const vPill = $("verdict-pill");
    const dotClass = c.score === 2 ? "dot-green" : c.score === 1 ? "dot-amber" : "dot-red";
    $("verdict-dot").className = "metric-dot " + dotClass;
    $("verdict-text").textContent = c.verdict;

    /* Warning */
    $("warning-banner").classList.toggle("hidden", c.go);

    /* LONG */
    $("long-entry").textContent = `$${fmt(t.entry_long, 4)}`;
    $("long-tp-pct").textContent = `+${t.tp_pct}%`;
    $("long-tp-roi").textContent = `+${t.roi_tp_pct}% ROI`;
    $("long-tp-pnl").textContent = `+$${fmt(t.win_amount)}`;
    $("long-tp-price").textContent = `$${fmt(t.long_tp, 4)}`;
    $("long-sl-pct").textContent = `-${t.sl_pct}%`;
    $("long-sl-roi").textContent = `-${t.roi_sl_pct}% ROI`;
    $("long-sl-pnl").textContent = `-$${fmt(t.loss_amount)}`;
    $("long-sl-price").textContent = `$${fmt(t.long_sl, 4)}`;

    /* SHORT */
    $("short-entry").textContent = `$${fmt(t.entry_short, 4)}`;
    $("short-tp-pct").textContent = `+${t.tp_pct}%`;
    $("short-tp-roi").textContent = `+${t.roi_tp_pct}% ROI`;
    $("short-tp-pnl").textContent = `+$${fmt(t.win_amount)}`;
    $("short-tp-price").textContent = `$${fmt(t.short_tp, 4)}`;
    $("short-sl-pct").textContent = `-${t.sl_pct}%`;
    $("short-sl-roi").textContent = `-${t.roi_sl_pct}% ROI`;
    $("short-sl-pnl").textContent = `-$${fmt(t.loss_amount)}`;
    $("short-sl-price").textContent = `$${fmt(t.short_sl, 4)}`;

    /* Summary */
    $("sum-position").textContent = `$${fmt(t.position_size_usdc, 0)}`;
    $("sum-collateral").textContent = `~$${fmt(t.collateral, 2)}`;
    $("sum-roi-target").textContent = `${t.target_roi_pct}% → $${fmt(t.profit_target)}`;
    $("sum-rr").textContent = `${t.rr_ratio.toFixed(2)} : 1 ${t.rr_ratio >= t.min_rr_ratio - 0.01 ? "✅" : "⚠️"}`;
    $("sum-net").textContent = `+$${fmt(t.net_profit)}`;
    $("sum-liq").textContent = `±$${fmt(t.liq_dist)}`;

    /* Audio Alert */
    if (c.score === 2 && lastScore !== 2 && isLiveMode()) {
        const audio = $("alert-sound");
        if (audio) {
            audio.play().catch(e => console.log("Audio play prevented", e));
        }
    }
    lastScore = c.score;

    /* Chart */
    updateChart(c.history_atr, c.history_bbw);
}

function updateChart(atrHist, bbwHist) {
    const ctx = document.getElementById('volatility-chart');
    if (!ctx) return;

    if (!volChart) {
        // Init chart
        Chart.defaults.color = '#a1a1aa';
        Chart.defaults.font.family = "'JetBrains Mono', monospace";
        volChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: Array(atrHist.length).fill(''),
                datasets: [
                    {
                        label: 'ATR',
                        data: atrHist,
                        borderColor: '#2E71FF',
                        backgroundColor: 'rgba(46, 113, 255, 0.1)',
                        borderWidth: 2,
                        tension: 0.3,
                        yAxisID: 'y'
                    },
                    {
                        label: 'BBW (%)',
                        data: bbwHist,
                        borderColor: '#a5b4fc',
                        backgroundColor: 'rgba(165, 180, 252, 0.1)',
                        borderWidth: 2,
                        tension: 0.3,
                        yAxisID: 'y1'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        enabled: false
                    }
                },
                scales: {
                    x: { display: false },
                    y: {
                        type: 'linear',
                        display: false,
                        position: 'left',
                    },
                    y1: {
                        type: 'linear',
                        display: false,
                        position: 'right',
                        grid: { drawOnChartArea: false }
                    }
                },
                elements: {
                    point: { radius: 0 }
                }
            }
        });
    } else {
        // Update data
        volChart.data.labels = Array(atrHist.length).fill('');
        volChart.data.datasets[0].data = atrHist;
        volChart.data.datasets[1].data = bbwHist;
        volChart.update();
    }
}

/* ── Utilities ── */
function setDot(id, isCompressed) {
    $(id).className = "status-dot " + (isCompressed ? "dot-green" : "dot-red");
}

function verdictClass(score) {
    if (score === 2) return "verdict-green";
    if (score === 1) return "verdict-amber";
    return "verdict-red";
}

/* ── Init ── */
document.addEventListener("DOMContentLoaded", () => {
    $("time-input").addEventListener("keydown", e => {
        if (e.key === "Enter") fetchAnalysis();
    });

    // Start in manual mode; fetch ML suggestion in background
    setMode("manual");
    fetchAnalysis();
});

/* ══════════════════════════════════════════════════
   ML Mode Selector
══════════════════════════════════════════════════ */

let currentMode = "manual";
let lastMlSuggestion = null;

function setMode(mode) {
    currentMode = mode;

    // Update tab styles
    $("mode-tab-manual").classList.toggle("active", mode === "manual");
    $("mode-tab-ml").classList.toggle("active", mode === "ml");

    if (mode === "ml") {
        // Open settings panel so the user sees the overridden values
        const body = $("settings-body");
        if (!body.classList.contains("open")) toggleSettings();
        fetchMlSuggestion();
    } else {
        // Hide the ML bubble when switching back to manual
        $("ml-bubble").classList.add("hidden");
    }
}

async function fetchMlSuggestion() {
    try {
        const res = await fetch(`${API_BASE}/api/ml-suggest`);
        if (!res.ok) throw new Error((await res.json()).error);
        const json = await res.json();
        if (!json.ok) throw new Error(json.error);

        lastMlSuggestion = json.suggestion;
        renderMlBubble(json.suggestion);
        applyMlParams(); // Auto-fill settings and re-analyze
    } catch (e) {
        showError(`ML : ${e.message}`);
    }
}

function renderMlBubble(s) {
    const bubble = $("ml-bubble");
    bubble.classList.remove("hidden");

    // Market state
    const ms = s.market_state;
    $("ml-market-state").textContent =
        `ATR ${ms.atr_now} · Rang ATR ${ms.atr_rank}% · Rang BBW ${ms.bbw_rank}%`;

    // Predicted PnL
    const pnlSign = s.predicted_pnl >= 0 ? "+" : "";
    $("ml-pnl-badge").textContent = `PnL prédit : ${pnlSign}$${s.predicted_pnl.toFixed(2)}`;

    // Params
    $("ml-atr-mult").textContent = `×${s.atr_multiplier}`;
    $("ml-roi").textContent = `${(s.target_roi_collateral * 100).toFixed(0)}%`;
    $("ml-comp").textContent = `${s.compression_percentile}e pctile`;
}

function applyMlParams() {
    if (!lastMlSuggestion) { fetchMlSuggestion(); return; }

    // Inject into settings fields
    $("p-atr-mult").value = lastMlSuggestion.atr_multiplier;
    $("p-roi").value = Math.round(lastMlSuggestion.target_roi_collateral * 100);
    $("p-compression").value = lastMlSuggestion.compression_percentile;

    // Use those as active params and refresh
    activeParams = collectParams();
    fetchAnalysis();
}

/* ══════════════════════════════════════════════════
   NAVIGATION & JOURNAL
══════════════════════════════════════════════════ */

function switchMainTab(tabId) {
    // Update active nav button
    document.querySelectorAll('.nav-tab').forEach(btn => {
        btn.classList.toggle('active', btn.textContent.toLowerCase().includes(tabId));
    });

    // Update main sections
    $("tab-dashboard").classList.toggle('hidden', tabId !== 'dashboard');
    $("tab-journal").classList.toggle('hidden', tabId !== 'journal');

    if (tabId === 'journal') {
        loadTrades();
    }
}

async function loadTrades() {
    try {
        const res = await fetch(`${API_BASE}/api/trades`);
        const json = await res.json();
        if (json.ok) {
            renderTrades(json.data);
        }
    } catch (e) {
        showError("Erreur chargement trades");
    }
}

function renderTrades(trades) {
    const tbody = $("trades-body");
    tbody.innerHTML = "";
    
    if (trades.length === 0) {
        tbody.innerHTML = `<tr><td colspan="8" style="text-align:center; color: var(--t3); padding: 2rem;">Aucun trade dans l'historique</td></tr>`;
        return;
    }

    trades.forEach(t => {
        const pnl = parseFloat(t.pnl);
        const pnlStr = pnl >= 0 ? `+${pnl.toFixed(2)}$` : `${pnl.toFixed(2)}$`;
        const pnlClass = pnl > 0 ? "td-green" : (pnl < 0 ? "td-red" : "");
        const statusClass = `status-${t.status.toLowerCase()}`;
        
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td class="td-mono">${t.date}</td>
            <td><strong>${t.symbol}</strong></td>
            <td class="td-mono">${t.entry_long.toFixed(4)}</td>
            <td class="td-mono">${t.entry_short.toFixed(4)}</td>
            <td class="td-mono ${pnlClass}"><strong>${pnlStr}</strong></td>
            <td><span class="status-badge ${statusClass}">${t.status}</span></td>
            <td style="max-width:200px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;" title="${t.notes}">${t.notes || '-'}</td>
            <td>
                <button class="btn-icon" onclick="deleteTrade('${t.id}')" title="Supprimer">🗑️</button>
            </td>
        `;
        tbody.appendChild(tr);
    });
}

async function submitTrade(e) {
    e.preventDefault();
    
    // Auto date if empty
    let d = $("t-date").value.trim();
    if (!d) {
        const now = new Date();
        const pad = n => n.toString().padStart(2, '0');
        d = `${now.getFullYear()}-${pad(now.getMonth()+1)}-${pad(now.getDate())} ${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}`;
    }

    const payload = {
        date: d,
        symbol: $("t-symbol").value.trim().toUpperCase() || "ETHUSDT",
        entry_long: parseFloat($("t-long").value),
        entry_short: parseFloat($("t-short").value),
        pnl: parseFloat($("t-pnl").value),
        status: $("t-status").value,
        notes: $("t-notes").value.trim()
    };

    try {
        const res = await fetch(`${API_BASE}/api/trades`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        const json = await res.json();
        if (json.ok) {
            $("trade-form").reset();
            loadTrades(); // reload table
        } else {
            throw new Error(json.error);
        }
    } catch (err) {
        showError("Erreur lors de l'ajout: " + err.message);
    }
}

async function deleteTrade(id) {
    if (!confirm("Supprimer définitivement ce trade ?")) return;
    try {
        const res = await fetch(`${API_BASE}/api/trades?id=${id}`, { method: "DELETE" });
        const json = await res.json();
        if (json.ok) loadTrades();
    } catch (e) {
        showError("Erreur lors de la suppression");
    }
}
