"""Forecast GTA VI trophy earn rates on Exophase at the case target.

Model (earners / players split, with a dilution-aware spread):

  rate = earners / players_tracked.  Exophase refreshes its counts on its own
  schedule; when the tracked population jumps, every rate drops even while
  players keep earning trophies.  So earners and players are forecast
  separately and the rate is their ratio, instead of extrapolating the rate
  itself.

  * Players: log-linear growth fitted on the last 24 h of reads, damped and
    capped.  When the history is flat (no growth seen yet) a refresh prior
    applies instead: launch night showed one refresh in ~5 h taking the
    tracked population 638 -> 3,868 (about 6x), so over a horizon h the
    expected population multiplier grows toward 6x with a ~6 h time constant.
  * Earners, per trophy: new players earn a trophy at a tier-dependent
    fraction of the population growth (early bronze story trophies fast,
    gold/platinum slow), plus steady progression among players already
    tracked.  Where the history shows real earner movement, a log-time
    regression of earners supplies a data component blended 50/50 with
    the tier prior.
  * Cross-check: a per-trophy straight-line fit of logit(rate) against
    ln(1 + hours since launch) over the last 24 h of reads (the house curve
    baseline's shape).  The median blends the two components in logit
    space, with the earners/players split earning weight only as a refresh
    becomes likely over the horizon (a maybe at 90 minutes, a near-certainty
    over the final's 4.5 days); at short horizons the wide low-side spread
    carries the refresh risk instead.
  * Spread: honest and wide.  A random-walk term growing with the horizon,
    plus an explicit refresh/dilution-jump term that grows over the horizon
    and saturates, plus the trend fit's prediction variance.  The low side
    is wider than the high side: a refresh can cut a rate several-fold,
    while upside is bounded by how fast players can progress.

Python 3.13, standard library only, no network, deterministic.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

# Launch anchor: GTA VI release day, 2026-11-19 00:00 UTC.
LAUNCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Tier priors: (share of population growth that becomes new earners,
# steady earner progression per day among players already tracked).
TIER_PRIOR = {
    "bronze": (0.55, 0.35),
    "silver": (0.45, 0.20),
    "gold": (0.30, 0.10),
    "platinum": (0.20, 0.05),
}


def parse_time(text):
    """Parse an ISO 8601 timestamp (with a trailing Z) into an aware datetime."""
    if text is None:
        return None
    try:
        return datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(l):
    return 100.0 / (1.0 + math.exp(-l))


def ols(xs, ys):
    """Least-squares line y = a + b*x. Returns (a, b, resid_var, x_mean, sxx)
    or None when the fit is degenerate."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    var = resid / max(n - 2, 1) if n > 2 else 0.0
    return (a, b, var, mx, sxx)


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_days = case["horizon_minutes"] / 1440.0
    horizon_hours = case["horizon_minutes"] / 60.0
    checkpoint = parse_time(case.get("at")) or LAUNCH
    target = parse_time(case.get("target")) or checkpoint

    # --- Read history into per-read series -------------------------------
    # Keep only reads with a usable timestamp, ordered oldest first.
    reads = []
    for h in history:
        t = parse_time(h.get("at"))
        if t is None:
            continue
        players = h.get("players_tracked")
        rates = h.get("rates") or []
        if not isinstance(players, (int, float)) or players <= 0:
            continue
        reads.append((t, float(players), rates))
    reads.sort(key=lambda r: r[0])

    n_troph = len(trophies)
    last_rates = [None] * n_troph
    last_players = None
    if reads:
        last_players = reads[-1][1]
        last_rates = list(reads[-1][2])
        # Pad / trim to trophy count.
        last_rates = (last_rates + [None] * n_troph)[:n_troph]

    # Hours since launch for each read; last 24 h window for the fits.
    window_start = checkpoint.timestamp() - 24 * 3600.0
    win = [r for r in reads if r[0].timestamp() >= window_start] or reads
    hours_win = [(r[0] - LAUNCH).total_seconds() / 3600.0 for r in win]
    x_win = [math.log1p(max(h, 0.0)) for h in hours_win]
    target_hours = max((target - LAUNCH).total_seconds() / 3600.0, 0.0)
    x_target = math.log1p(target_hours)
    last_hours = max((reads[-1][0] - LAUNCH).total_seconds() / 3600.0, 0.0) if reads else 0.0
    x_last = math.log1p(last_hours)

    # --- Players forecast -------------------------------------------------
    # Log-linear growth of the tracked population over the window.
    p_growth = 0.0
    if len(win) >= 2:
        fit_p = ols(hours_win, [math.log(r[1]) for r in win])
        if fit_p is not None:
            # Slope is per hour; convert to a per-day log growth rate.
            # Damped (0.5/day cap): log-linear extrapolation over a 4.5-day
            # horizon explodes if early growth was fast, and population
            # growth saturates as the launch rush passes.
            p_growth = max(0.0, min(0.5, fit_p[1] * 24.0))
    data_factor = math.exp(p_growth * horizon_days)
    # Refresh prior: expected population multiplier when the history is
    # flat.  One refresh in ~5 h took 638 -> 3,868 (~6x); the multiplier
    # grows toward 6x with a 6 h time constant over the horizon.
    prior_factor = 1.0 + 5.0 * (1.0 - math.exp(-horizon_hours / 6.0))
    pop_factor = max(data_factor, prior_factor)
    pop_factor = min(pop_factor, 30.0)
    players_target = (last_players or 638.0) * pop_factor
    dilution = pop_factor  # earners are divided by this population jump

    # --- Per-trophy forecasts ---------------------------------------------
    rows = []
    for i, tr in enumerate(trophies):
        tier = str(tr.get("tier", "bronze")).lower()
        k_new, prog = TIER_PRIOR.get(tier, (0.45, 0.20))
        r_last = last_rates[i]
        if r_last is None:
            r_last = 1.0
        r_last = float(r_last)

        # The 100% trophy: every tracked player holds it, and the refresh
        # that cut every other rate left it at 100%.  Forecast 100 flat.
        if r_last >= 99.99:
            rows.append({"id": tr["id"], "q": [100.0, 100.0, 100.0, 100.0, 100.0]})
            continue

        # Earners series for this trophy (rate * players / 100).
        e_series = []
        x_series = []
        for (t, players, rates) in win:
            if i < len(rates) and rates[i] is not None and rates[i] > 0:
                e_series.append(float(rates[i]) / 100.0 * players)
                x_series.append(math.log1p(max((t - LAUNCH).total_seconds() / 3600.0, 0.0)))

        # Component 1: earners / players split.
        # Prior earner growth: new-player earnings plus progression.
        g_prior = (dilution ** k_new) * math.exp(prog * horizon_days)
        g_data = None
        if len(e_series) >= 5 and max(e_series) > min(e_series):
            fit_e = ols(x_series, [math.log(e) for e in e_series])
            if fit_e is not None:
                g_data = math.exp(fit_e[1] * (x_target - x_last))
                g_data = max(0.2, min(20.0, g_data))
        if g_data is not None:
            g_earners = math.exp(0.5 * math.log(g_prior) + 0.5 * math.log(g_data))
        else:
            g_earners = g_prior
        r_split = clamp_pct(r_last * g_earners / dilution)

        # Component 2: logit(rate) vs ln(1 + hours) trend fit.
        y_series = []
        for (t, players, rates) in win:
            if i < len(rates) and rates[i] is not None:
                y_series.append(logit(float(rates[i])))
        r_trend = r_last
        trend_var = 0.0
        if len(y_series) >= 3:
            fit_t = ols(x_win[: len(y_series)], y_series)
            if fit_t is not None:
                a, b, var, mx, sxx = fit_t
                r_trend = clamp_pct(inv_logit(a + b * x_target))
                # A straight line in log-time extrapolates explosively when
                # the window's growth was fast; cap the trend's move at 5x
                # up / 5x down from the last read.
                r_trend = min(max(r_trend, 0.2 * r_last), 5.0 * r_last)
                # Prediction variance at the target (leverage grows as the
                # target moves away from the fitted window).
                trend_var = var * (1.0 + 1.0 / len(y_series)
                                   + (x_target - mx) ** 2 / sxx) if sxx > 0 else var
        # Blend the two medians in logit space.  The split component only
        # earns weight as a refresh becomes likely over the horizon: over
        # 90 minutes a refresh is a maybe (~12% weight), over the final's
        # 4.5 days it is a near-certainty — but the split never takes more
        # than 60%: the trend is what captures rates rising as players
        # progress through the story, and a pure dilution median would
        # miss that rise.  The wide low-side spread below carries the
        # refresh risk at short horizons instead.
        w_split = min(0.6, 1.0 - math.exp(-horizon_hours / 12.0))
        med_logit = (1.0 - w_split) * logit(r_trend) + w_split * logit(r_split)

        # --- Spread --------------------------------------------------------
        # Random-walk term + refresh/dilution-jump term (grows with the
        # horizon, saturates) + trend prediction variance + a floor.
        sigma_rw = 0.5 * math.sqrt(horizon_days)
        sigma_dil = 1.35 * math.sqrt(1.0 - math.exp(-horizon_hours / 10.0))
        sigma = math.sqrt(sigma_rw ** 2 + sigma_dil ** 2 + trend_var + 0.02)
        # A refresh cuts rates much further than it lifts them: the low
        # side of the band is wider than the high side.
        sigma_low = sigma * 1.30
        sigma_high = sigma * 0.85

        qs = []
        for z in Z:
            s = sigma_low if z < 0 else (sigma_high if z > 0 else sigma)
            qs.append(round(clamp_pct(inv_logit(med_logit + z * s)), 4))
        # Enforce non-decreasing after rounding.
        for j in range(1, 5):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": tr["id"], "q": qs})

    return {"trophies": rows}


def main():
    in_dir = INPUT_DIR
    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in in_dir.glob("*.in"))
    for name in names:
        case = json.loads((in_dir / name).read_text(encoding="utf-8"))
        result = forecast_case(case)
        out_path = out_dir / (Path(name).stem + ".out")
        out_path.write_text(json.dumps(result, separators=(",", ":")),
                            encoding="utf-8")


if __name__ == "__main__":
    main()
