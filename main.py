#!/usr/bin/env python3
"""Forecast GTA VI trophy earn rates on Exophase at a target time.

Model (per trophy, per case)
-----------------------------
A trophy's earn rate is earners / players_tracked, and Exophase refreshes
its tracked population on its own schedule: when the population jumps, every
rate drops even while players keep earning trophies. So earners and players
are forecast separately and the rate is their ratio:

* Players: damped log-linear growth fitted on the history (daily log-growth
  capped at 0.5, total multiple capped at 30x). When the history is flat
  (no growth observed), a refresh prior pulls the median toward a ~6x
  population jump at the target, because a flat launch-night read is
  exactly the state just before Exophase's first refresh.
* Earners: a per-tier prior (bronze story trophies earn fast, gold and
  platinum slow) blended 50/50 in log space with a per-trophy regression
  of log(earners) against log-time since launch (regression multiplier
  capped at 8x, so a hot streak cannot run away).
* Cross-check: a per-trophy linear fit of logit(rate) against
  ln(1 + hours since launch); its implied move is capped at 5x. The final
  median blends 60% split-model / 40% trend, so the trend's story-
  progression rise is never discarded.
* Spread: a random-walk term growing with the horizon plus an explicit
  refresh / dilution-jump component that also grows with the horizon. The
  low side is 1.3x wider than the high side, because dilution can only
  push rates down sharply. Quantiles are built in logit space, so 0% and
  100% rates are handled through the 0.01-99.99 clamp and the 100% trophy
  forecasts flat at 100%.

Deterministic, standard library only, no network, offline.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_SCORES = [-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722]

# Per-day earner growth prior by tier (bronze story trophies earn fastest).
TIER_EARNER_PER_DAY = {
    "platinum": 1.18,
    "gold": 1.35,
    "silver": 1.70,
    "bronze": 2.20,
}

# Refresh prior: flat history implies the next Exophase refresh is pending.
REFRESH_JUMP = 6.0
# Caps.
MAX_PLAYER_MULTIPLE = 30.0
MAX_DAILY_LOG_PLAYER_GROWTH = 0.5
MAX_EARNER_REGRESSION_MULT = 8.0
MAX_TREND_MOVE = 5.0
SPLIT_WEIGHT = 0.50
LOW_SIDE_WIDEN = 1.30
RW_SIGMA_PER_SQRT_DAY = 0.55
JUMP_SIGMA = 1.30


def parse_time(text):
    """Parse an ISO-8601 timestamp (with Z) into an aware datetime."""
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def logit(pct):
    p = min(99.99, max(0.01, pct)) / 100.0
    return math.log(p / (1.0 - p))


def logistic(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def linfit(xs, ys):
    """Least-squares slope/intercept for y ~ a + b*x. Returns (b, a)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return 0.0, my
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return b, my - b * mx


def launch_time(game):
    rel = game.get("release_date") if isinstance(game, dict) else None
    if rel:
        try:
            return parse_time(rel + "T00:00:00Z")
        except ValueError:
            pass
    return datetime(2026, 11, 19, tzinfo=timezone.utc)


def player_multiple(hours, players, horizon_hours):
    """Median players at target / players at last read."""
    days = horizon_hours / 24.0
    slope = 0.0
    if len(hours) >= 2 and max(hours) > min(hours):
        slope, _ = linfit(hours, [math.log(max(1, p)) for p in players])
    # Damp the fitted growth: launch-night exponentials never persist.
    grow_log = 0.5 * slope * horizon_hours
    grow_log = min(grow_log, MAX_DAILY_LOG_PLAYER_GROWTH * days,
                   math.log(MAX_PLAYER_MULTIPLE))
    grow_log = max(grow_log, -0.25)
    if abs(slope) < 0.004:
        # Flat history: a refresh is pending; pull toward a ~6x jump.
        exponent = 0.55 + 0.45 * min(1.0, days / 4.5)
        prior = REFRESH_JUMP ** exponent
        return max(prior, math.exp(grow_log))
    return math.exp(grow_log)


def earner_multiple(tier, xs, earners, horizon_x, horizon_days):
    """Median earners at target / earners at last read."""
    daily = TIER_EARNER_PER_DAY.get(tier, 1.70)
    tier_mult = daily ** horizon_days
    reg_mult = 1.0
    pts = [(x, e) for x, e in zip(xs, earners) if e > 0.0]
    if len(pts) >= 2 and max(p[0] for p in pts) > min(p[0] for p in pts):
        slope, _ = linfit([p[0] for p in pts],
                          [math.log(p[1]) for p in pts])
        dx = horizon_x - pts[-1][0]
        if dx > 0:
            reg_mult = math.exp(slope * dx)
            reg_mult = min(max(reg_mult, 0.05), MAX_EARNER_REGRESSION_MULT)
    # 50/50 blend of tier prior and per-trophy regression, in log space.
    return math.sqrt(tier_mult * reg_mult)


def trend_rate(xs, rates, horizon_x, last_rate):
    """Logit(rate) vs ln(1 + hours since launch) trend, move capped at 5x."""
    pts = [(x, r) for x, r in zip(xs, rates)
           if r is not None and 0.0 < r < 100.0]
    if len(pts) < 2 or max(p[0] for p in pts) <= min(p[0] for p in pts):
        return last_rate
    slope, _ = linfit([p[0] for p in pts], [logit(p[1]) for p in pts])
    move = slope * (horizon_x - pts[-1][0])
    cap = math.log(MAX_TREND_MOVE)
    move = min(max(move, -cap), cap)
    return logistic(logit(last_rate) + move) * 100.0


def spread_sigmas(horizon_days):
    """(low-side sigma, high-side sigma) in logit space."""
    rw = RW_SIGMA_PER_SQRT_DAY * math.sqrt(max(horizon_days, 0.0))
    jump = JUMP_SIGMA * min(1.0, 0.35 + horizon_days)
    sigma = math.sqrt(rw * rw + jump * jump)
    return sigma * LOW_SIDE_WIDEN, sigma


def forecast_trophy(tier, xs, hours, rates, players, horizon_x, horizon_days,
                    horizon_hours):
    """Return the five quantiles (percent) for one trophy."""
    # Last non-null rate and its earner count.
    last_rate = None
    last_players = players[-1]
    for r in reversed(rates):
        if r is not None:
            last_rate = r
            break
    if last_rate is None:
        last_rate = 1.0
    if last_rate >= 99.99:
        return [100.0, 100.0, 100.0, 100.0, 100.0]

    earners = []
    for r, p in zip(rates, players):
        if r is None:
            earners.append(None)
        else:
            earners.append(max(0.0, r / 100.0 * p))
    last_earners = None
    for e in reversed(earners):
        if e is not None:
            last_earners = e
            break
    if last_earners is None or last_earners <= 0.0:
        # Nobody has earned it yet: seed half an earner so growth can lift it.
        last_earners = 0.005 * last_players

    p_mult = player_multiple(hours, players, horizon_hours)
    # Earner regression uses only reads with a known rate.
    ex = [x for x, e in zip(xs, earners) if e is not None]
    ee = [e for e in earners if e is not None]
    e_mult = earner_multiple(tier, ex, ee, horizon_x, horizon_days)

    split_rate = last_earners * e_mult / (last_players * p_mult) * 100.0
    trend = trend_rate(xs, rates, horizon_x, last_rate)
    split_rate = min(max(split_rate, 0.01), 99.99)
    trend = min(max(trend, 0.01), 99.99)
    median = math.exp(SPLIT_WEIGHT * math.log(split_rate) +
                      (1.0 - SPLIT_WEIGHT) * math.log(trend))
    median = min(max(median, 0.01), 99.99)

    sig_lo, sig_hi = spread_sigmas(horizon_days)
    centre = logit(median)
    out = []
    for z in Z_SCORES:
        s = sig_lo if z < 0 else sig_hi
        out.append(round(logistic(centre + z * s) * 100.0, 4))
    out = [min(99.99, max(0.01, v)) for v in out]
    for i in range(1, 5):
        out[i] = max(out[i], out[i - 1])
    return out


def process_case(case):
    launch = launch_time(case.get("game"))
    at = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_minutes = case.get("horizon_minutes")
    if horizon_minutes is None:
        horizon_minutes = (target - at).total_seconds() / 60.0
    horizon_hours = horizon_minutes / 60.0
    horizon_days = horizon_minutes / 1440.0
    horizon_x = math.log1p(max(0.0, (target - launch).total_seconds() / 3600.0))

    history = case.get("history") or []
    hours = []
    players = []
    for read in history:
        t = parse_time(read["at"])
        hours.append(max(0.0, (t - launch).total_seconds() / 3600.0))
        players.append(max(1, int(read["players_tracked"])))
    if not history:
        hours = [0.0]
        players = [1]

    xs = [math.log1p(h) for h in hours]
    rows = []
    trophies = case["trophies"]
    for idx, trophy in enumerate(trophies):
        rates = []
        for read in history:
            rs = read.get("rates") or []
            rates.append(rs[idx] if idx < len(rs) else None)
        if not rates:
            rates = [None]
        q = forecast_trophy(trophy.get("tier", "bronze"), xs, hours, rates,
                            players, horizon_x, horizon_days, horizon_hours)
        rows.append({"id": trophy["id"], "q": q})
    rows.sort(key=lambda r: r["id"])
    return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith(".in"):
            continue
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        result = process_case(case)
        stem = case.get("id") or name[:-3]
        with open(os.path.join(out_dir, stem + ".out"), "w",
                  encoding="utf-8") as fh:
            json.dump(result, fh, separators=(",", ":"))
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
