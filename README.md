# GTA VI trophy rarity — week-one forecast

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ruri08219/gta6-trophy-rarity-forecast-01-entry/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ruri08219/gta6-trophy-rarity-forecast-01-entry/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Entry for the [make.fun bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01):
forecast, for each of Grand Theft Auto VI's 55 PlayStation 5 trophies, its
earn rate on Exophase at 2026-11-26 00:00 UTC, as five quantiles
(5 / 25 / 50 / 75 / 95%) per trophy. `main.py` is Python 3.13, standard
library only, no network, deterministic, and runs in well under the
8-second limit.

## Model

A trophy's rate is earners divided by players tracked, and Exophase
refreshes its counts on its own schedule: when the tracked population
jumps, every rate drops even while players keep earning trophies. Launch
night showed exactly that — 638 players tracked and every rate flat from
23:48 UTC, then one refresh by 04:30 UTC took the population to 3,868 and
cut every rate but the one at 100%. So earners and players are forecast
separately and the rate is their ratio, instead of extrapolating the rate
itself.

1. **Players:** log-linear growth fitted on the last 24 h of reads, damped
   (0.5/day cap) and capped at 30x — population growth saturates as the
   launch rush passes. When the history is flat, a refresh prior applies
   instead: the expected population multiplier grows toward ~6x over the
   horizon with a ~6 h time constant.
2. **Earners, per trophy:** new players earn a trophy at a tier-dependent
   fraction of the population growth (early bronze story trophies fast,
   gold/platinum slow), plus steady progression among players already
   tracked. Where the history shows real earner movement, a log-time
   regression of earners supplies a data component blended 50/50 with the
   tier prior.
3. **Cross-check:** a per-trophy straight-line fit of logit(rate) against
   ln(1 + hours since launch) over the last 24 h of reads (the house curve
   baseline's shape), its move capped at 5x up or down from the last read.
   The median blends the two components in logit space; the earners/players
   split earns weight only as a refresh becomes likely over the horizon
   (a maybe at 90 minutes, capped at 60% over the final's 4.5 days), so the
   trend — rates rising as players progress through the story — is never
   discarded.
4. **Spread, honest and wide:** a random-walk term growing with the horizon,
   plus an explicit refresh/dilution-jump term that grows over the horizon
   and saturates, plus the trend fit's prediction variance. The low side is
   wider than the high side: a refresh can cut a rate several-fold, while
   upside is bounded by how fast players can progress. The 100% trophy is
   forecast flat at 100% — the refresh that cut every other rate left it
   untouched.

## Local validation

Exact pinball scorer on the logit scale (clamp 0.01–99.99), synthetic
dilution scenarios built from the practice cases, plus a rich-history
robustness test (fabricated evolving reads) and a final-like long-horizon
scenario (6,510 min):

| scenario | this entry | house starter |
|---|---|---|
| dilution, players 638 → 3,868, earners bronze ×3 / silver ×2 / gold ×1.5 / plat ×1.2 | 0.377 | 0.688 |
| strong dilution, earners bronze ×2 / silver ×1.5 / gold ×1.2 / plat ×1.1 | 0.614 | 1.063 |
| moderate dilution, players 638 → 2,000 | 0.221 | 0.406 |
| no refresh at all (the refresh the practice answer actually contains did not happen) | 0.168 | 0.151 |
| final-like, 6,510 min horizon, players → 50,000 | 0.518 | 0.881 |

90% coverage under the dilution scenarios is 0.84–1.00 (the starter's is
0.00 — its band misses every refreshed rate). The no-refresh row is the
honest price of insuring against the refresh: a slightly worse loss when
no refresh comes, a much better one when it does — and the practice answer
does contain a refresh, as does the final's 4.5-day horizon with near
certainty. Leave-future-out validation inside the practice history is
degenerate (the history is flat), so the synthetic dilution scenarios above
are the real check. Outputs are byte-identical across runs, ~0.05 s per
case, format valid (55 rows, non-decreasing quantiles, ≤ 65,536 bytes).
