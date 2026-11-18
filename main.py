"""Forecast GTA VI's launch-day Twitch viewers hour by hour.

Entry for make.fun competition gta6-twitch-ccu-forecast-01 (program/v1,
Python 3.13, standard library only, no network).

Method -- learned from the comparables shipped inside each case file
--------------------------------------------------------------------
Every case carries ~38-42 past big launches ("comparables") with their full
hourly Twitch category counts from 14 days before to 3 days after their
release day.  For each comparable I compute, per release-day hour h = 2..24,
the log ratio log(viewers_h / viewers_1), where viewers_1 is that launch's
hour-1 print.  The forecast for the target case is then, per hour:

  centre_h = 0.5 * median_h  +  0.5 * regression_h(log p1)

where median_h is the median log ratio over the rolling-midnight pool and
regression_h is a per-hour linear regression of the log ratio on
log(p1) - 12 (p1 = the target case's hour-1 print).  The spread is the
regression's residual standard deviation, floored at 0.45 log units, and
quantiles are centre_h + z_tau * spread_h with z = (-1.645, -0.674, 0,
0.674, 1.645).  Centres are smoothed with a 3-hour moving average, and for
hours >= 14 the centre is floored at log(1.25): a rolling-midnight launch
whose US unlock (05:00 / 08:00 UTC) is still to come at the checkpoint does
not peak below 1.25x its hour-1 print.

Why this shape, and why not the house baseline (persistence, spread
0.15*sqrt(h-1))?  I validated it by leave-one-out scoring on the kit's
comparables, scoring each comparable as a pseudo-case with the pinball loss
on the log scale (the competition's score):

  * rolling-midnight, tier-comparable launches (n=11, the closest analogues
    to GTA VI: console, rolling midnight-local unlock, no early access):
    baseline 0.363, this model 0.253-0.257  (-30%)
  * all non-simultaneous launches (n=20): baseline 0.883, this model 0.524
  * all comparables (n=37): baseline 0.930, this model 0.554
  * console-only rolling launches (n=4): baseline ~0.30, this model 0.185

The two ingredients both matter.  The rolling-midnight median alone says
viewership dips to ~0.75x hour 1 around hour 4 (overnight UTC), stays near
1x through the morning, then rises to ~2.4x by hours 19-20 (US daytime /
evening) and eases to ~1.9x by hour 24.  The log(p1) regression matters
because bigger hour-1 prints rise less: the per-hour slope on log(p1) is
about -0.4 at the peak hours (correlation about -0.7 within the rolling
pool, -0.9 across all launches) -- a launch whose hour 1 is already huge
has largely peaked, while a small hour 1 still has its unlock wave ahead
of it.  Blending the regression 50/50 with the median (instead of using it
whole) keeps the forecast from extrapolating flat when GTA VI's hour-1
print is larger than anything in the pool, and the 1.25x floor is the same
insurance for the US unlock that is still pending at the checkpoint.

Pool selection: a comparable is used when its unlock text says
"Tier: comparable" and describes a midnight-local / rolling unlock, and it
is not early-access-contaminated or weak.  The Grand Theft Auto VI
category's own pre-release history (in every case's comparables) is not a
launch and is excluded by name.  If the rolling pool has fewer than 6
launches, I fall back to the non-simultaneous comparable pool, then to all
comparables; if there is no usable pool at all, I fall back to the house
baseline formula so the output is always valid.

Input / output (per RULES.txt): read every <id>.in in $MAKEFUN_INPUT_DIR
(else /input), write <id>.out to $MAKEFUN_OUTPUT_DIR (else /output) as
{"hours": [{"hour": h, "q": [q05, q25, q50, q75, q95]}, ...]} for h = 2..24,
quantiles rounded to 2 decimals half-up, strictly increasing by
construction, each in (0, 1e9].  Deterministic: no randomness, no time,
no network, no files outside the case.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
# Regression shrinkage, spread floor and peak floor (validated, see above).
SHRINK = 0.5
SD_FLOOR = 0.45
PEAK_FLOOR_LOG = math.log(1.25)
PEAK_FLOOR_FROM_HOUR = 14
LOG_P1_REF = 12.0  # log(162,755); regression is centred near here.


def _parse_time(s):
    return datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def _median(values):
    v = sorted(values)
    n = len(v)
    if n == 0:
        return 0.0
    mid = n // 2
    return v[mid] if n % 2 else 0.5 * (v[mid - 1] + v[mid])


def _pstdev(values, mean):
    if len(values) < 2:
        return 0.0
    return math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _linreg(xs, ys):
    """Least-squares y ~ a + b*x; returns (a, b, residual_sd)."""
    n = len(xs)
    if n < 3:
        return (_mean(ys), 0.0, 0.5)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx < 1e-9:
        return (my, 0.0, _pstdev(ys, my))
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    return (a, b, _pstdev(resid, 0.0))


def _round2(x):
    """Round to 2 decimals, half up, as the starter does (JS Math.round)."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if float(value).is_integer() else value


def _comparable_trajectories(case):
    """Return {name: (p1, {hour: log(viewers_h / p1)})} for usable launches."""
    out = {}
    for comp in case.get("comparables") or []:
        name = str(comp.get("name") or "")
        if "Grand Theft Auto VI" in name:
            continue  # the target's own pre-release history, not a launch
        unlock = str(comp.get("unlock") or "").lower()
        hourly = comp.get("hourly") or []
        if not hourly:
            continue
        by_time = {}
        for row in hourly:
            try:
                viewers = float(row["viewers"])
                by_time[_parse_time(row["from"])] = viewers
            except (KeyError, TypeError, ValueError):
                continue
        try:
            day_start = _parse_time(comp["day_start"])
        except (KeyError, TypeError, ValueError):
            continue
        p1 = by_time.get(day_start)
        if not p1 or p1 <= 0:
            continue
        traj = {}
        for h in range(2, 25):
            y = by_time.get(day_start + timedelta(hours=h - 1))
            if y and y > 0:
                traj[h] = math.log(y / p1)
        if len(traj) < 15:
            continue
        out[name] = {
            "p1": p1,
            "traj": traj,
            "rolling": ("midnight local" in unlock or "rolling" in unlock),
            "comparable": "tier: comparable" in unlock,
            "early": "early-access-contaminated" in unlock,
            "weak": "tier: weak" in unlock,
            "simultaneous": (
                "simultaneous" in unlock or "single global" in unlock
            ),
        }
    return out


def _pick_pool(comps):
    """Rolling-midnight comparable launches first, with fallbacks."""
    rolling = [
        c for c in comps.values()
        if c["comparable"] and c["rolling"] and not c["early"] and not c["weak"]
    ]
    if len(rolling) >= 6:
        return rolling
    nonsim = [
        c for c in comps.values()
        if c["comparable"] and not c["simultaneous"]
        and not c["early"] and not c["weak"]
    ]
    if len(nonsim) >= 6:
        return nonsim
    all_comp = [c for c in comps.values() if c["comparable"]]
    if len(all_comp) >= 6:
        return all_comp
    return rolling or nonsim or all_comp


def _baseline_quantiles(p1, h):
    return [
        _round2(p1 * math.exp(z * 0.15 * math.sqrt(h - 1))) for z in Z
    ]


def forecast_case(case):
    if case.get("levels") != LEVELS:
        raise ValueError("case asks for levels %r, not %r"
                         % (case.get("levels"), LEVELS))
    first = next(
        (k for k in case.get("known") or [] if k.get("hour") == 1), None)
    if first is None:
        raise ValueError("case has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))

    pool = _pick_pool(_comparable_trajectories(case))
    hours = [row["hour"] for row in case.get("forecast") or []]
    if not hours:
        hours = list(range(2, 25))

    centres = {}
    spreads = {}
    if pool:
        x_target = math.log(p1) - LOG_P1_REF
        for h in hours:
            vals = [c["traj"][h] for c in pool if h in c["traj"]]
            if len(vals) < 4:
                centres[h] = 0.0
                spreads[h] = 0.6
                continue
            med = _median(vals)
            xs = [math.log(c["p1"]) - LOG_P1_REF
                  for c in pool if h in c["traj"]]
            a, b, resid_sd = _linreg(xs, vals)
            mu = a + b * x_target
            centre = (1.0 - SHRINK) * med + SHRINK * mu
            if h >= PEAK_FLOOR_FROM_HOUR:
                centre = max(centre, PEAK_FLOOR_LOG)
            centres[h] = centre
            spreads[h] = max(resid_sd, SD_FLOOR)
        # Smooth centres over a 3-hour window (the diurnal shape is smooth;
        # per-hour medians on ~11 launches are noisy).
        raw = [centres[h] for h in hours]
        for i, h in enumerate(hours):
            lo = max(0, i - 1)
            hi = min(len(raw), i + 2)
            centres[h] = sum(raw[lo:hi]) / (hi - lo)
            if h >= PEAK_FLOOR_FROM_HOUR:
                centres[h] = max(centres[h], PEAK_FLOOR_LOG)

    rows = []
    for h in hours:
        if pool and h in centres:
            qs = [
                _round2(p1 * math.exp(centres[h] + z * spreads[h]))
                for z in Z
            ]
        else:
            qs = _baseline_quantiles(p1, h)
        # Enforce the output contract: (0, 1e9], non-decreasing.
        fixed = []
        prev = 0.0
        for q in qs:
            q = min(max(float(q), 0.01), 1_000_000_000.0)
            q = max(q, prev)
            fixed.append(_round2(q) if float(q) != int(q) else int(q))
            prev = float(fixed[-1])
        rows.append({"hour": h, "q": fixed})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
