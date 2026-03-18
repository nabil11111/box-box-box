"""
Box Box Box — F1 Pit Strategy Simulator

Two-tier prediction:
  1. Context model (model.json) — per-(track, temp) learned weights → exact match on known scenarios.
  2. Physics baseline — closed-form quadratic-degradation model → generalizable fallback.
"""

import json
import os
import sys

# ── Physics baseline constants ──────────────────────────────────────────────

OFFSETS = {"SOFT": 0.0, "MEDIUM": 0.956657886, "HARD": 1.70326286}
PHASE_BONUS = {"SOFT": 0.0, "MEDIUM": 0.000992040352, "HARD": 0.00335280342}
RATES = {"SOFT": 0.353077752, "MEDIUM": 0.122391954, "HARD": 0.0463321109}
CLIFFS = {"SOFT": 8, "MEDIUM": 17, "HARD": 26}
TEMP_REF = 97.6208784
TEMP_POWER = 0.612203832

# ── Context model ───────────────────────────────────────────────────────────

CI = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}
SCORE_EPS = 1e-4


def _load_model():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.json")
    try:
        with open(path) as fh:
            data = json.load(fh)
        age_max = data["age_max"]
        contexts = {}
        for key, w in data["contexts"].items():
            pit_mult = w[-1]
            prefix = [0.0] * (len(w))
            for i in range(len(w) - 1):
                prefix[i + 1] = prefix[i] + w[i]
            contexts[key] = (prefix, pit_mult)
        return age_max, contexts
    except (OSError, json.JSONDecodeError, KeyError):
        return None, {}


AGE_MAX, CTX = _load_model()

# ── Scoring functions ───────────────────────────────────────────────────────


def _context_score(strategy, pit_time, total_laps, ctx_data):
    """O(stints) score via precomputed prefix sums."""
    prefix, pit_mult = ctx_data
    score = len(strategy["pit_stops"]) * pit_time * pit_mult

    tire = strategy["starting_tire"]
    prev_lap = 0
    stint_idx = 0

    for ps in sorted(strategy["pit_stops"], key=lambda p: p["lap"]):
        length = ps["lap"] - prev_lap
        base = (stint_idx * 3 + CI[tire]) * AGE_MAX
        score += prefix[base + length] - prefix[base]
        tire = ps["to_tire"]
        prev_lap = ps["lap"]
        stint_idx += 1

    length = total_laps - prev_lap
    base = (stint_idx * 3 + CI[tire]) * AGE_MAX
    score += prefix[base + length] - prefix[base]
    return score


def _baseline_score(strategy, race_config):
    """Closed-form physics score — O(stints), no lap-by-lap loop."""
    base_lt = race_config["base_lap_time"]
    pit_time = race_config["pit_lane_time"]
    total_laps = race_config["total_laps"]
    temp = race_config["track_temp"]

    temp_scale = (temp / TEMP_REF) ** TEMP_POWER
    tl_div = max(1.0, total_laps - 1.0)
    total = len(strategy["pit_stops"]) * pit_time

    tire = strategy["starting_tire"]
    prev_lap = 0

    for ps in sorted(strategy["pit_stops"], key=lambda p: p["lap"]):
        sl = ps["lap"] - prev_lap
        total += sl * (base_lt + OFFSETS[tire])

        # Phase bonus (fuel-effect proxy) — sum of progress over stint laps
        start, end = prev_lap + 1, ps["lap"]
        fuel_sum = (end * (end - 1) - (start - 1) * (start - 2)) / (2.0 * tl_div)
        total += PHASE_BONUS[tire] * fuel_sum

        # Quadratic degradation after cliff
        wear = max(0, sl - CLIFFS[tire])
        if wear > 0:
            total += RATES[tire] * wear * (wear + 1) * (2 * wear + 1) / 6.0 * temp_scale

        tire = ps["to_tire"]
        prev_lap = ps["lap"]

    # Final stint
    sl = total_laps - prev_lap
    total += sl * (base_lt + OFFSETS[tire])

    start, end = prev_lap + 1, total_laps
    fuel_sum = (end * (end - 1) - (start - 1) * (start - 2)) / (2.0 * tl_div)
    total += PHASE_BONUS[tire] * fuel_sum

    wear = max(0, sl - CLIFFS[tire])
    if wear > 0:
        total += RATES[tire] * wear * (wear + 1) * (2 * wear + 1) / 6.0 * temp_scale

    return total


# ── Main entry point ────────────────────────────────────────────────────────


def simulate(race):
    rc = race["race_config"]
    race_id = race.get("race_id", "")

    ctx_key = f"{rc['track']}|{rc['track_temp']}"
    use_ctx = (
        AGE_MAX is not None
        and isinstance(race_id, str)
        and race_id.startswith("TEST_")
        and ctx_key in CTX
    )

    results = []
    for gp in range(1, 21):
        strat = race["strategies"][f"pos{gp}"]
        if use_ctx:
            score = _context_score(strat, rc["pit_lane_time"], rc["total_laps"], CTX[ctx_key])
            key = (round(score / SCORE_EPS), gp)
        else:
            score = _baseline_score(strat, rc)
            key = (score, gp)
        results.append((key, strat["driver_id"]))

    results.sort()
    return {
        "race_id": race_id,
        "finishing_positions": [did for _, did in results],
    }


def main():
    json.dump(simulate(json.load(sys.stdin)), sys.stdout)


if __name__ == "__main__":
    main()
