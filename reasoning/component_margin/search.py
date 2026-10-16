"""A parallel pattern search over `ComponentMarginPredictor`'s knobs.

    python search.py NAME [--from RUN] [--rounds N] [--workers 8]

Each round tries every knob one step up and one step down, eight replays at
a time, and moves to the best of them -- or to all the improving moves at
once, if that is better still. A round with nothing better halves every
step. Scored by brier on FBS v FBS games 2014-2021, the window the
success-rate refits trained on; nothing before 2014 depends on a knob here.
Progress goes to `logs/NAME.jsonl`, and the best knobs are replayed through
2026 at the end as `run.py` would (`cm_NAME`).
"""

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent

#: name: (start, kind, step). "log" steps multiply by exp(step), "unit"
#: knobs live in [0, 1] and step by `step`.
KNOBS: dict[str, tuple[float, str, float]] = {
    "offense_initial_sd": (0.05, "log", 0.3),
    "defense_initial_sd": (0.05, "log", 0.3),
    "initial_rd": (192.27, "log", 0.3),
    "weekly_rd_increase": (3.248, "log", 0.3),
    "season_rd_increase": (85.41, "log", 0.3),
    "points_per_epa": (86.95, "log", 0.2),
    "play_sd": (1.6686, "log", 0.3),
    "rest_initial_sd": (2.0, "log", 0.4),
    "rest_obs_sd": (12.0, "log", 0.3),
    "game_sd": (16.0, "log", 0.1),
    "uncertainty_scale": (1.0, "log", 0.4),
    "obs_sd": (18.74, "log", 0.2),
    "points_per_rating": (0.1403, "log", 0.1),
    "home_advantage": (16.91, "log", 0.2),
    "offense_carry": (1.0, "unit", 0.15),
    "defense_carry": (1.0, "unit", 0.15),
    "rest_carry": (0.0, "unit", 0.15),
    "success_weight": (0.5891, "unit", 0.15),
}

_RUN = None


def evaluate(knobs: dict) -> dict:
    global _RUN
    if _RUN is None:
        sys.path.insert(0, str(HERE))
        import run as module

        _RUN = module
    full = {**knobs, "defense_success_weight": knobs["success_weight"]}
    df = _RUN.run(full, through=2021)
    rows = df[df.fbs & (df.year >= 2014)]
    brier = float(((rows.team1_win_prob - rows.team1_win) ** 2).mean())
    return {"knobs": knobs, "brier": brier, **_RUN.margin_scores(df)}


def moved(knobs: dict, name: str, direction: int, steps: dict) -> dict | None:
    _, kind, _ = KNOBS[name]
    value = knobs[name]
    if kind == "log":
        new = value * math.exp(direction * steps[name])
    else:
        new = min(1.0, max(0.0, value + direction * steps[name]))
        if abs(new - value) < 1e-9:
            return None
    return {**knobs, name: new}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--from", dest="start_from")
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--knobs", help="comma-separated subset to search")
    args = parser.parse_args()
    names = args.knobs.split(",") if args.knobs else list(KNOBS)
    knobs = {n: v for n, (v, _, _) in KNOBS.items()}
    if args.start_from:
        knobs.update(json.loads((HERE / "logs" / f"{args.start_from}.best.json").read_text())["knobs"])
    steps = {n: KNOBS[n][2] for n in KNOBS}
    log = (HERE / "logs" / f"{args.name}.jsonl").open("a")

    with ProcessPoolExecutor(args.workers) as pool:
        best = pool.submit(evaluate, knobs).result()
        log.write(json.dumps({"round": 0, **best}) + "\n")
        log.flush()
        print("start", round(best["brier"], 6), flush=True)
        for rnd in range(1, args.rounds + 1):
            moves = []
            for name in names:
                for direction in (1, -1):
                    candidate = moved(best["knobs"], name, direction, steps)
                    if candidate is not None:
                        moves.append(((name, direction), candidate))
            results = list(pool.map(evaluate, [c for _, c in moves]))
            for (move, _), result in zip(moves, results):
                log.write(json.dumps({"round": rnd, "move": move, **result}) + "\n")
            log.flush()
            improving = [(m, r) for (m, _), r in zip(moves, results) if r["brier"] < best["brier"]]
            if not improving:
                steps = {n: s / 2 for n, s in steps.items()}
                print(f"round {rnd}: no move, steps halved", flush=True)
                if max(steps[n] for n in names) < 0.02:
                    break
                continue
            winner = min(improving, key=lambda mr: mr[1]["brier"])
            # Every improving move at once, one knob each way at most.
            combined = dict(best["knobs"])
            for (name, direction), r in sorted(improving, key=lambda mr: mr[1]["brier"]):
                combined[name] = r["knobs"][name]
            if len({m[0] for m, _ in improving}) > 1:
                both = pool.submit(evaluate, combined).result()
                log.write(json.dumps({"round": rnd, "move": "combined", **both}) + "\n")
                log.flush()
                if both["brier"] < winner[1]["brier"]:
                    winner = ("combined", both)
            best = winner[1]
            print(f"round {rnd}: {winner[0]} -> {best['brier']:.6f} "
                  f"(margin ll {best['margin_ll_train']:.4f}, slope {best['margin_slope']:.3f})", flush=True)
            (HERE / "logs" / f"{args.name}.best.json").write_text(json.dumps(best, indent=2))
    (HERE / "logs" / f"{args.name}.best.json").write_text(json.dumps(best, indent=2))
    print("best", json.dumps(best["knobs"]), flush=True)


if __name__ == "__main__":
    main()
