"""Re-pin the configs that copied a model's fit, from that model's latest fit.

    python sync_pins.py --league ncaafb --source glicko_full
    python sync_pins.py --league ncaafb,nfl --source glicko_full --download

`models/<league>/glicko_blend.json` and `glicko_compound.json` pin the whole
Glicko machinery at `glicko_full`'s fitted values so that their own zero --
`play_weight` 0, `unit_weight` 0 -- *is* `glicko_full`, and a comparison
between the two is one process scoring one model twice rather than two runs
against two slightly different games. That identity holds only while the
pins match the fit, and the fit moves every time `glicko_full` is searched.
The run report's FROZEN PARAMETERS section finds the drift; this is the edit
it asks for, done the same way for every pin at once instead of by hand.

The rule is the report's rule. For every config whose `fixed_from` names
the source: a pinned parameter the source *searches* takes the source's
fitted value, and one the source itself *fixes* takes that fixed value.
"Searches" is read in constructor terms -- a source in the points frame
(`cassandra.predictor.frame`) that searches `hfa_pts` and `sigmoid_scale`
moves `home_advantage`, so a pin of it takes the fitted value.
A pin the source knows nothing about -- `mov_scale`, say -- is the target's
own and is left alone, and nothing is added that the target did not already
pin. Parameters the target searches are never touched, even when the source
fitted them too.

The fit is read from `~/.cassandra/models/<league>/<source>_result.json`,
which is what the evaluate stage downloads and what a hand run writes.
`--download` pulls the league's results from the batch bucket first, which
is what a laptop wants after a Batch run.
"""

import argparse
import asyncio
import json
from pathlib import Path

from endgame_aws import Config

from cassandra.batch import artifacts
from cassandra.constants import CASSANDRA_HOME
from cassandra.predictor import OptimizationConfig, PredictorConfig
from cassandra.predictor import frame as frames

MODELS_DIR = Path(__file__).parent / "models"


def sync(league: str, source: str) -> list[str]:
    """Rewrite every config in `league` pinned from `source`; the lines to report."""
    league_dir = MODELS_DIR / league
    source_config = OptimizationConfig.model_validate_json(
        (league_dir / f"{source}.json").read_text()
    )
    result_path = CASSANDRA_HOME / "models" / league / f"{source}_result.json"
    if not result_path.exists():
        raise FileNotFoundError(
            f"{result_path} -- no fit for {league}/{source} on disk; "
            "pass --download, or run the search"
        )
    result = PredictorConfig.model_validate_json(result_path.read_text())
    fitted = result.params
    # The arguments the source's own pins fix, in constructor terms. A pin
    # the source fixes by name is that value; one it fixes through its frame
    # -- `hfa_pts` with `sigmoid_scale` -- is what the frame makes of it.
    weeks = result.search.weeks_per_season if result.search else 1.0
    source_fixed = frames.to_params(source_config.frame, source_config.fixed, weeks)
    if source_config.frame != frames.RATING and result.search is None:
        # The fit on disk is from before the source moved to this frame; the
        # arguments the frame derives are all "not in the fit yet".
        fitted = {
            name: value
            for name, value in fitted.items()
            if name not in frames.derived_params(source_config.frame)
        }

    lines = []
    for path in sorted(league_dir.glob("*.json")):
        if path.stem.endswith(("_result", "_state")):
            continue
        raw = json.loads(path.read_text())
        if raw.get("fixed_from") != source:
            continue
        changed = []
        waiting = []
        # In the source's constructor terms, whatever frame it searched in:
        # a `glicko_full` that searches `hfa_pts` still moves `home_advantage`,
        # and that is the name the pin here carries.
        searched = source_config.searched_params()
        for name, pinned in raw["fixed"].items():
            if name in searched:
                if name not in fitted:
                    # The source searches it and the fit on disk predates the
                    # dimension -- a parameter added to the search that has
                    # not been run yet. The pin keeps the value it was
                    # written with, which is the default the fit replayed
                    # at, and moves on the run after.
                    waiting.append(name)
                    continue
                current = fitted[name]
            elif name in source_fixed:
                current = source_fixed[name]
            else:
                continue
            if current != pinned:
                raw["fixed"][name] = current
                changed.append(f"{name} {pinned!r} -> {current!r}")
        if changed:
            path.write_text(json.dumps(raw, indent=4) + "\n")
            lines.append(f"{league}/{path.stem}: " + ", ".join(changed))
        else:
            lines.append(f"{league}/{path.stem}: already in sync")
        if waiting:
            lines.append(
                f"{league}/{path.stem}: {', '.join(waiting)} not in the fit yet; "
                f"re-run after the next {source} search"
            )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--league", required=True, help="one league, or several comma-separated"
    )
    parser.add_argument(
        "--source", default="glicko_full", help="the model whose fit the pins copy"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="pull the leagues' results from the batch bucket first",
    )
    args = parser.parse_args()
    leagues = args.league.split(",")

    if args.download:
        bucket = Config.init_from_file().bucket
        for league in leagues:
            pulled = asyncio.run(artifacts.download(bucket, f"models/{league}/"))
            print(f"{league}: {len(pulled)} result file(s) from s3://{bucket}")

    for league in leagues:
        for line in sync(league, args.source):
            print(line)


if __name__ == "__main__":
    main()
