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
    fitted = PredictorConfig.model_validate_json(result_path.read_text()).params

    lines = []
    for path in sorted(league_dir.glob("*.json")):
        if path.stem.endswith(("_result", "_state")):
            continue
        raw = json.loads(path.read_text())
        if raw.get("fixed_from") != source:
            continue
        changed = []
        for name, pinned in raw["fixed"].items():
            if name in source_config.parameters:
                current = fitted[name]
            elif name in source_config.fixed:
                current = source_config.fixed[name]
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
