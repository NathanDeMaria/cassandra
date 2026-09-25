"""One replay per model, kept on disk, so asking it a question costs seconds.

Every tool that asks where a model is wrong -- `diagnose.py`,
`team_seasons.py`, `evidence.py`, `betting.py` -- starts from the same
thing: the model replayed over its league's whole history, one forecast per
game. That replay is the expensive part (minutes on ncaafb, longer for the
play-by-play models) and it is the same replay every time, so it is done once
and kept at `~/.cassandra/replays/<league>/<model>.parquet`. The questions
asked of it are cheap and get asked many times: a new hypothesis is a new
join against the same residuals, not a new replay.

A kept replay is reused only while it is still the replay the model would
produce. `replay_key` fingerprints everything that decides that -- the
config, the search config whose priors the replay warms up under, the code
that replays it, and the pinned dependencies (`poetry.lock`) that supply its
team names and play-by-play indexes -- and a replay whose key differs is
replayed again. What the key can't see is new data: a Saturday's games
arrive in s3 without anything in the repo changing. So a replay also expires
after `DEFAULT_MAX_AGE`, and every tool says how old the one it read is.

The replay is the one `diagnose.py` has always made -- `get_predictions`
under the search's own priors -- which is also the one `evaluate` scores.
`betting.py` used to replay cold, from whatever priors file the machine
happened to have, so its picks could differ from the published model's by
the warm start; reading this instead closes that.

`load_odds` keeps the other slow input the same way: every odds pull's
reads, which `betting.py` needs line by line and which took a minute to read
from s3 in September 2026. It has no fingerprint -- nothing in the repo
decides what the pulls said -- only an age, `DEFAULT_ODDS_MAX_AGE`.
"""

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from endgame_aws import Config

from .constants import CASSANDRA_HOME
from .model_eval import get_predictions
from .odds import OddsDatabase, OddsSnapshot

_REPO = Path(__file__).parent.parent
_PACKAGE = Path(__file__).parent

#: The checked-in configs, and the ones an optimize run wrote. A freshly
#: optimized model wins over the checked-in baseline of the same name -- the
#: precedence `evaluate_models` has always used.
AUTHORED_DIR = _REPO / "models"
GENERATED_DIR = CASSANDRA_HOME / "models"
REPLAY_DIR = CASSANDRA_HOME / "replays"

#: How long a replay is trusted to have every game. A day, because in season
#: the scores that matter arrive about that often, and because a tool that
#: silently reads last week's replay is a tool that reports last week.
DEFAULT_MAX_AGE = timedelta(hours=24)

#: Modules that read a replay and never feed one. Left out of the fingerprint
#: so that editing an analysis doesn't throw away the replay it analyses --
#: which, while an analysis is being written, would be every run. Anything
#: not named here is assumed to reach the replay, so a new module errs on the
#: side of replaying again.
_READERS_OF_REPLAYS = frozenset(
    {"betting", "evidence", "replay_cache", "residuals", "team_seasons"}
)


def config_path(league: str, model: str) -> Path:
    """The config a model is replayed from: the optimized result, else the baseline."""
    generated = GENERATED_DIR / league / f"{model}_result.json"
    if generated.exists():
        return generated
    authored = AUTHORED_DIR / league / f"{model}.json"
    if authored.exists():
        return authored
    raise FileNotFoundError(
        f"no config for {league}/{model}: looked at {generated} and {authored}"
    )


def _priors_config(league: str, model: str) -> Path | None:
    authored = AUTHORED_DIR / league / f"{model}.json"
    return authored if authored.exists() else None


def _replay_sources() -> list[Path]:
    return sorted(
        path
        for path in _PACKAGE.rglob("*.py")
        if not path.name.endswith("_test.py")
        and path.name != "conftest.py"
        and not (path.parent == _PACKAGE and path.stem in _READERS_OF_REPLAYS)
    )


def replay_key(league: str, model: str) -> str:
    """A fingerprint of everything that decides what a replay of this model says.

    The config and the priors config by content, not path or mtime, so a
    re-pin that changes nothing keeps the replay and one that changes a
    decimal doesn't. Source files by path and content, so a rename counts.
    """
    digest = hashlib.sha256()
    digest.update(f"{league}/{model}".encode())
    config = config_path(league, model)
    digest.update(config.read_bytes())
    priors = _priors_config(league, model)
    if priors is not None and priors != config:
        digest.update(priors.read_bytes())
    for path in _replay_sources():
        digest.update(str(path.relative_to(_PACKAGE)).encode())
        digest.update(path.read_bytes())
    lock = _REPO / "poetry.lock"
    if lock.exists():
        digest.update(lock.read_bytes())
    return digest.hexdigest()[:16]


class Replay(NamedTuple):
    """A model's forecast on every game, and where it came from.

    `predictions` is `save_predictions`' frame, unscored: add residuals with
    `cassandra.residuals.add_residuals`. `replayed_at` is when the replay
    ran, which is what bounds how recent its games can be.
    """

    league: str
    model: str
    predictions: pd.DataFrame
    replayed_at: datetime
    config: Path
    from_cache: bool

    def describe(self) -> str:
        age = datetime.now(UTC) - self.replayed_at
        hours = age.total_seconds() / 3600
        how = (
            f"cached, replayed {hours:.1f}h ago" if self.from_cache else "replayed now"
        )
        last = pd.to_datetime(self.predictions["date"], utc=True).max()
        return (
            f"{self.league}/{self.model} from {self.config} ({how}); "
            f"{len(self.predictions)} games through {last:%Y-%m-%d}"
        )


def _paths(league: str, model: str) -> tuple[Path, Path]:
    base = REPLAY_DIR / league / model
    return base.with_suffix(".parquet"), base.with_suffix(".json")


def _read_kept(
    league: str, model: str, key: str, max_age: timedelta
) -> tuple[pd.DataFrame, datetime] | None:
    frame_path, meta_path = _paths(league, model)
    if not (frame_path.exists() and meta_path.exists()):
        return None
    try:
        meta = json.loads(meta_path.read_text())
        replayed_at = datetime.fromisoformat(meta["replayed_at"])
        if meta.get("key") != key or datetime.now(UTC) - replayed_at > max_age:
            return None
        return pd.read_parquet(frame_path), replayed_at
    except (OSError, ValueError, KeyError):
        # A half-written or older-format cache is a cache miss, not an error:
        # the replay it stood in for can always be made again.
        return None


def _keep(
    league: str, model: str, key: str, frame: pd.DataFrame, replayed_at: datetime
) -> None:
    frame_path, meta_path = _paths(league, model)
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    # `spread` is float-or-None, which pandas holds as object; parquet wants
    # one type per column, and a float column with NaN is what every reader
    # of it coerces to anyway.
    frame.assign(spread=pd.to_numeric(frame["spread"], errors="coerce")).to_parquet(
        frame_path, index=False
    )
    meta_path.write_text(
        json.dumps({"key": key, "replayed_at": replayed_at.isoformat()}, indent=2)
    )


#: How long a local copy of the odds history is trusted. Shorter than a
#: replay's, because in season the odds job writes every hour and a betting
#: question is usually about this week.
DEFAULT_ODDS_MAX_AGE = timedelta(hours=3)


async def load_odds(
    refresh: bool = False, max_age: timedelta = DEFAULT_ODDS_MAX_AGE
) -> tuple[OddsDatabase, datetime]:
    """Every odds pull's reads, from a local copy if it's fresh enough.

    The whole history is one s3 object per hourly pull -- about a minute to
    read as of September 2026, and growing by a day's worth every day --
    against a second for the local copy. Returns when it was read.
    """
    path = REPLAY_DIR / "odds.parquet"
    if not refresh and path.exists():
        read_at = datetime.fromtimestamp(path.stat().st_mtime, UTC)
        if datetime.now(UTC) - read_at <= max_age:
            try:
                return _odds_from_frame(pd.read_parquet(path)), read_at
            except (OSError, ValueError, KeyError):
                pass
    read_at = datetime.now(UTC)
    odds_db = await OddsDatabase.from_s3(Config.init_from_file().bucket)
    path.parent.mkdir(parents=True, exist_ok=True)
    _odds_to_frame(odds_db).to_parquet(path, index=False)
    return odds_db, read_at


def _odds_to_frame(odds_db: OddsDatabase) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"game_id": game_id, **snapshot._asdict()}
            for game_id, series in odds_db.history().items()
            for snapshot in series
        ],
        columns=["game_id", *OddsSnapshot._fields],
    )


def _odds_from_frame(frame: pd.DataFrame) -> OddsDatabase:
    frame = frame.sort_values(["game_id", "read_at"], kind="stable")
    read_at = [t.to_pydatetime() for t in pd.to_datetime(frame["read_at"], utc=True)]
    prices = [
        [None if np.isnan(v) else float(v) for v in frame[column].to_numpy(float)]
        for column in ("spread", "home_moneyline", "away_moneyline")
    ]
    history: dict[str, list[OddsSnapshot]] = {}
    for game_id, at, spread, home, away in zip(
        frame["game_id"].astype(str), read_at, *prices
    ):
        history.setdefault(game_id, []).append(OddsSnapshot(at, spread, home, away))
    return OddsDatabase.from_history(history)


async def load_replay(
    league: str,
    model: str,
    refresh: bool = False,
    max_age: timedelta = DEFAULT_MAX_AGE,
) -> Replay:
    """The model's replay: the kept one if it is current, else a new one, kept.

    `refresh` replays regardless -- for a question about games that finished
    since the kept replay ran, sooner than `max_age` would notice.
    """
    config = config_path(league, model)
    key = replay_key(league, model)
    if not refresh:
        kept = _read_kept(league, model, key, max_age)
        if kept is not None:
            frame, replayed_at = kept
            return Replay(league, model, frame, replayed_at, config, from_cache=True)

    replayed_at = datetime.now(UTC)
    frame = await get_predictions(
        config,
        league,
        GENERATED_DIR / league / f"{model}_diagnose_state.json",
        priors_from=_priors_config(league, model),
        # Its own priors file, beside the replay, not the class's shared one
        # under ~/.cassandra/predictor/data: the warm-up deletes and rewrites
        # the file it builds, and on a laptop running several sessions the
        # shared one is being read by somebody else's replay.
        priors_path=REPLAY_DIR / league / f"{model}.priors.json",
    )
    _keep(league, model, key, frame, replayed_at)
    return Replay(league, model, frame, replayed_at, config, from_cache=False)
