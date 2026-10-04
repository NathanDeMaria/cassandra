"""What a search consumed, so a run can tell whether repeating it is pointless.

A search is deterministic: `optimize` fixes `random_state`, and a replay is
the same arithmetic over the same games every time. So if the data, the
config, the code and the starting point are all unchanged since a fit was
written, re-searching cannot reach a different answer -- it will spend hours
arriving back where it started.

That is not a hypothetical. Basketball and hockey have no new games between
spring and November, and the 2026-10-04 run spent 10.4 of its 75.6
instance-hours on `mens`, `womens`, `ncaawvb` and `wnba` -- 14% of the
stage, including the second-longest child in the whole run -- recomputing
answers that could not have moved.

The shape stored beside a fit is `predictor.config.InputFingerprint`; this is
where the four digests come from.

## What counts as an input

`data` is the league's stored seasons plus the per-league predictor indexes,
digested from their s3 keys and ETags. ETags rather than modification times
because a re-uploaded identical file should not invalidate anything, and
because a listing gives them away for free -- the whole check is one
`list_objects_v2` per prefix, which is why it can run before the seasons are
read rather than after.

Two deliberate exclusions:

- *Other leagues' indexes.* The daily sweeps rewrite ncaafb's and nfl's
  every day, so a fingerprint covering the whole prefix would invalidate
  basketball nightly and skip nothing, ever. Scoped by the `<league>_`
  filename convention the sweeps already write under.
- *Priors.* `{league}_{Class}_priors.json` is written *by the search*, as a
  warm-up pass. Including it would mean every run invalidated its own
  fingerprint and nothing ever matched. They are derived from the seasons
  and the pins, both of which are already covered, so excluding them loses
  nothing.

`config` is the checked-in model config, canonicalised so reformatting it
isn't a change. Bounds, pins, frame, `n_iter`, seeds and the objective all
live there.

`code` is the commit the image was built from, which the image carries in
`CASSANDRA_GIT_SHA`. Coarse on purpose -- see `InputFingerprint`.

`seed` is the fit the search starts from, which is an input exactly like the
others. It has one consequence worth knowing: a run that *improves* on its
seed changes this digest, so the following run must repeat the search once
more before anything settles. A search only becomes skippable once it has
reproduced its own answer, which is the right bar.
"""

import hashlib
import json
import os
from collections.abc import AsyncIterator
from pathlib import Path

from aiobotocore.session import get_session

from cassandra.batch.artifacts import ARTIFACT_PREFIX, PREDICTOR_DATA_PREFIX
from cassandra.predictor import InputFingerprint, PredictorConfig

# Where the image records what it was built from. Set in the runtime stage of
# the Dockerfile from a build arg; absent in a source checkout, which is why
# a laptop run never skips.
GIT_SHA_ENV_VAR = "CASSANDRA_GIT_SHA"

# Season pickles are `seasons/<year>/<league>.pkl`, written by endgame rather
# than by anything here -- see `save_predictions.read_all_seasons`, which
# filters the same listing the same way.
_SEASONS_PREFIX = "seasons/"

# A predictor data file this league reads. The sweeps and the anchor fit both
# name their outputs `<league>_<what>.json`.
_PRIORS_SUFFIX = "_priors.json"


def code_version() -> str:
    """The commit this build came from, or "" when it wasn't recorded."""
    return os.environ.get(GIT_SHA_ENV_VAR, "").strip()


async def of_inputs(
    league: str,
    bucket: str,
    config_path: Path,
    previous: PredictorConfig | None,
) -> InputFingerprint:
    """The fingerprint of what a search of `config_path` is about to read."""
    return InputFingerprint(
        data=await _data_digest(league, bucket),
        config=_config_digest(config_path),
        code=code_version(),
        seed=_seed_digest(previous),
    )


async def _data_digest(league: str, bucket: str) -> str:
    """Digest of every stored object this league's search reads.

    Sorted before hashing: s3 returns keys in lexicographic order per page,
    but pagination order is not something to depend on for a value that has
    to be stable across runs.
    """
    session = get_session()
    async with session.create_client("s3") as client:
        entries = [entry async for entry in _relevant_objects(client, bucket, league)]
    return _digest(sorted(entries))


async def _relevant_objects(
    client, bucket: str, league: str
) -> AsyncIterator[tuple[str, str]]:
    """`(key, etag)` for the seasons and indexes of one league."""
    paginator = client.get_paginator("list_objects_v2")
    prefixes = (
        _SEASONS_PREFIX,
        f"{ARTIFACT_PREFIX}/{PREDICTOR_DATA_PREFIX}",
    )
    for prefix in prefixes:
        async for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                key = item["Key"]
                if _reads(key, league):
                    yield key, item["ETag"].strip('"')


def _reads(key: str, league: str) -> bool:
    """Whether a search of `league` reads the object at `key`.

    Both prefixes hold every league's files, so both are filtered. A season
    is `seasons/<year>/<league>.pkl`; an index is `<league>_<what>.json`,
    minus the priors the search writes itself.
    """
    name = key.rsplit("/", 1)[-1]
    if key.startswith(_SEASONS_PREFIX):
        return name == f"{league}.pkl"
    if not name.startswith(f"{league}_"):
        return False
    return not name.endswith(_PRIORS_SUFFIX)


def _config_digest(config_path: Path) -> str:
    """Digest of the model config, insensitive to how the json is laid out."""
    return _digest(json.loads(config_path.read_text()))


def _seed_digest(previous: PredictorConfig | None) -> str:
    """Digest of the fit a search would start from.

    Only the parts `optimize._seeds` actually reads. Not the whole file: it
    carries the previous fingerprint, so hashing it would make this value
    depend on itself.
    """
    if previous is None:
        return ""
    return _digest(
        {
            "params": previous.params,
            "search": None if previous.search is None else previous.search.model_dump(),
            "predictor_class": previous.predictor_class,
        }
    )


def _digest(value: object) -> str:
    """A stable short digest of any json-able value."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()[:16]
