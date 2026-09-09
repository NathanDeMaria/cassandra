"""Shared parquet mechanics for the per-model artifacts.

`history.parquet` and `predictions.parquet` are different tables with the
same handling: fixed dtypes, a deterministic row order, an atomic full
rewrite from the replay, and an idempotent upsert from the refresh. That
handling lives here once rather than twice.

Nothing here imports pyarrow at module scope, and that is the point.
`cassandra.serving` is the half of the package a webapp installs without the
`fit` group, and pyarrow is ~152MB -- the pyproject says so, at length,
about `endgame_aws`. pandas resolves the engine inside `to_parquet` and
`read_parquet`, so a consumer can import these schemas, build rows and hand
them to a writer without pyarrow ever being touched; only the reads and
writes below need it, and only the fitting side calls those.

The dtypes are pinned rather than inferred because an upsert is a
read-modify-write: the frame that comes back off disk has to line up with
the one built in memory, or `pd.concat` silently widens a column to object
and the next write changes the file's schema. Pinning them is also what
makes a re-upsert of unchanged rows byte-identical, which is the property
that lets the refresh job run on a schedule without churning the object.
"""

import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pandas as pd

# A column whose values are datetimes, handled apart from the plain
# `astype` path: the season pickles carry naive datetimes in some leagues
# and aware ones in others (see `publish.rollover_due`), and `astype` on a
# mix of the two raises rather than picking one.
DATETIME = "datetime"

# What a column's declared type means here. `object` for strings rather
# than pandas' StringDtype, because object is what `read_parquet` hands
# back by default -- declaring StringDtype would make every round trip a
# dtype change, which is exactly what the pinning is here to prevent.
Dtypes = Mapping[str, str]


def _as_datetime(series: pd.Series) -> pd.Series:
    """Datetimes, with a mixed naive/aware column resolved to UTC.

    Uniform input keeps whatever it had: converting a league whose games
    are stored naive into UTC would move every timestamp by the offset
    nobody recorded. Only a genuine mix -- which `pd.to_datetime` reports
    by handing back an object column -- is forced, since concatenating one
    is otherwise a TypeError deep inside the upsert.
    """
    converted = pd.to_datetime(series)
    if converted.dtype == object:
        return pd.to_datetime(series, utc=True)
    return converted


def normalized(frame: pd.DataFrame, dtypes: Dtypes) -> pd.DataFrame:
    """`frame` as exactly the declared columns, in declared order and type.

    Extra columns are dropped and the order is taken from `dtypes`, so the
    file's schema is a property of the schema module rather than of whatever
    the caller happened to assemble.
    """
    out = pd.DataFrame(index=frame.index)
    for column, dtype in dtypes.items():
        series = frame[column]
        if dtype == DATETIME:
            out[column] = _as_datetime(series)
        else:
            # pandas-stubs overloads `astype` on literal dtype strings, so
            # a dtype read out of a mapping matches none of them. The cast
            # is about that, not about the value being unchecked -- an
            # unknown dtype still raises here at runtime.
            out[column] = series.astype(cast(Any, dtype))
    return out


def empty(dtypes: Dtypes) -> pd.DataFrame:
    """The zero-row frame with the right schema.

    What a read of a file that doesn't exist yet returns, so the first
    upsert concatenates against something with the same columns instead of
    special-casing the empty case.
    """
    return pd.DataFrame(
        {
            column: pd.Series(
                dtype=cast(Any, "datetime64[ns]" if dtype == DATETIME else dtype)
            )
            for column, dtype in dtypes.items()
        }
    )


def ordered(frame: pd.DataFrame, sort_by: Sequence[str]) -> pd.DataFrame:
    """One canonical row order, so the same rows are the same bytes.

    `kind="stable"`, and every caller sorts by something that totally
    orders its rows -- see each schema's `_SORT`. Both matter for the
    idempotency the refresh depends on: a tie broken by input order would
    make a re-upsert reorder rows it didn't change.
    """
    return frame.sort_values(list(sort_by), kind="stable").reset_index(drop=True)


def to_bytes(frame: pd.DataFrame) -> bytes:
    """The file's exact contents, in memory.

    Bytes rather than a path because the same object goes two places -- a
    local directory and an s3 key -- and serializing twice would leave no
    guarantee they match. `pandas.to_parquet` with no path returns exactly
    what it would have written to one; there's a test holding that.
    """
    payload = frame.to_parquet(index=False)
    # `to_parquet` is typed as returning `bytes | None` because it returns
    # None when handed a path. It isn't, here.
    assert payload is not None
    return payload


def write_bytes(payload: bytes, path: Path) -> None:
    """Replace `path` with `payload`, atomically.

    Written to a sibling temp file and renamed, because both writers
    rewrite the whole object: without this a reader mid-write sees a
    truncated parquet, and a crashed full replay leaves one behind. The
    rename is what makes "the full replay rewrites the file end to end"
    safe to say.

    That is also the whole of the concurrency story, and it's worth being
    explicit that it's last-writer-wins rather than a lock. If a full
    replay and a daily refresh overlap, one of them lands whole and the
    other is lost -- never a blend of the two. The `run_id` on every row
    says which run's numbers are in the file, so the outcome is at least
    legible after the fact.

    s3 gets the same guarantee for free: a `put_object` is atomic, and it
    is the same `payload` these bytes are, so the object in the bucket and
    the file on disk are byte-identical rather than two serializations that
    ought to agree.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # `dir=path.parent`, not the system temp dir: `os.replace` is only
    # atomic within a filesystem, and raises outright across one -- which a
    # container with /tmp on a different mount would hit every time.
    #
    # `mkstemp` rather than a name of our own so uniqueness is the stdlib's
    # problem. Two writers must not share a temp file and hand each other
    # half a frame, and a pid isn't enough for that: one process publishing
    # several models writes this path more than once, and pids get reused.
    handle, name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    tmp = Path(name)
    try:
        with os.fdopen(handle, "wb") as f:
            f.write(payload)
        os.replace(tmp, path)
    finally:
        # Only reached with the file still there if something above raised;
        # `os.replace` has renamed it out from under this by then.
        tmp.unlink(missing_ok=True)


def write(frame: pd.DataFrame, path: Path) -> None:
    """Serialize `frame` and replace `path` with it, atomically."""
    write_bytes(to_bytes(frame), path)


def read(path: Path, dtypes: Dtypes, sort_by: Sequence[str]) -> pd.DataFrame:
    """The file, or an empty frame with the right schema if there isn't one."""
    if not path.exists():
        return empty(dtypes)
    return ordered(normalized(pd.read_parquet(path), dtypes), sort_by)


def upserted(
    rows: pd.DataFrame,
    existing: pd.DataFrame,
    key: Sequence[str],
    dtypes: Dtypes,
    sort_by: Sequence[str],
) -> pd.DataFrame:
    """`existing` with `rows` merged in on `key`, newest wins.

    Upsert rather than append because both tables are re-derived rather
    than accumulated: ratings move within a week as its games land, and a
    game gets re-fetched with a corrected score. `keep="last"` puts the
    incoming row on the winning side, which is what makes re-running a
    refresh over a week it already covered a no-op.
    """
    incoming = normalized(rows, dtypes)
    combined = pd.concat([existing, incoming], ignore_index=True)
    return ordered(combined.drop_duplicates(subset=list(key), keep="last"), sort_by)
