"""What actually goes to Batch.

These assert on the shape of the SubmitJob request rather than on AWS, because
the parts that are easy to get wrong -- the array-of-one branch, and which
stages carry a dependency -- are all in building that dict.
"""

import asyncio

import pytest

from . import dag
from .dag import _MIN_ARRAY_SIZE, _Submitter


class _FakeBatchClient:
    """Records submissions and hands back predictable job ids."""

    def __init__(self) -> None:
        self.requests: list[dict] = []

    async def submit_job(self, **request):
        self.requests.append(request)
        return {"jobId": f"job-{len(self.requests)}"}


def _environment(request: dict) -> dict[str, str]:
    return {
        entry["name"]: entry["value"]
        for entry in request["containerOverrides"]["environment"]
    }


def test_a_multi_item_run_becomes_an_array_job() -> None:
    client = _FakeBatchClient()
    submitter = _Submitter(client, "queue", dry_run=False)

    asyncio.run(
        submitter.run(name="n", job_definition="d", command=["optimize"], size=4)
    )

    request = client.requests[0]
    assert request["arrayProperties"] == {"size": 4}
    # Batch sets the index per child; pinning one here would give every child
    # the same work item.
    assert dag.manifest.ARRAY_INDEX_ENV_VAR not in _environment(request)
    assert dag.manifest.PINNED_INDEX_ENV_VAR not in _environment(request)


def test_a_single_item_run_becomes_a_plain_job_with_the_index_pinned() -> None:
    client = _FakeBatchClient()
    submitter = _Submitter(client, "queue", dry_run=False)

    asyncio.run(
        submitter.run(name="n", job_definition="d", command=["optimize"], size=1)
    )

    request = client.requests[0]
    # Batch rejects arrayProperties.size below 2, and a plain job never gets
    # AWS_BATCH_JOB_ARRAY_INDEX, so the child would not know which item it is.
    assert "arrayProperties" not in request
    assert _environment(request)[dag.manifest.PINNED_INDEX_ENV_VAR] == "0"
    assert _MIN_ARRAY_SIZE == 2
    # Not under the reserved name: Batch accepts an AWS_BATCH_* override and
    # then drops it, which is silent and cost a whole run's publish scope.
    assert dag.manifest.ARRAY_INDEX_ENV_VAR not in _environment(request)
    assert not dag.manifest.PINNED_INDEX_ENV_VAR.startswith("AWS_BATCH")


def test_dependencies_are_passed_as_job_ids() -> None:
    client = _FakeBatchClient()
    submitter = _Submitter(client, "queue", dry_run=False)

    asyncio.run(
        submitter.run(
            name="n",
            job_definition="d",
            command=["publish"],
            size=2,
            depends_on=["abc"],
        )
    )

    assert client.requests[0]["dependsOn"] == [{"jobId": "abc"}]


def test_no_dependency_key_when_there_is_nothing_to_wait_for() -> None:
    client = _FakeBatchClient()
    submitter = _Submitter(client, "queue", dry_run=False)

    asyncio.run(
        submitter.run(name="n", job_definition="d", command=["publish"], size=2)
    )

    # An empty dependsOn list is not the same as omitting it; Batch validates
    # the key when present.
    assert "dependsOn" not in client.requests[0]


def test_dry_run_submits_nothing() -> None:
    client = _FakeBatchClient()
    submitter = _Submitter(client, "queue", dry_run=True)

    submitted = asyncio.run(
        submitter.run(name="n", job_definition="d", command=["optimize"], size=3)
    )

    assert client.requests == []
    assert submitted.size == 3


def _submit(monkeypatch: pytest.MonkeyPatch, **kwargs) -> list[dag.Submitted]:
    """A dry-run submit with the job definitions filled in.

    Dry run because the point of these is the shape of the DAG, and building
    it must not need AWS -- so the batch client is booby-trapped rather than
    faked.
    """

    def _explode(*args, **kwargs):
        raise AssertionError("dry run touched the batch client")

    monkeypatch.setattr(dag, "get_session", _explode)

    return asyncio.run(
        dag.submit(
            anchors_job_definition="a",
            optimize_job_definition="o",
            evaluate_job_definition="e",
            publish_job_definition="p",
            job_queue="queue",
            dry_run=True,
            **kwargs,
        )
    )


def _stages(submitted: list[dag.Submitted]) -> list[str]:
    """Stage names, with `_job_name`'s timestamp suffix trimmed off."""
    return [job.name.rsplit("-", 2)[0] for job in submitted]


def test_dry_run_never_builds_a_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """The point of a dry run is checking a submission without AWS set up."""
    submitted = _submit(monkeypatch, leagues=["mens"])

    assert _stages(submitted) == [
        "cassandra-anchors",
        "cassandra-optimize",
        "cassandra-evaluate",
        "cassandra-publish",
    ]


def test_a_league_with_no_anchors_gets_no_anchors_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """nfl's 32 teams all play each other; there is no tier gap to fit."""
    assert "nfl" not in dag.ANCHOR_LEAGUES
    submitted = _submit(monkeypatch, leagues=["nfl"])

    assert "cassandra-anchors" not in _stages(submitted)


def test_skipping_optimize_skips_the_anchors_it_would_have_fed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What the daily publish submits: republish, don't re-decide the scale."""
    submitted = _submit(monkeypatch, leagues=["mens"], skip_optimize=True)

    assert _stages(submitted) == ["cassandra-evaluate", "cassandra-publish"]


def test_optimize_waits_for_the_anchors() -> None:
    """A search that starts before the anchors land is fit on the wrong scale."""
    client = _FakeBatchClient()
    submitter = _Submitter(client, "queue", dry_run=False)

    anchors = asyncio.run(
        submitter.run(name="a", job_definition="a", command=["anchors"], size=3)
    )
    asyncio.run(
        submitter.run(
            name="o",
            job_definition="o",
            command=["optimize"],
            size=8,
            depends_on=[anchors.job_id],
        )
    )

    assert client.requests[1]["dependsOn"] == [{"jobId": anchors.job_id}]


def test_skipping_anchors_alone_still_optimizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submitted = _submit(monkeypatch, leagues=["mens"], skip_anchors=True)

    assert _stages(submitted) == [
        "cassandra-optimize",
        "cassandra-evaluate",
        "cassandra-publish",
    ]


def test_publish_league_indexes_the_pinned_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CASSANDRA_PUBLISH_LEAGUES", "mens,nfl,womens")

    assert dag.publish_league(0) == "mens"
    assert dag.publish_league(2) == "womens"
    with pytest.raises(IndexError):
        dag.publish_league(3)


def test_publish_league_without_the_list_says_what_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CASSANDRA_PUBLISH_LEAGUES", raising=False)

    with pytest.raises(ValueError, match="CASSANDRA_PUBLISH_LEAGUES"):
        dag.publish_league(0)


def test_anchor_league_reads_its_own_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two fan-outs, two lists: an anchors child must not index publish's."""
    monkeypatch.setenv("CASSANDRA_ANCHOR_LEAGUES", "mens,ncaafb")
    monkeypatch.setenv("CASSANDRA_PUBLISH_LEAGUES", "nfl,womens")

    assert dag.anchor_league(1) == "ncaafb"
    assert dag.publish_league(1) == "womens"


def test_the_anchors_child_gets_the_list_the_array_was_sized_against() -> None:
    """The pinned list is the contract between launcher and child."""
    client = _FakeBatchClient()
    submitter = _Submitter(client, "queue", dry_run=False)

    asyncio.run(
        submitter.run(
            name="a",
            job_definition="a",
            command=["anchors"],
            size=2,
            environment={"CASSANDRA_ANCHOR_LEAGUES": "mens,ncaafb"},
        )
    )

    request = client.requests[0]
    assert request["arrayProperties"] == {"size": 2}
    assert _environment(request)["CASSANDRA_ANCHOR_LEAGUES"] == "mens,ncaafb"


def test_league_args_survive_the_round_trip_through_fire() -> None:
    """
    The launcher writes this command line and `jobs.py` reads it back, so
    the two have to agree about what more than one league looks like.

    Repeated flags don't survive: fire keeps the last and the run scores one
    league while reporting success for all of them.
    """
    import fire

    from jobs import _as_list

    args = dag._league_args(["nhl", "wnba"])
    assert args == ["--league", "nhl,wnba"]

    parsed: list[str] | None = None

    class _Stage:
        def evaluate(self, league=None):
            nonlocal parsed
            parsed = _as_list(league)

    fire.Fire(_Stage, command=["evaluate"] + args)
    assert parsed == ["nhl", "wnba"]


def test_league_args_are_empty_without_leagues() -> None:
    assert dag._league_args(None) == []
    assert dag._league_args([]) == []


class _FakeSession:
    """Enough of the aiobotocore session for `submit` to reach a fake client.

    The dry-run helper above booby-traps this, which is right for the tests
    that only care which stages get submitted. Asserting on a *command*
    needs the request itself, and that only exists on the real path.
    """

    def __init__(self, client: _FakeBatchClient) -> None:
        self._client = client

    def create_client(self, service: str) -> "_FakeSession":
        assert service == "batch"
        return self

    async def __aenter__(self) -> _FakeBatchClient:
        return self._client

    async def __aexit__(self, *exc: object) -> bool:
        return False


def _submitted_requests(
    monkeypatch: pytest.MonkeyPatch, **kwargs
) -> list[dict]:
    """Every SubmitJob request a real (non-dry) submit would send."""
    client = _FakeBatchClient()
    monkeypatch.setattr(dag, "get_session", lambda: _FakeSession(client))
    asyncio.run(
        dag.submit(
            anchors_job_definition="a",
            optimize_job_definition="o",
            evaluate_job_definition="e",
            publish_job_definition="p",
            job_queue="queue",
            **kwargs,
        )
    )
    return client.requests


def _command(requests: list[dict], stage: str) -> list[str]:
    for request in requests:
        command = request["containerOverrides"]["command"]
        if command[0] == stage:
            return command
    raise AssertionError(f"no {stage} job was submitted")


def test_the_anchors_job_leaves_existing_anchors_alone_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A scheduled run must not re-rate a league it was only asked to publish."""
    requests = _submitted_requests(monkeypatch, leagues=["mens"])

    assert _command(requests, "anchors") == ["anchors"]


def test_rebuild_anchors_tells_the_job_to_refit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one way to overwrite anchors from a batch run."""
    requests = _submitted_requests(
        monkeypatch, leagues=["mens"], rebuild_anchors=True
    )

    assert _command(requests, "anchors") == ["anchors", "--if-missing=False"]


@pytest.mark.parametrize("dropped", ["skip_anchors", "skip_optimize"])
def test_rebuilding_anchors_that_would_not_run_is_an_error(
    monkeypatch: pytest.MonkeyPatch, dropped: str
) -> None:
    """Both readings of the contradiction re-rate a league, or don't. Ask."""
    with pytest.raises(ValueError, match="rebuild_anchors"):
        _submit(monkeypatch, leagues=["mens"], rebuild_anchors=True, **{dropped: True})


def test_the_rebuild_flag_survives_the_round_trip_through_fire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The launcher writes `--if-missing=False`; `jobs.py` has to read it as
    a boolean. Parsed as the *string* "False" it is truthy, the anchors job
    leaves the existing file alone, and a run asked to re-rate the league
    reports success having changed nothing.
    """
    import fire

    command = _command(
        _submitted_requests(monkeypatch, leagues=["mens"], rebuild_anchors=True),
        "anchors",
    )

    seen: object = "not called"

    class _Stage:
        def anchors(self, league=None, index=None, if_missing=True, upload=True):
            nonlocal seen
            seen = if_missing

    fire.Fire(_Stage, command=command)
    assert seen is False


# --- what a football league does and doesn't get ---------------------------
# ncaafb is the league to assert against: it is in ANCHOR_LEAGUES and was in
# CONTROL_LEAGUES too, so it is the only one that ever had two parents to
# lose one of.


def _by_command(requests: list[dict]) -> dict[str, dict]:
    """Requests keyed by the stage they run, since submission order shifts.

    Indexing by position breaks the moment a run includes a stage it didn't
    before -- which is exactly what adding and then removing one did.
    """
    return {r["containerOverrides"]["command"][0]: r for r in requests}


def test_a_football_league_gets_no_game_control_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sweep fed `glicko_control`, and both leagues searched its weight
    to zero -- so the models are gone from `models/` and the node that fed
    them is gone from here. `cassandra.predictor.control` has the numbers.
    """
    submitted = _submit(monkeypatch, leagues=["ncaafb"])

    assert "cassandra-game-control" not in _stages(submitted)
    assert _stages(submitted) == [
        "cassandra-anchors",
        "cassandra-optimize",
        "cassandra-evaluate",
        "cassandra-publish",
    ]


def test_optimize_waits_for_the_anchors_and_nothing_else(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One parent, not two.

    The anchors set the scale a search is fit against, and are the only
    upstream stage left that decides anything a child reads. An extra
    dependency here isn't wrong so much as unpayable: it names a job the
    launcher no longer submits, and Batch rejects the whole request.
    """
    requests = _submitted_requests(monkeypatch, leagues=["ncaafb"])
    depends = _by_command(requests)["optimize"]["dependsOn"]

    # The fake client numbers jobs in submission order, and anchors is first.
    assert depends == [{"jobId": "job-1"}]


def test_a_football_republish_waits_for_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--skip-optimize` used to leave the sweep behind for publish to wait
    on, even for a football league. With the sweep gone there is no upstream
    stage left at all, and a republish starts immediately.
    """
    requests = _submitted_requests(monkeypatch, leagues=["ncaafb"], skip_optimize=True)
    by_command = _by_command(requests)

    assert set(by_command) == {"evaluate", "publish"}
    assert "dependsOn" not in by_command["publish"]
    assert "dependsOn" not in by_command["evaluate"]
