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
    assert _environment(request)[dag.manifest.ARRAY_INDEX_ENV_VAR] == "0"
    assert _MIN_ARRAY_SIZE == 2


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


def test_dry_run_never_builds_a_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """The point of a dry run is checking a submission without AWS set up."""

    def _explode(*args, **kwargs):
        raise AssertionError("dry run touched the batch client")

    monkeypatch.setattr(dag, "get_session", _explode)

    submitted = asyncio.run(
        dag.submit(
            optimize_job_definition="o",
            evaluate_job_definition="e",
            publish_job_definition="p",
            job_queue="queue",
            leagues=["mens"],
            dry_run=True,
        )
    )

    assert [job.name.rsplit("-", 2)[0] for job in submitted] == [
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
