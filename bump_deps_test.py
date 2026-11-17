"""Pins move to what the upstream merged, and only the rev on their line changes."""

import base64
from typing import Any

import pytest

import bump_deps

OLD, NEW, AWS_NEW, THIRD = "a" * 40, "b" * 40, "c" * 40, "d" * 40

PYPROJECT = f"""\
[tool.uv.sources]
# A comment that names endgame = {{ git = "x", rev = "{OLD}" }} and stays put
endgame = {{ git = "https://github.com/o/EndGame.git", rev = "{OLD}", subdirectory = "py-endgame" }}
endgame-aws = {{ git = "https://github.com/o/EndGame.git", rev = "{OLD}", subdirectory = "py-endgame-aws" }}
names = {{ git = "https://github.com/o/names.git", rev = "{OLD}" }}
local = {{ path = "../local" }}
"""  # noqa: E501


def _contents(pyproject: str) -> dict[str, str]:
    return {"content": base64.b64encode(pyproject.encode()).decode()}


def _upstream(**overrides: Any):
    """A fake GitHub: every package has a newer commit, and `endgame-aws` pins
    `endgame` at `endgame_pin`."""
    answers: dict[str, Any] = {
        "repos/o/EndGame/commits?per_page=1&path=py-endgame": [{"sha": NEW}],
        "repos/o/EndGame/commits?per_page=1&path=py-endgame-aws": [{"sha": AWS_NEW}],
        "repos/o/names/commits?per_page=1": [{"sha": NEW}],
        f"repos/o/EndGame/contents/py-endgame-aws/pyproject.toml?ref={AWS_NEW}": (
            _contents(
                "[tool.uv.sources]\n"
                'endgame = { git = "https://github.com/o/EndGame.git", '
                f'rev = "{NEW}", subdirectory = "py-endgame" }}\n'
                'names = { git = "https://github.com/o/names.git", '
                f'rev = "{THIRD}" }}\n'
            )
        ),
    }
    answers.update(overrides)

    def get(path: str) -> Any:
        if path in answers:
            return answers[path]
        if "/compare/" in path:
            subject = {"commit": {"message": "A change\n\nwith a body"}}
            return {"status": "behind", "commits": [subject]}
        return None

    return get


def _revs(bumps: list[bump_deps.Bump]) -> dict[str, str]:
    return {bump.pin.name: bump.rev for bump in bumps}


def test_reads_only_commit_pins_of_github_repositories():
    pins = bump_deps.read_pins(PYPROJECT)
    assert [pin.name for pin in pins] == ["endgame", "endgame-aws", "names"]
    assert pins[0] == bump_deps.Pin("endgame", "o/EndGame", OLD, "py-endgame")
    assert pins[2].subdirectory is None and pins[2].group == "names"


def test_each_pin_moves_to_its_latest_commit():
    bumps = bump_deps.plan(PYPROJECT, _upstream())
    assert _revs(bumps) == {"endgame": NEW, "endgame-aws": AWS_NEW, "names": NEW}


def test_a_sibling_follows_the_pin_its_repository_mate_declares():
    """`endgame-aws` still pins the old `endgame`, so `endgame` stays -- but a
    pin of another repository's package there is that package's own business."""
    stale = _contents(
        "[tool.uv.sources]\n"
        'endgame = { git = "https://github.com/o/EndGame.git", '
        f'rev = "{OLD}", subdirectory = "py-endgame" }}\n'
    )
    key = f"repos/o/EndGame/contents/py-endgame-aws/pyproject.toml?ref={AWS_NEW}"
    bumps = bump_deps.plan(PYPROJECT, _upstream(**{key: stale}))
    assert _revs(bumps) == {"endgame-aws": AWS_NEW, "names": NEW}

    followed = bump_deps.plan(PYPROJECT, _upstream())
    assert [bump.follows for bump in followed] == ["endgame-aws", None, None]


def test_a_pin_ahead_of_the_default_branch_is_left_alone():
    ahead = {f"repos/o/names/compare/{NEW}...{OLD}": {"status": "ahead"}}
    bumps = bump_deps.plan(PYPROJECT, _upstream(**ahead))
    assert "names" not in _revs(bumps)


def test_apply_changes_the_rev_and_nothing_else():
    bumps = bump_deps.plan(PYPROJECT, _upstream())
    names_only = [bump for bump in bumps if bump.pin.group == "names"]
    changed = bump_deps.apply(PYPROJECT, names_only)
    assert changed == PYPROJECT.replace(
        f'names.git", rev = "{OLD}"', f'names.git", rev = "{NEW}"'
    )


def test_apply_refuses_a_line_it_cannot_find():
    missing = bump_deps.Bump(bump_deps.Pin("names", "o/names", THIRD), NEW)
    with pytest.raises(ValueError, match="names"):
        bump_deps.apply(PYPROJECT, [missing])


def test_the_pull_request_names_every_pin_and_what_merged():
    get = _upstream()
    bumps = [b for b in bump_deps.plan(PYPROJECT, get) if b.pin.group == "EndGame"]
    title, body = bump_deps.describe(bumps, get)
    assert title == "Bump endgame and endgame-aws"
    assert f"- `endgame` {OLD[:7]} -> {NEW[:7]}" in body
    assert "the rev `endgame-aws` pins it at" in body
    assert "  - A change" in body and "with a body" not in body


CONSUMER = f"""\
[tool.uv.sources]
cassandra = {{ git = "https://github.com/o/cassandra.git", rev = "{OLD}" }}
names = {{ git = "https://github.com/o/names.git", rev = "{OLD}" }}
lucky = {{ git = "https://github.com/o/lucky.git", rev = "{OLD}" }}
other = {{ git = "https://github.com/o/other.git", rev = "{OLD}" }}

[tool.bump-deps]
follow = {{ names = "cassandra", lucky = "cassandra" }}
"""  # noqa: E501


def _consumer_upstream():
    """`cassandra` moved and pins `names` at THIRD but has no `lucky`; neither
    follower's own repository is ever asked for its head."""
    return _upstream(
        **{
            "repos/o/cassandra/commits?per_page=1": [{"sha": NEW}],
            "repos/o/other/commits?per_page=1": [{"sha": NEW}],
            f"repos/o/cassandra/contents/pyproject.toml?ref={NEW}": _contents(
                "[tool.uv.sources]\n"
                f'names = {{ git = "https://github.com/o/names.git", rev = "{THIRD}" }}\n'
            ),
        }
    )


def test_a_declared_follower_takes_its_leaders_pin_in_its_leaders_pull_request():
    get = _consumer_upstream()
    bumps = bump_deps.plan(CONSUMER, get)
    assert _revs(bumps) == {"cassandra": NEW, "names": THIRD, "other": NEW}
    assert [(b.pin.name, b.group, b.follows) for b in bumps] == [
        ("cassandra", "cassandra", None),
        ("names", "cassandra", "cassandra"),
        ("other", "other", None),
    ]
    title, body = bump_deps.describe(bumps[:2], get)
    assert title == "Bump cassandra and names"
    assert body.startswith("[o/cassandra](https://github.com/o/cassandra) has merged")
    assert f"https://github.com/o/names/compare/{OLD}...{THIRD}" in body
    assert "the rev `cassandra` pins it at" in body


def test_a_follow_must_name_two_pins_and_one_leader():
    for follow, problem in [
        ('{ names = "nobody" }', "both must be pins"),
        ('{ names = "cassandra", cassandra = "other" }', "cassandra follows too"),
    ]:
        pyproject = CONSUMER.replace(
            'follow = { names = "cassandra", lucky = "cassandra" }',
            f"follow = {follow}",
        )
        with pytest.raises(ValueError, match=problem):
            bump_deps.plan(pyproject, _consumer_upstream())


def test_the_cli_bumps_the_pyproject_it_is_pointed_at(tmp_path, monkeypatch, capsys):
    path = tmp_path / "pyproject.toml"
    path.write_text(CONSUMER)
    monkeypatch.setattr(bump_deps, "github", _consumer_upstream())
    assert bump_deps.main(["--pyproject", str(path), "list"]) == 0
    assert capsys.readouterr().out.split() == ["cassandra", "other"]
    assert bump_deps.main(["--pyproject", str(path), "apply", "cassandra"]) == 0
    assert capsys.readouterr().out.startswith("Bump cassandra and names\n\n")
    assert bump_deps.read_pins(path.read_text())[1].rev == THIRD
