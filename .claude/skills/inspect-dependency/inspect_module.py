"""Print the source of an installed dependency, for exploring an API with no
local checkout to read.

`call_it_what_you_want` and `endgame` are installed as opaque site-packages
here -- there's no repo to `grep`, so the fast way to answer "does this class
already have X" is to import it and dump its source rather than guess from
package docs that may be stale.

Restricted to this project's own dependencies rather than any importable
name: this runs whatever `pip` put on the path, so scoping it to modules
cassandra actually declared a dependency on keeps it from becoming a generic
"eval anything importable" tool.

    python3 inspect_module.py call_it_what_you_want
    python3 inspect_module.py call_it_what_you_want.Teams
    python3 inspect_module.py endgame.types.NcaaFbGroup
"""

import argparse
import importlib
import inspect

_ALLOWED_TOP_LEVEL = {
    "call_it_what_you_want",
    "endgame",
    "endgame_aws",
    "cassandra",
}


def _resolve(dotted: str) -> object:
    """Import as much of `dotted` as is a module, then walk the rest as attributes."""
    parts = dotted.split(".")
    if parts[0] not in _ALLOWED_TOP_LEVEL:
        raise ValueError(
            f"{parts[0]!r} is not one of this project's dependencies "
            f"({sorted(_ALLOWED_TOP_LEVEL)}); refusing to import it"
        )
    module = None
    consumed = 0
    for end in range(len(parts), 0, -1):
        candidate = ".".join(parts[:end])
        try:
            module = importlib.import_module(candidate)
            consumed = end
            break
        except ImportError:
            continue
    if module is None:
        raise ImportError(f"couldn't import any prefix of {dotted!r}")
    obj = module
    for name in parts[consumed:]:
        obj = getattr(obj, name)
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        help="dotted path, e.g. call_it_what_you_want.Teams or endgame.types",
    )
    args = parser.parse_args()

    obj = _resolve(args.target)

    print(f"# {args.target} -> {obj!r}")
    file = inspect.getsourcefile(obj) if inspect.ismodule(obj) else None
    if file:
        print(f"# file: {file}")
    if inspect.ismodule(obj):
        names = [n for n in dir(obj) if not n.startswith("_")]
        print(f"# public names: {names}")
        return
    try:
        print(inspect.getsource(obj))  # ty: ignore[invalid-argument-type]
    except (TypeError, OSError) as e:
        print(f"# no source available ({e}); repr/help instead:\n")
        help(obj)


if __name__ == "__main__":
    main()
