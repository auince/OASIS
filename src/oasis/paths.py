import os
from pathlib import Path


def resolve_path(value, reference_file, root=None):
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    base = Path(root).expanduser().resolve() if root is not None else Path(reference_file).resolve().parent
    return (base / path).resolve()


def relative_path(value, output_file):
    return Path(os.path.relpath(Path(value).resolve(), Path(output_file).resolve().parent)).as_posix()
