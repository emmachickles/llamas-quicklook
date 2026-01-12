# utils/paths.py
from pathlib import Path

def get_repo_root() -> Path:
    """Return the project root directory (parent of utils/)."""
    return Path(__file__).parent.parent

def p(*parts) -> Path:
    return get_repo_root().joinpath(*parts)

def plots_path(*parts) -> Path:
    return p("plots", *parts)

def config_path(*parts) -> Path:
    return p("config", *parts)