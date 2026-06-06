from __future__ import annotations

import fnmatch
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "dist"
OUTPUT_ZIP = OUTPUT_DIR / "realmlp-tabarena-kaggle-package.zip"


INCLUDE_ROOTS = [
    "src",
    "data",
    "notebooks",
    "docs",
]

INCLUDE_FILES = [
    "requirements.txt",
    "pyproject.toml",
    "README.md",
]


EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "cache",
    "__pycache__",
    ".ipynb_checkpoints",
    "logs",
    "dist",
    "results/autogluon_models",
}

EXCLUDE_FILES = {
    ".env",
    ".env.local",
    ".DS_Store",
}

EXCLUDE_PATTERNS = [
    "*.pyc",
    "*.pyo",
    "*.pkl",
    "*.joblib",
    "*.ubj",
    "*.log",
    "*.tmp",
    "*.zip",
]


def as_posix_relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def is_excluded(path: Path) -> bool:
    rel = as_posix_relative(path)

    if path.name in EXCLUDE_FILES:
        return True

    for excluded_dir in EXCLUDE_DIRS:
        if rel == excluded_dir or rel.startswith(f"{excluded_dir}/"):
            return True

    for pattern in EXCLUDE_PATTERNS:
        if fnmatch.fnmatch(path.name, pattern):
            return True

    return False


def iter_files_to_package() -> list[Path]:
    files: list[Path] = []

    for root_name in INCLUDE_ROOTS:
        root = PROJECT_ROOT / root_name
        if not root.exists():
            continue

        for path in root.rglob("*"):
            if path.is_file() and not is_excluded(path):
                files.append(path)

    for file_name in INCLUDE_FILES:
        file_path = PROJECT_ROOT / file_name
        if file_path.exists() and file_path.is_file() and not is_excluded(file_path):
            files.append(file_path)

    return sorted(set(files), key=lambda p: as_posix_relative(p))


def build_zip() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()

    files = iter_files_to_package()

    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in files:
            arcname = as_posix_relative(file_path)
            zip_file.write(file_path, arcname)

    print(f"Pacote criado: {OUTPUT_ZIP}")
    print(f"Total de arquivos incluídos: {len(files)}")

    print("\nArquivos incluídos:")
    for file_path in files:
        print(f"- {as_posix_relative(file_path)}")


if __name__ == "__main__":
    build_zip()
