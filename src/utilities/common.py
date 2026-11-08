"""Shared helpers for the LCI-IEDB pipeline.

Every pipeline script imports this module.  It deliberately has no third-party
dependency beyond the standard library so that it can be imported anywhere.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import platform
import re
import sys
import time
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

AA20: str = "ACDEFGHIKLMNPQRSTVWY"
AA_SET: frozenset[str] = frozenset(AA20)
AA_INDEX: dict[str, int] = {a: i for i, a in enumerate(AA20)}
MIN_LEN: int = 8
MAX_LEN: int = 11
DEFAULT_SEED: int = 42

_TAXON_RE = re.compile(r"NCBITaxon_(\d+)")
_UNIPROT_RE = re.compile(r"uniprot/([A-Za-z0-9_\-]+)")


def repo_root() -> Path:
    """Return the repository root (two levels above ``src/<pkg>/``)."""
    return Path(__file__).resolve().parents[2]


def data_dir(sub: str = "") -> Path:
    """Return ``<repo>/data/<sub>``."""
    return repo_root() / "data" / sub


def setup_logging(script_name: str, verbose: bool = False) -> logging.Logger:
    """Configure console + file logging for a script and return the logger."""
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    log_dir = data_dir("metadata/run_logs")
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / f"{script_name}.log", mode="w")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except OSError:  # logging to file is best-effort
        logger.warning("could not open log file in %s", log_dir)
    return logger


def is_canonical_peptide(seq: str) -> bool:
    """True if ``seq`` is non-empty and contains only the 20 canonical residues."""
    return bool(seq) and all(c in AA_SET for c in seq)


def taxon_id_from_iri(iri: str) -> int | None:
    """Extract an NCBI taxon id from an IEDB/OBO IRI, else ``None``."""
    m = _TAXON_RE.search(iri or "")
    return int(m.group(1)) if m else None


def uniprot_from_iri(iri: str) -> str | None:
    """Extract a UniProt accession from a UniProt IRI, else ``None``."""
    m = _UNIPROT_RE.search(iri or "")
    return m.group(1) if m else None


def sha256_file(path: Path, block: int = 1 << 20) -> str:
    """SHA-256 of a file, streamed."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def package_versions(names: list[str]) -> dict[str, str]:
    """Installed versions of the given packages (``'missing'`` if absent)."""
    out: dict[str, str] = {}
    for n in names:
        try:
            out[n] = importlib_metadata.version(n)
        except importlib_metadata.PackageNotFoundError:
            out[n] = "missing"
    return out


def update_manifest(key: str, payload: dict[str, Any]) -> None:
    """Merge ``payload`` under ``key`` into ``data/metadata/manifest.json``."""
    path = data_dir("metadata/manifest.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        manifest = json.loads(path.read_text()) if path.exists() else {}
    except json.JSONDecodeError:
        manifest = {}
    payload = dict(payload)
    payload["_written_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    payload["_python"] = platform.python_version()
    manifest[key] = payload
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def write_json(path: Path, obj: Any) -> None:
    """Write ``obj`` as pretty JSON, creating parent folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))


def open_text(path: Path, mode: str = "rt"):
    """Open plain or gzip text transparently."""
    return gzip.open(path, mode, encoding="utf-8") if str(path).endswith(".gz") else open(path, mode, encoding="utf-8")


def normalise_name(text: str) -> str:
    """Lower-case, strip punctuation/strain noise: used for antigen and taxon grouping."""
    t = re.sub(r"\([^)]*\)", " ", (text or "").lower())
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()
