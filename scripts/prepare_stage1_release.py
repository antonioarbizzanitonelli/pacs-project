"""Export the explicit through-AGH publication without changing the workspace.

The allowlist is deliberately independent of current Git tracking. Four versioned
files use their already-published Stage-1 blobs, with explicit fixture-export and
optional-solver test projections. All other files use live inputs. Known text
files receive explicit LF transport normalization before hashing and writing.
The exported allowlist reads the delivered files directly, so preparation also
works in a fresh clone of the publication repository.
No Git index, branch, commit, tracked file or existing output is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "configs" / "stage1_publication.json"
GENERATED_MANIFEST = "stage1-manifest.json"


def publication_path(value: str) -> PurePosixPath:
    """Accept one normalized repository-relative file path."""

    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError(f"invalid publication path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(
        part in ("", ".", "..") or part.endswith((" ", "."))
        for part in value.split("/")
    ):
        raise ValueError(f"invalid publication path: {value!r}")
    if path.parts[0].casefold() == ".git" or value.casefold() == GENERATED_MANIFEST:
        raise ValueError(f"reserved publication path: {value!r}")
    return path


def git_blob(revision: str, path: str) -> bytes:
    """Read an immutable published blob using a full commit identifier."""

    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("published overrides require a full lowercase commit SHA")
    result = subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise ValueError(f"cannot read published blob {revision}:{path}")
    return result.stdout


def project_coherent_fixture_export(payload: bytes) -> bytes:
    """Extend the published initializer with the coherent tiny fixture."""

    text = payload.decode("utf-8")
    markers = (
        ("from .tiny_instances import (", "from .tiny_instances import (\n    make_coherent_tiny_instance,"),
        ("__all__ = [", '__all__ = [\n    "make_coherent_tiny_instance",'),
    )
    for marker, replacement in markers:
        if text.count(marker) != 1:
            raise ValueError("published data initializer differs from the approved projection")
        text = text.replace(marker, replacement, 1)
    return text.encode("utf-8")


def project_optional_gurobi_coverage(payload: bytes) -> bytes:
    """Run generic tolerance coverage with HiGHS and an available optional Gurobi."""

    text = payload.decode("utf-8")
    markers = (
        (
            "from llm_resource_allocation.optimization.solver import SolverBackend, SolverConfig",
            "from llm_resource_allocation.optimization.solver import (\n"
            "    SolverBackend, SolverBackendAvailability, SolverConfig, probe_solver_backend,\n)",
        ),
        (
            "        for backend in (SolverBackend.HIGHS, SolverBackend.GUROBI):\n"
            "            with self.subTest(backend=backend):",
            "        for backend in (SolverBackend.HIGHS, SolverBackend.GUROBI):\n"
            "            if (\n"
            "                backend is SolverBackend.GUROBI\n"
            "                and probe_solver_backend(backend).availability is not SolverBackendAvailability.AVAILABLE\n"
            "            ):\n"
            "                continue\n"
            "            with self.subTest(backend=backend):",
        ),
    )
    for marker, replacement in markers:
        if text.count(marker) != 1:
            raise ValueError("published tolerance test differs from the approved projection")
        text = text.replace(marker, replacement, 1)
    return text.encode("utf-8")


def prepare_files(manifest_path: Path) -> tuple[dict[str, bytes], dict]:
    """Read and hash every allowed input before creating an output directory."""

    manifest_bytes = manifest_path.read_bytes()
    specification = json.loads(manifest_bytes)
    if not isinstance(specification, dict) or specification.get("format_version") != 1:
        raise ValueError("unsupported publication manifest format")
    paths = specification.get("files")
    if not isinstance(paths, list) or not paths:
        raise ValueError("publication manifest must contain an explicit file list")
    normalized = [str(publication_path(path)) for path in paths]
    if len({path.casefold() for path in normalized}) != len(normalized):
        raise ValueError("publication manifest contains duplicate paths")
    overrides = specification.get("published_blob_overrides", {})
    if not isinstance(overrides, dict) or set(overrides).difference(normalized):
        raise ValueError("published overrides must refer to allowlisted files")

    payloads = {}
    evidence = []
    for path in normalized:
        override = overrides.get(path)
        if override is None:
            source = (REPOSITORY_ROOT / path).resolve()
            if not source.is_relative_to(REPOSITORY_ROOT) or not source.is_file():
                raise ValueError(f"missing or external publication source: {path}")
            payload = source.read_bytes()
            provenance = {"source": "workspace"}
        else:
            if not isinstance(override, dict) or not override.get("reason"):
                raise ValueError(f"published override requires a reason: {path}")
            revision = override.get("revision")
            payload = git_blob(revision, path)
            provenance = {
                "source": "published_git_blob",
                "revision": revision,
                "reason": override["reason"],
            }
            projection = override.get("projection")
            if projection is not None:
                provenance["published_blob_sha256"] = hashlib.sha256(payload).hexdigest()
                if (
                    projection == "coherent_tiny_fixture_export"
                    and path == "src/llm_resource_allocation/data/__init__.py"
                ):
                    payload = project_coherent_fixture_export(payload)
                elif (
                    projection == "optional_gurobi_tolerance_coverage"
                    and path == "tests/test_validation_tolerances.py"
                ):
                    payload = project_optional_gurobi_coverage(payload)
                else:
                    raise ValueError(f"unsupported publication projection: {path}")
                provenance["projection"] = projection
        if path == "configs/stage1_publication.json" and overrides:
            standalone_specification = dict(specification)
            standalone_specification.pop("published_blob_overrides", None)
            provenance["input_sha256"] = hashlib.sha256(payload).hexdigest()
            provenance["projection"] = "standalone_publication_config"
            payload = (
                json.dumps(standalone_specification, indent=2, ensure_ascii=False) + "\n"
            ).encode("utf-8")
        if (
            Path(path).suffix in (".py", ".md", ".toml", ".json", ".yml", ".yaml")
            or path in (".gitignore", ".gitattributes")
        ):
            normalized_payload = payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            provenance["text_normalization"] = "lf"
            if normalized_payload != payload:
                provenance["input_sha256"] = hashlib.sha256(payload).hexdigest()
                payload = normalized_payload
        payloads[path] = payload
        evidence.append({
            "path": path,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            **provenance,
        })

    record = {
        "format_version": 1,
        "scope": specification["scope"],
        "allowlist_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "file_count": len(evidence),
        "files": evidence,
    }
    return payloads, record


def output_directory(value: str) -> Path:
    """Resolve the destination and protect the repository's source directories."""

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    path = path.resolve()
    protected = (".git", "src", "tests", "docs", "scripts", "configs", "examples", ".github")
    if path == REPOSITORY_ROOT or any(
        path.is_relative_to(REPOSITORY_ROOT / name) for name in protected
    ):
        raise ValueError("output must be separate from repository source directories")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="build/stage1", help="new destination directory")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--check", action="store_true", help="validate inputs without writing")
    args = parser.parse_args()
    try:
        destination = output_directory(args.output)
        if not args.check and destination.exists():
            raise ValueError(f"output already exists; choose a new directory: {destination}")
        payloads, record = prepare_files(args.manifest)
        manifest_bytes = (json.dumps(record, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        if not args.check:
            destination.mkdir(parents=True, exist_ok=False)
            for path, payload in payloads.items():
                target = destination / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
            (destination / GENERATED_MANIFEST).write_bytes(manifest_bytes)
    except (OSError, ValueError, KeyError) as error:
        print(f"Stage-1 export failed: {error}", file=sys.stderr)
        return 1

    print(f"{'Checked' if args.check else 'Prepared'} {len(payloads)} Stage-1 files")
    print(f"Tree: {destination}")
    print(f"Manifest SHA256: {manifest_sha256}")
    print("Git index and existing workspace files were preserved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
