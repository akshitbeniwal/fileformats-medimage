import os
import typing as ty
from pathlib import Path

import pytest
from fileformats.core import extra_implementation, validated_property
from fileformats.core.exceptions import FormatMismatchError
from fileformats.generic import Directory, File

from fileformats.medimage import MedicalImagingData


class Secret(File, MedicalImagingData):
    ext = ".secret"


class Opaque(File, MedicalImagingData):
    """Contains PHI but has no deidentify implementation"""

    ext = ".opaque"


class Leaf(Directory, MedicalImagingData):
    contains_phi = "nested"

    @validated_property
    def secrets(self) -> list[Secret]:
        found = [Secret(p) for p in sorted(self.fspath.glob("*.secret"))]
        if not found:
            raise FormatMismatchError(f"No secrets in {self.fspath}")
        return found


class Branch(Directory, MedicalImagingData):
    contains_phi = "nested"

    @validated_property
    def leaves(self) -> dict[str, Leaf]:
        return {p.name: Leaf(p) for p in self.fspath.glob("leaf*")}

    @validated_property
    def opaques(self) -> list[Opaque]:
        return [Opaque(p) for p in self.fspath.glob("*.opaque")]


class Trunk(Directory, MedicalImagingData):
    """A container with a log file of its own that may contain PHI"""

    @validated_property
    def branch(self) -> Branch:
        return Branch(self.fspath / "branch")

    @property
    def log_file(self) -> File:
        return File(self.fspath / "log.txt")


@extra_implementation(MedicalImagingData.deidentify)
def trunk_deidentify(
    trunk: Trunk,
    out_dir: os.PathLike[str],
    recipe: None = None,
    fail: bool = False,
    **kwargs: ty.Any,
) -> Trunk:
    deidentified = trunk.deidentify_nested(out_dir, recipe=recipe, **kwargs)
    if fail:
        raise RuntimeError("deidentification failed")
    deidentified.log_file.fspath.write_text("redacted")
    return deidentified


@extra_implementation(MedicalImagingData.deidentify)
def secret_deidentify(
    secret: Secret,
    out_dir: os.PathLike[str],
    recipe: None = None,
    **kwargs: ty.Any,
) -> Secret:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / secret.fspath.name
    out_path.write_text("redacted")
    return Secret(out_path)


def _make_branch(root: Path) -> Branch:
    branch = root / "branch"
    for leaf in ("leaf1", "leaf2"):
        (branch / leaf).mkdir(parents=True)
        (branch / leaf / "a.secret").write_text("Jane Doe")
        (branch / leaf / "notes.txt").write_text("plain")
    (branch / ".hidden").write_text("hidden")
    return Branch(branch)


def test_deidentify_nested(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    deidentified = _make_branch(tmp_path / "in").deidentify(out_dir)

    assert isinstance(deidentified, Branch)
    assert deidentified.fspath == out_dir / "branch"
    assert {
        str(p.relative_to(out_dir)): p.read_text()
        for p in deidentified.all_file_paths()
    } == {
        "branch/.hidden": "hidden",
        "branch/leaf1/a.secret": "redacted",
        "branch/leaf1/notes.txt": "plain",
        "branch/leaf2/a.secret": "redacted",
        "branch/leaf2/notes.txt": "plain",
    }
    # the staging directories are cleaned up
    assert sorted(p.name for p in out_dir.iterdir()) == ["branch"]


def _make_trunk(root: Path) -> Trunk:
    _make_branch(root / "trunk")
    (root / "trunk" / "log.txt").write_text("Jane Doe")
    return Trunk(root / "trunk")


def test_deidentify_own_files_in_place(tmp_path: Path) -> None:
    trunk = _make_trunk(tmp_path / "in")
    deidentified = trunk.deidentify(tmp_path / "out")

    assert deidentified.fspath == tmp_path / "out" / "trunk"
    assert deidentified.log_file.fspath.read_text() == "redacted"
    assert (deidentified.fspath / "branch" / "leaf1" / "a.secret").read_text() == (
        "redacted"
    )
    # the originals are left untouched
    assert trunk.log_file.fspath.read_text() == "Jane Doe"


def test_deidentify_failure_leaves_nothing(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    with pytest.raises(RuntimeError, match="deidentification failed"):
        _make_trunk(tmp_path / "in").deidentify(out_dir, fail=True)
    assert not out_dir.exists()
    # the staging directories are cleaned up
    assert sorted(p.name for p in tmp_path.iterdir()) == ["in"]


def test_deidentify_without_implementation_leaves_nothing(tmp_path: Path) -> None:
    branch = _make_branch(tmp_path / "in")
    (branch.fspath / "x.opaque").write_text("Jane Doe")
    out_dir = tmp_path / "out"
    with pytest.raises(NotImplementedError):
        Branch(branch.fspath).deidentify(out_dir)
    assert not out_dir.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["in"]


def test_deidentify_nested_wont_overwrite(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    (out_dir / "branch").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        _make_branch(tmp_path / "in").deidentify(out_dir)


def test_contains_phi_nested_is_truthy() -> None:
    assert Branch.contains_phi
