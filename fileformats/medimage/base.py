import logging
import os
import shutil
import sys
import tempfile
import typing as ty
from pathlib import Path

from fileformats.core import FileSet, Loaded, extra, mtime_cached_property
from fileformats.core.mixin import WithClassifiers

from .contents import ContentsClassifier
from .contents.anatomical_entity import AnatomicalEntity
from .contents.imaging.derivatives import Derivative
from .contents.imaging.modality import ImagingModality

logger = logging.getLogger("fileformats")

if sys.version_info >= (3, 9):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias
if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

if ty.TYPE_CHECKING:
    import numpy.typing  # noqa: F401


# =====================================================================
# Custom loader functions for different image types
# =====================================================================

DataArrayType: TypeAlias = (
    ty.Any
)  # In Py<3.9 this is problematic "numpy.typing.NDArray[typing.Union[numpy.floating[typing.Any], numpy.integer[typing.Any]]]"


def _deidentify_wrapper(
    implementation: ty.Callable[..., ty.Any],
    fileset: FileSet,
    out_dir: os.PathLike[str],
    *args: ty.Any,
    **kwargs: ty.Any,
) -> FileSet:
    """Runs the deidentify implementation with a hidden staging directory alongside
    `out_dir` as its output directory, and only moves its output into `out_dir` once it
    has returned, so that only fully deidentified files end up there. Implementations
    are therefore free to copy files into their output directory and deidentify them
    in place."""
    out_dir = Path(out_dir).absolute()
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(dir=out_dir.parent, prefix=f".deidentify-{out_dir.name}-")
    )
    try:
        deidentified: FileSet = implementation(fileset, staging_dir, *args, **kwargs)
        try:
            rel_fspaths = [
                p.absolute().relative_to(staging_dir) for p in deidentified.fspaths
            ]
        except ValueError:
            raise RuntimeError(
                f"The deidentify implementation for {type(fileset)} filesets, "
                f"{implementation}, didn't write its output to the 'out_dir' it was "
                f"passed: {deidentified.fspaths}"
            ) from None
        entries = sorted(staging_dir.iterdir())
        if existing := [e.name for e in entries if (out_dir / e.name).exists()]:
            raise FileExistsError(
                f"Cannot deidentify {fileset} into {out_dir} as {existing} already "
                "exist(s) there"
            )
        out_dir.mkdir(exist_ok=True)
        for entry in entries:
            os.replace(entry, out_dir / entry.name)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
    return type(deidentified)([out_dir / p for p in rel_fspaths])


class MedicalImagingData(FileSet):
    """Base class for all medical imaging data including pre-image raw data and
    associated data"""

    # Whether the files of the format may contain protected health information (PHI).
    # "nested" means that the format's own files don't, but formats nested within it
    # (i.e. returned by its validated properties) may, so deidentifying it deidentifies
    # them (see `deidentify_nested`). Being truthy, it is still treated as holding PHI
    contains_phi: bool | ty.Literal["nested"] = True

    @extra(wrapper=_deidentify_wrapper)
    def deidentify(
        self,
        out_dir: os.PathLike[str],
        recipe: Loaded[FileSet] | None = None,
        **kwargs: ty.Any,
    ) -> Self:
        """
        Deidentifies the image by stripping any subject-identifying information from the
        image header. The exact implementation of this method will depend on the
        specific image format and the type of identifying information that is present. The
        output files should be named with a new file path(s) that is derived from the metadata,
        such that it doesn't contain any subject-identifying information within it.

        Implementations are passed a hidden staging directory as `out_dir`, the
        contents of which are only moved into the requested directory once they have
        returned, so they can copy files there and deidentify them in place without
        any risk of partially deidentified files being left in the output if they fail.

        Implementations only need to strip the identifying data -- they don't need to
        track/report what was changed. Callers that need that (e.g. for re-identification
        audit trails) can diff `metadata` before and after calling this method instead,
        since that works uniformly across formats without extra bookkeeping in each
        implementation.

        Parameters
        ----------
        out_dir: PathLike[str]
            The directory where the deidentified image should be saved
        recipe: Loaded[FileSet], optional
            A recipe for the deidentification process, loaded from a format that depends
            on the implementation (e.g. `DeidRecipe` for DICOM), which may include
            details on which fields to remove or how to handle certain types of data.
            Implementations annotate it with ``Loaded[<recipe format>]``, or a union of
            them in order of preference (e.g. ``Loaded[DeidRecipeX] | Loaded[DeidRecipe]``,
            for recipes with and without side-cars), or ``None`` if they don't accept a
            recipe. Callers can find the formats to load the recipe from with
            ``LoadedMarker.all_from_hint(get_type_hints(find_extra_implementation(
            MedicalImagingData.deidentify, type(image)), include_extras=True)["recipe"])``
            Formats nested within one another are passed the same recipe, so they need
            to be able to read a common recipe format.
        **kwargs: Any
            Additional format-specific keyword arguments (e.g. concurrency options),
            which implementations that don't use them should accept and ignore

        Returns
        -------
        Self
            A new instance of the image with any subject-identifying information stripped from
            the image header.

        Formats whose `contains_phi` is "nested" are deidentified by
        `deidentify_nested` if no implementation is registered for them.
        """
        if self.contains_phi == "nested":
            return self.deidentify_nested(out_dir, recipe=recipe, **kwargs)
        raise NotImplementedError

    def deidentify_nested(
        self,
        out_dir: os.PathLike[str],
        recipe: Loaded[FileSet] | None = None,
        **kwargs: ty.Any,
    ) -> Self:
        """Deidentifies the medical imaging data nested within the fileset (i.e.
        returned by its validated properties) with their own implementations, and
        copies its remaining files unchanged, keeping the same layout.

        For use within `deidentify` implementations of formats that contain others,
        which can then deidentify any of the format's own files that may contain PHI
        (e.g. logs) in place in the returned copy. As `deidentify` stages the output
        of its implementations, they are only moved to the requested output directory
        once that is done.

        Parameters
        ----------
        out_dir : PathLike[str]
            the directory to write the deidentified fileset to
        recipe : Loaded[FileSet], optional
            the recipe passed on to the deidentification of each nested fileset
        **kwargs : Any
            passed on to the deidentification of each nested fileset

        Returns
        -------
        Self
            the copy of the fileset with the nested filesets deidentified
        """
        out_dir = Path(out_dir)
        src_root = _root_dir(self)
        own_paths = set(self.all_file_paths())
        # Deidentify the nested formats first, each writing straight into its place in
        # the copy's layout. Formats without an implementation raise here, before
        # anything is copied
        deidentified_paths: set[Path] = set()
        for nested in _direct_nested_filesets(self):
            if not (isinstance(nested, MedicalImagingData) and nested.contains_phi):
                continue  # copied unchanged along with the rest
            nested_paths = set(nested.all_file_paths())
            if outside := sorted(nested_paths - own_paths):
                raise ValueError(
                    f"{nested} is nested within {self} but contains paths that aren't "
                    f"part of it, {outside}, so the properties of {type(self)} should "
                    "only return filesets made up of its own paths"
                )
            if nested_paths <= deidentified_paths:
                continue  # already deidentified as part of another nested fileset
            nested.deidentify(
                out_dir / _root_dir(nested).relative_to(src_root),
                recipe=recipe,
                **kwargs,
            )
            deidentified_paths.update(nested_paths)
        for path in own_paths - deidentified_paths:
            dest = out_dir / path.relative_to(src_root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
        # Reinstantiating the type checks that the nested deidentifications preserved
        # the layout it requires (e.g. didn't rename files it looks for)
        return type(self)([out_dir / p.relative_to(src_root) for p in self.fspaths])


def _root_dir(fileset: FileSet) -> Path:
    """The directory containing all of the fileset's paths (the parent of a directory
    fileset rather than the directory itself)"""
    return Path(os.path.commonpath([p.parent for p in fileset.fspaths]))


def _direct_nested_filesets(fileset: FileSet) -> list[FileSet]:
    """The filesets returned by the fileset's validated properties, including those
    within dicts and sequences, but not those nested further down (unlike
    `FileSet.nested_filesets()`), which are left to the filesets they are nested in"""
    nested: list[FileSet] = []

    def collect(value: ty.Any) -> None:
        if isinstance(value, FileSet):
            nested.append(value)
        elif isinstance(value, ty.Mapping):
            for v in value.values():
                collect(v)
        elif isinstance(value, (list, tuple, set, frozenset)):
            for v in value:
                collect(v)

    for name in sorted(fileset.validated_properties()):
        collect(getattr(fileset, name))
    return nested


class MedicalImage(WithClassifiers, MedicalImagingData):

    INCLUDE_HDR_KEYS: ty.Optional[ty.Tuple[str, ...]] = None
    IGNORE_HDR_KEYS: ty.Optional[ty.Tuple[str, ...]] = None
    binary = True
    classifiers_attr_name = "image_contents"
    image_contents = ()
    allowed_classifiers = (ContentsClassifier,)
    exclusive_classifiers = (ImagingModality, AnatomicalEntity, Derivative)

    @extra
    def read_array(self) -> DataArrayType:
        """
        Returns the binary data of the image in a numpy array
        """

    @mtime_cached_property
    def data_array(self) -> DataArrayType:
        return self.read_array()

    @extra
    def vox_sizes(self) -> ty.Tuple[float, float, float]:
        """The length of the voxels along each dimension"""
        raise NotImplementedError

    @extra
    def dims(self) -> ty.Tuple[int, int, int]:
        """The dimensions of the image"""
        raise NotImplementedError
