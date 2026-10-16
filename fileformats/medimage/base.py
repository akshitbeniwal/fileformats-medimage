import logging
import os
import sys
import typing as ty
from pathlib import Path

from fileformats.core import (
    FileSet,
    Loaded,
    extra,
    mtime_cached_property,
    staged_out_dir,
)
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


class MedicalImagingData(FileSet):
    """Base class for all medical imaging data including pre-image raw data and
    associated data"""

    # Whether the files of the format may contain protected health information (PHI).
    # "nested" means that the format's own files don't, but formats nested within it
    # (i.e. returned by its validated properties) may, so deidentifying it deidentifies
    # them (see `deidentify_nested`). Being truthy, it is still treated as holding PHI
    contains_phi: bool | ty.Literal["nested"] = True

    @extra(wrapper=staged_out_dir)
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
        copies its remaining files unchanged, keeping the same layout (see
        `FileSet.map_nested`).

        Nested filesets of other types are deidentified too if they have a truthy
        `contains_phi` attribute, which they then need a `deidentify` method for
        (taking the same arguments), so formats from other packages that follow the
        same convention can be nested within each other.

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

        def deidentify(nested: FileSet, dest_dir: Path) -> FileSet | None:
            # Duck-typed, so nested formats from other families that follow the same
            # `contains_phi`/`deidentify` convention (e.g. medical imaging data and
            # biosignals) are also deidentified. One that may contain PHI but doesn't
            # have a `deidentify` method raises, rather than being copied
            if getattr(nested, "contains_phi", False):
                deidentified: FileSet = nested.deidentify(  # type: ignore[attr-defined]
                    dest_dir, recipe=recipe, **kwargs
                )
                return deidentified
            return None  # copied unchanged

        return self.map_nested(out_dir, deidentify)


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
