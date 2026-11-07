import os
from glob import glob
from typing import Optional, Dict, List
from pathlib import Path


class ShapeFromShadingDataset:
    """Indexes a directory of images against a directory of matching masks.

    Images and masks are paired by basename (without extension): for an
    object named ``"sphere"``, it expects ``{images_dir}/sphere.{image_ext}``
    and ``{masks_dir}/sphere.{mask_ext}``.
    """

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        image_ext: str = "png",
        mask_ext: Optional[str] = None,
    ):
        """Build the dataset by indexing `images_dir` against `masks_dir`.

        Args:
            images_dir: Directory containing the input images.
            masks_dir: Directory containing the corresponding masks.
            image_ext: File extension (without dot) of the images.
            mask_ext: File extension (without dot) of the masks; defaults to
                `image_ext` when not given.

        Raises:
            FileNotFoundError: If `images_dir` or `masks_dir` doesn't exist,
                or if an image has no matching mask.
            RuntimeError: If no image/mask pairs are found at all.
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_ext = image_ext
        self.mask_ext = mask_ext or image_ext  # default: same as image_ext

        if not os.path.isdir(self.images_dir):
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not os.path.isdir(self.masks_dir):
            raise FileNotFoundError(f"Masks directory not found: {self.masks_dir}")

        self.samples: Dict[str, Dict[str, str]] = self._index()

    def _index(self) -> Dict[str, Dict[str, str]]:
        """Scan `images_dir` and build the name -> {image, mask} path mapping.

        Returns:
            Dict[str, Dict[str, str]]: Maps each object basename to a dict
            with ``"image"`` and ``"mask"`` file paths.

        Raises:
            FileNotFoundError: If an image has no matching mask file.
            RuntimeError: If no image/mask pairs are found.
        """
        samples: Dict[str, Dict[str, str]] = {}

        pattern = os.path.join(self.images_dir, f"*.{self.image_ext}")
        for img_path in glob(pattern):
            name = Path(img_path).stem
            mask_path = os.path.join(self.masks_dir, f"{name}.{self.mask_ext}")

            if not os.path.exists(mask_path):
                raise FileNotFoundError(
                    f"Mask not found for '{name}': expected {mask_path}"
                )

            samples[name] = {
                "image": img_path,
                "mask": mask_path,
            }

        if not samples:
            raise RuntimeError(
                f"No image/mask pairs found in {self.images_dir} / {self.masks_dir}"
            )

        return samples

    def list_objects(self) -> List[str]:
        """List all indexed object names.

        Returns:
            List[str]: Object basenames (without extension), sorted
            alphabetically.
        """
        return sorted(self.samples.keys())

    def get(self, name: str) -> Dict[str, str]:
        """
        Return dict with image + mask paths for a given object name.

        Args:
            name: Object basename, as returned by :meth:`list_objects`.

        Returns:
            Dict[str, str]: A dict with ``"image"`` and ``"mask"`` file paths.

        Raises:
            ValueError: If `name` isn't a known object.

        Example::

            dataset.get("sphere") ->
            {
                "image": ".../Images/sphere.png",
                "mask":  ".../Masks/sphere.png",
            }
        """
        if name not in self.samples:
            raise ValueError(f"Unknown object '{name}'. Known: {self.list_objects()}")
        return self.samples[name]


if __name__ == "__main__":
    ...
