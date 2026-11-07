from dataclasses import dataclass, asdict
from typing import Optional, Literal
import torch
import json


@dataclass
class SFSConfig:
    """Configuration for a single Shape-from-Shading optimization run.

    One :class:`SFSConfig` is built per object in :func:`src.run.run` and
    passed straight into :func:`src.sfs.shape_from_shading`.
    """

    image_path: str
    """Path to the grayscale input image."""

    num_iters: int = 2000
    """Number of optimizer steps to run."""

    lambda_smooth: float = 1.0
    """Weight of the normal-smoothness regularizer."""

    lr: float = 0.01
    """Learning rate for the Adam optimizer."""

    optimize_light: bool = True
    """Whether the light direction is a learnable parameter."""

    light_direction: Optional[torch.Tensor] = None
    """Initial light direction as a 3-vector; if ``None``,
    :func:`src.core_utils.get_light_direction` supplies a default."""

    save_to: Optional[str] = None
    """Output directory for this run's artifacts (plots, tensors, params.json)."""

    optimize_albedo: bool = False
    """Whether albedo is jointly optimized."""

    use_mask: bool = False
    """Whether the data loss is restricted to the masked region."""

    init_kind: Literal["init_grads", "zeroes", "ones", "sphere"] = "init_grads"
    """Strategy used to initialize the normal map; see
    :func:`utils.initializations.get_init`."""

    desc: Optional[str] = None
    """Optional free-text description of the run."""

    parametrization_type: Literal["cartesian", "stereographic"] = "stereographic"
    """Whether normals are optimized directly (``"cartesian"``) or via a
    stereographic projection (``"stereographic"``)."""

    def __post_init__(self):
        """Validate field values after dataclass construction.

        Raises:
            ValueError: If ``parametrization_type`` is not one of
                ``"cartesian"`` or ``"stereographic"``.
        """
        valid_parametrizations = ["cartesian", "stereographic"]
        if self.parametrization_type not in valid_parametrizations:
            raise ValueError("Parametrization should be cartesian or stereographic")

    def to_dict(self) -> dict:
        """Convert the config to a JSON-serializable dict.

        Drops ``image_path`` and ``save_to`` (filesystem-specific, not
        hyperparameters) and converts any tensor-valued fields (e.g.
        ``light_direction``) to plain lists.

        Returns:
            dict: The remaining fields, JSON-serializable.
        """
        d = asdict(self)
        exclude = ["image_path", "save_to"]
        d = {k: v for k, v in d.items() if k not in exclude}
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                d[k] = v.tolist()
        return d

    def save_params(self, path: str):
        """Save this config's parameters (via :meth:`to_dict`) to a JSON file.

        Args:
            path: Destination file path for the JSON output.
        """
        params = self.to_dict()
        with open(path, "w") as f:
            json.dump(params, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    cfg = SFSConfig("some_path")
    print(cfg.to_dict())
