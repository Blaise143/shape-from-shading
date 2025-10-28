from dataclasses import dataclass, asdict
from typing import Optional
import torch
import json


@dataclass
class SFSConfig:
    image_path: str
    num_iters: int = 1000
    lambda_smooth: float = 0.1
    lr: float = 0.1
    optimize_light: bool = False
    light_direction: Optional[torch.Tensor] = None
    save_to: Optional[str] = None
    optimize_albedo: bool = False
    use_mask: bool = False
    init_kind: str = "init_grads"

    def to_dict(self):
        d = asdict(self)
        exclude = ["image_path", "save_to"]
        d = {k: v for k, v in d.items() if k not in exclude}
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                d[k] = v.tolist()
        return d

    def save_params(self, path: str):
        """
        Saves the parameters to json file
        """
        params = self.to_dict()
        with open(path, "w") as f:
            json.dump(params, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    cfg = SFSConfig("some_path")
    print(cfg.to_dict())
