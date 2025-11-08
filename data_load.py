import os
from glob import glob


class ShapeFromShadingDataset:
    def __init__(self, root_dir="GT_data4", cues=("Cue1", "Cue2", "Cue3")):
        self.root_dir = root_dir
        self.cues = cues
        self.masks_dir = os.path.join(root_dir, "Masks")
        self.data = self._load_all()

    def _load_all(self):
        data = {}
        for cue in self.cues:
            cue_path = os.path.join(self.root_dir, cue)
            data[cue] = {}
            for img_path in glob(os.path.join(cue_path, "*.png")):
                obj_name = os.path.splitext(os.path.basename(img_path))[0]
                mask_path = os.path.join(self.masks_dir, f"{obj_name}.png")
                data[cue][obj_name] = {
                    "image": img_path,
                    "mask": mask_path if os.path.exists(mask_path) else None,
                }
        return data

    def get(self, cue="Cue1", obj="sphere"):
        """Return dict with image + mask paths for a given cue/object"""
        if cue not in self.data or obj not in self.data[cue]:
            raise ValueError(f"Invalid cue/object: {cue}/{obj}")
        return self.data[cue][obj]


if __name__ == "__main__":
    import pprint as pp

    dataset = ShapeFromShadingDataset()
    pp.pprint(dataset._load_all())

    pyramid = dataset.get("Cue1", obj="pyramid")
    pp.pprint(pyramid)
