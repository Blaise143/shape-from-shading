import torch
from enum import StrEnum
from synthesis import Sphere

class Initializers(StrEnum):
    SPHERE = "sphere"
    ZEROS = "zeros"
    ONES = "ones"


def initialize_normals(kind: Initializers = Initializers.SPHERE, image_size: int=1024, device="mps"):
    if kind not in Initializers:
        message = "kind should be one of ['sphere', 'zeros', 'ones']"
        raise Exception(message)

    if kind == Initializers.SPHERE:
        return _get_sphere_normals(image_size, device=device)
    elif kind == Initializers.ONES:
        return torch.ones(size=(image_size, image_size, 3), device=device)
    else:
        zeros = torch.zeros(size=(image_size, image_size, 3), device=device)
        zeros[..., 2] = 1.0
        return zeros


def _get_sphere_normals(img_size: int = 1024, device: str= "mps"):
    radius = (img_size // 2) - 10
    grid_range = torch.linspace(-radius, radius, img_size)
    y, x = torch.meshgrid(grid_range, grid_range, indexing='ij')

    sphere = Sphere(center=[0.0, 0.0, 0.0], radius=radius,
                    light_direction=torch.tensor([0.0, 0.0, 1.0]))

    radiance, mask, normals = sphere.run(x, y)
    normals = normals.to(device)
    if normals.shape[0]==3:
        return normals.permute(1,2,0)
    return normals


if __name__ == "__main__":
    spherical = initialize_normals()
    ones = initialize_normals(Initializers.ONES)
    zeros = initialize_normals(Initializers.ZEROS)
    print(spherical)
    print(ones)
    print(zeros)
    print(spherical.min(), spherical.max())
