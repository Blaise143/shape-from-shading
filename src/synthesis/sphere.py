import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
# from utils import plot_normals, plot_component_normals


class Sphere:
    def __init__(
        self, center, radius, albedo=1, light_direction=torch.tensor([0.3, 0.3, 0.9])
    ):
        assert len(center) == 3, "Center must be a 3D coordinate"
        self.center = torch.tensor(center, dtype=torch.float32)
        self.radius = radius
        self.albedo = albedo
        self.I = F.normalize(light_direction.float(), dim=0)

    def generate_surface(self, x, y):
        """
        Generate the z-values and normals for the upper hemisphere.
        """
        x_centered = x - self.center[0]
        y_centered = y - self.center[1]
        r_sq = self.radius**2
        d_sq = x_centered**2 + y_centered**2
        z = torch.zeros_like(x)
        # r**2 = x**2 + y**2
        mask = d_sq <= r_sq
        z[mask] = torch.sqrt(torch.clamp(r_sq - d_sq[mask], min=0.0))

        points = torch.stack([x_centered, y_centered, z], dim=0)
        normals = points / self.radius
        return z, mask, normals

    def compute_radiance(self, normals, mask):
        """
        radiance: albedo * max(0, dot(I, N))
        """
        dot_product = (normals * self.I.view(3, 1, 1)).sum(dim=0)
        radiance = self.albedo * torch.clamp(dot_product, min=0.0)

        radiance[~mask] = 0.0

        # if torch.max(radiance) > 0:
        #    radiance = radiance / torch.max(radiance)

        return radiance

    def run(self, x, y):
        z, mask, normals = self.generate_surface(x, y)
        radiance = self.compute_radiance(normals, mask)
        return radiance, mask, normals


if __name__ == "__main__":
    step = 1.0
    img_size = 1024
    inch_size = 5
    dpi = img_size / inch_size

    radius = 0.8  # img_size // 2 - 10
    grid_range = torch.linspace(-1, 1, img_size)
    y, x = torch.meshgrid(grid_range, grid_range, indexing="ij")

    sphere = Sphere(
        center=[0.0, 0.0, 2.0], radius=radius, light_direction=torch.tensor([0, 0, 1.0])
    )

    radiance, mask, normals = sphere.run(x, y)
    # breakpoint()

    fig = plt.figure(figsize=(inch_size, inch_size))
    # plt.figure()
    img = radiance.numpy()
    img[~mask.numpy()] = 0  # np.nan
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    fig.savefig("../images/sphere_two.png", dpi=dpi)

    breakpoint()
    from PIL import Image

    Image.fromarray((img * 255).astype(np.uint8), mode="L").save(
        "../images/sphere.png", "PNG"
    )

    plt.show()
    breakpoint()

    # pyrefly: ignore [missing-module-attribute]
    from utils import load_img

    img_2 = load_img("../images/sphere.png")

    normals_masked = normals.permute(1, 2, 0).clone()
    normals_masked = (normals_masked + 1.0) / 2.0
    normals_masked[~mask] = np.nan
    fig = plt.figure(figsize=(5, 5))
    plt.imshow(normals_masked)
    plt.axis("off")
    # dpi = fig.get_dpi()
    fig.savefig("../images/normals.png", dpi=dpi)
    plt.show()
