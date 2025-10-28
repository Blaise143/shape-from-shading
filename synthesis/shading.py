import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class Sphere:
    def __init__(self, center, radius, albedo=0.5, I=torch.tensor([0.3, 0.3, 0.9])):
        assert len(center) == 3, "Center has to be x, y and z"
        self.center = center
        self.radius = radius
        self.albedo = albedo
        self.I = I

    def get_partials(self, x, y):
        denom = (self.radius**2 - (x**2 + y**2)).sqrt()
        self.mask = denom > 0
        p = -x/denom
        q = -y/denom
        return p, q

    def get_radiance(self, x, y):
        p, q = self.get_partials(x, y)
        normal = torch.stack([-p, -q, torch.ones_like(p)])
        normal = normal / (normal[0]**2 + normal[1]**2 + normal[2]**2).sqrt()
        I = self.I.view(3, 1, 1)
        radiance = self.albedo * torch.clamp((I * normal).sum(0), min=0)
        radiance = radiance * self.mask
        if torch.max(radiance) > 0:
            radiance = radiance / torch.max(radiance)
        return radiance

    def run(self, x, y):
        radiance_ = self.get_radiance(x, y)
        return radiance_, self.mask


if __name__ == "__main__":
    step = 0.1
    img_size = 1024#256
    radius = img_size // 2 - 10
    grid_range = torch.arange(-1.5 * radius, 1.5 * radius + step, step)
    y, x = torch.meshgrid(grid_range, grid_range, indexing='ij')
    x = x.float()
    y = y.float()

    sphere = Sphere(center=[0, 0, 0], radius=radius)
    radiance, mask = sphere.run(x, y)

    plt.figure(figsize=(5, 5))
    img = radiance.numpy()
    img[~mask.numpy()] = np.nan
    print(f"image shape: {img.shape}")
    plt.imshow(img, cmap="gray")  # , cmap='inferno')
    plt.axis('off')
    # plt.title("Lambertian Shaded Sphere")
    plt.savefig("sphere.png")
    plt.show()
