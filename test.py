from synthesis.sphere import Sphere
import torch
import matplotlib.pyplot as plt


if __name__ == "__main__":
    step = 1.0
    img_size = 1024
    inch_size = 5
    dpi = img_size / inch_size

    radius = 0.8  # img_size // 2 - 10
    # grid_range = torch.linspace(-radius, radius, img_size)
    grid_range = torch.linspace(-1, 1, img_size)
    y, x = torch.meshgrid(grid_range, grid_range, indexing="ij")

    sphere = Sphere(
        center=[0.0, 0.0, 2.0], radius=radius, light_direction=torch.tensor([0, 0, 1.0])
    )

    depth, mask, normals = sphere.generate_surface(x, y)
    radiance, _, _ = sphere.run(x, y)
    plt.imshow(depth)
    plt.show()
    plt.imshow(radiance)
    plt.show()
    from utils.plot_helpers import plot_depth_3d_plotly
    from utils.helpers import integrate_normals

    z = depth
    plot_depth_3d_plotly(
        depth=z,
        save_path=None,
        show=True,
        normalize_xyz=False,
        mask=mask,
        hide_invalid=False,
    )
