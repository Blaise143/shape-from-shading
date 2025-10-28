import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

class Plane:
    def __init__(self, point, normal, albedo=0.5, light_direction=torch.tensor([0., 0, 1])):
        """
        point: a point on the plane (3D)
        normal: normal vector of the plane (3D)
        """
        assert len(point) == 3, "Point must be a 3D coordinate"
        assert len(normal) == 3, "Normal must be 3D"
        self.point = torch.tensor(point, dtype=torch.float32)
        self.normal = F.normalize(torch.tensor(normal, dtype=torch.float32), dim=0)
        self.albedo = albedo
        self.I = F.normalize(light_direction.float(), dim=0)

    def generate_surface(self, x, y):
        """
        Generate the z-values and normals for the plane.
        """
        a, b, c = self.normal
        d = -torch.dot(self.normal, self.point)

        z = torch.full_like(x, float("nan"))

        if c != 0:
            z = -(a * (x - self.point[0]) + b * (y - self.point[1]) + d) / c
        else:
            raise ValueError("Plane normal has no z component, cannot represent as z=f(x,y).")

        mask = torch.ones_like(x, dtype=torch.bool)
        normals = self.normal.view(3, 1, 1).expand(3, *x.shape)

        return z, mask, normals

    def compute_radiance(self, normals, mask):
        dot_product = (normals * self.I.view(3, 1, 1)).sum(dim=0)
        radiance = self.albedo * torch.clamp(dot_product, min=0.0)
        radiance[~mask] = 0.0

        # if torch.max(radiance) > 0:
        #     radiance = radiance / torch.max(radiance)
            # pass
        return radiance

    def run(self, x, y):
        z, mask, normals = self.generate_surface(x, y)
        radiance = self.compute_radiance(normals, mask)
        return radiance, mask, normals

def quick_run(res=1024, normal = [1., 0, 1],
              light_direction= torch.tensor([0.5, 0.5, 1])):
    # res = 1024
    x = torch.linspace(-1, 1, res)
    y = torch.linspace(-1, 1, res)
    X, Y = torch.meshgrid(x, y, indexing="ij")

    plane = Plane(point=[0, 0, 0], normal=normal,
                  albedo=0.7, light_direction=light_direction)

    radiance, _, normals = plane.run(X, Y)
    return radiance, normals


if __name__ == "__main__":
    # res = 1024
    # x = torch.linspace(-1, 1, res)
    # y = torch.linspace(-1, 1, res)
    # X, Y = torch.meshgrid(x, y, indexing="ij")
    #
    # plane = Plane(point=[0, 0, 0], normal=[1, 0, 1.],
    #               albedo=0.7, light_direction=torch.tensor([0.5, 0.5, 0]))
    #
    # radiance, _, normals = plane.run(X, Y)
    radiance, normals = quick_run(normal=[1,1,1], )
    # breakpoint()
    # torch.save(radiance, "../plane_data/plane_radiance.pt")
    # torch.save(mask, "plane_data/plane_mask.pt")
    # torch.save(normals, "../plane_data/plane_normals.pt")
    # breakpoint()

    plt.figure(figsize=(5, 5))
    plt.imshow(radiance.numpy(), cmap="gray", vmin=0. , vmax=1.)# extent=(-1, 1, -1, 1), origin="lower", cmap="grayi")
    plt.title("Synthetic Plane Radiance")
    plt.colorbar()
    plt.show()
    # plt.imshow()
