import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F


class Cube:
    def __init__(
        self,
        center,
        side_length,
        albedo=1,
        light_direction=torch.tensor([1.0, 1.0, 1.0]),
        view_direction=torch.tensor([1.0, 1.0, 1.0]),
    ):
        assert len(center) == 3, "Center must be a 3D coordinate"
        self.center = torch.tensor(center, dtype=torch.float32)
        self.side_length = side_length
        self.albedo = albedo
        self.I = F.normalize(light_direction.float(), dim=0)
        self.V = F.normalize(view_direction.float(), dim=0)

        self.half_side = side_length / 2

        # Define cube vertices
        self.vertices = (
            torch.tensor(
                [
                    [-1, -1, -1],  # 0: back-bottom-left
                    [1, -1, -1],  # 1: back-bottom-right
                    [1, 1, -1],  # 2: back-top-right
                    [-1, 1, -1],  # 3: back-top-left
                    [-1, -1, 1],  # 4: front-bottom-left
                    [1, -1, 1],  # 5: front-bottom-right
                    [1, 1, 1],  # 6: front-top-right
                    [-1, 1, 1],  # 7: front-top-left
                ],
                dtype=torch.float32,
            )
            * self.half_side
            + self.center
        )

        # Define faces with their normals
        self.faces = [
            {
                "vertices": [0, 1, 2, 3],
                "normal": torch.tensor([0.0, 0.0, -1.0]),
            },  # back
            {
                "vertices": [4, 5, 6, 7],
                "normal": torch.tensor([0.0, 0.0, 1.0]),
            },  # front
            {
                "vertices": [0, 4, 7, 3],
                "normal": torch.tensor([-1.0, 0.0, 0.0]),
            },  # left
            {
                "vertices": [1, 5, 6, 2],
                "normal": torch.tensor([1.0, 0.0, 0.0]),
            },  # right
            {"vertices": [3, 2, 6, 7], "normal": torch.tensor([0.0, 1.0, 0.0])},  # top
            {
                "vertices": [0, 1, 5, 4],
                "normal": torch.tensor([0.0, -1.0, 0.0]),
            },  # bottom
        ]

    def project_to_2d(self, vertices):
        """Project 3D vertices to 2D using the view direction"""
        # Simple orthographic projection along view direction
        # For corner view [1,1,1], we'll use a projection that shows three faces
        projection_matrix = torch.tensor(
            [
                [1, 0, 0],  # Keep x
                [0, 1, 0],  # Keep y (drop z for simple orthographic)
            ],
            dtype=torch.float32,
        )

        # Apply projection
        projected = torch.matmul(vertices, projection_matrix.T)
        return projected

    def is_face_visible(self, face_normal, face_center):
        """Check if a face is visible given view direction"""
        # Face is visible if its normal points away from view direction
        return torch.dot(face_normal, self.V) < 0

    def generate_surface(self, x, y):
        """Generate the visible surface of the cube"""
        batch_shape = x.shape
        z = torch.zeros_like(x)
        mask = torch.zeros_like(x, dtype=torch.bool)
        normals = torch.zeros((3, *batch_shape), dtype=torch.float32)

        # Project all vertices to 2D
        projected_vertices = self.project_to_2d(self.vertices)

        # For each face, check visibility and rasterize
        for face in self.faces:
            face_verts_3d = self.vertices[face["vertices"]]
            face_verts_2d = projected_vertices[face["vertices"]]
            face_normal = face["normal"]
            face_center = face_verts_3d.mean(dim=0)

            if not self.is_face_visible(face_normal, face_center):
                continue

            # Get bounding box of face in 2D
            min_x, max_x = face_verts_2d[:, 0].min(), face_verts_2d[:, 0].max()
            min_y, max_y = face_verts_2d[:, 1].min(), face_verts_2d[:, 1].max()

            # Create mask for pixels within bounding box
            bbox_mask = (x >= min_x) & (x <= max_x) & (y >= min_y) & (y <= max_y)

            if not bbox_mask.any():
                continue

            # For pixels in bbox, check if they're inside the face polygon
            for i in range(batch_shape[0]):
                for j in range(batch_shape[1]):
                    if bbox_mask[i, j]:
                        if self.point_in_quadrilateral(x[i, j], y[i, j], face_verts_2d):
                            # Use the face's plane equation to find z
                            z_val = self.solve_plane_z(
                                x[i, j], y[i, j], face_verts_3d[0], face_normal
                            )
                            z[i, j] = z_val
                            normals[:, i, j] = face_normal
                            mask[i, j] = True

        return z, mask, normals

    def point_in_quadrilateral(self, x, y, verts):
        """Check if point (x,y) is inside quadrilateral using cross product method"""
        verts = verts.numpy()
        x, y = x.item(), y.item()

        def cross_product(p1, p2, p3):
            return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

        d1 = cross_product(verts[0], verts[1], [x, y])
        d2 = cross_product(verts[1], verts[2], [x, y])
        d3 = cross_product(verts[2], verts[3], [x, y])
        d4 = cross_product(verts[3], verts[0], [x, y])

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0) or (d4 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0) or (d4 > 0)

        return not (has_neg and has_pos)

    def solve_plane_z(self, x, y, point_on_plane, normal):
        """Solve for z given x,y on a plane defined by a point and normal"""
        # Plane equation: normal·(x,y,z) = normal·point_on_plane
        # normal_x*x + normal_y*y + normal_z*z = normal·point_on_plane
        # z = (normal·point_on_plane - normal_x*x - normal_y*y) / normal_z

        dot_product = torch.dot(normal, point_on_plane)
        numerator = dot_product - normal[0] * x - normal[1] * y

        if abs(normal[2]) > 1e-6:  # Avoid division by zero
            return numerator / normal[2]
        else:
            return point_on_plane[2]  # Vertical face, use original z

    def compute_radiance(self, normals, mask):
        """radiance: albedo * max(0, dot(I, N))"""
        dot_product = (normals * self.I.view(3, 1, 1)).sum(dim=0)
        radiance = self.albedo * torch.clamp(dot_product, min=0.0)
        radiance[~mask] = 0.0
        return radiance

    def run(self, x, y):
        z, mask, normals = self.generate_surface(x, y)
        radiance = self.compute_radiance(normals, mask)
        return radiance, mask, normals


# Simpler approach that actually shows a proper cube
class SimpleCube:
    def __init__(
        self,
        center,
        side_length,
        albedo=1,
        light_direction=torch.tensor([1.0, 1.0, 1.0]),
    ):
        self.center = torch.tensor(center, dtype=torch.float32)
        self.side_length = side_length
        self.albedo = albedo
        self.I = F.normalize(light_direction.float(), dim=0)
        self.half_side = side_length / 2
        half_side = self.half_side
        # Define faces with their plane equations and normals
        self.faces = [
            # Each face: [normal, plane_constant, bounds_function]
            {
                "normal": torch.tensor([0.0, 0.0, 1.0]),
                "plane": center[2] + half_side,  # front
                "bounds": lambda x, y: (x, y),
            },
            {
                "normal": torch.tensor([0.0, 0.0, -1.0]),
                "plane": center[2] - half_side,  # back
                "bounds": lambda x, y: (x, y),
            },
            {
                "normal": torch.tensor([1.0, 0.0, 0.0]),
                "plane": center[0] + half_side,  # right
                "bounds": lambda y, z: (y, z),
            },
            {
                "normal": torch.tensor([-1.0, 0.0, 0.0]),
                "plane": center[0] - half_side,  # left
                "bounds": lambda y, z: (y, z),
            },
            {
                "normal": torch.tensor([0.0, 1.0, 0.0]),
                "plane": center[1] + half_side,  # top
                "bounds": lambda x, z: (x, z),
            },
            {
                "normal": torch.tensor([0.0, -1.0, 0.0]),
                "plane": center[1] - half_side,  # bottom
                "bounds": lambda x, z: (x, z),
            },
        ]

    def generate_surface(self, x, y):
        """Generate a proper 3D cube surface with multiple visible faces"""
        batch_shape = x.shape
        z = torch.zeros_like(x)
        mask = torch.zeros_like(x, dtype=torch.bool)
        normals = torch.zeros((3, *batch_shape), dtype=torch.float32)

        # Create a proper 3D cube by combining multiple faces
        # We'll create faces in different orientations

        center_x, center_y, center_z = self.center
        half = self.half_side

        # Front face (facing camera)
        front_mask = (
            (x >= center_x - half)
            & (x <= center_x + half)
            & (y >= center_y - half)
            & (y <= center_y + half)
        )
        z[front_mask] = center_z + half
        normals[2, front_mask] = 1.0  # facing forward
        mask = mask | front_mask

        # Right face (visible on the right side)
        right_mask = (
            (y >= center_y - half)
            & (y <= center_y + half)
            & (x >= center_x + half - 0.3)
            & (x <= center_x + half + 0.3)
        )
        z[right_mask] = torch.where(
            x[right_mask] < center_x + half,
            center_z + half - (center_x + half - x[right_mask]),
            center_z - half + (x[right_mask] - (center_x + half)),
        )
        normals[0, right_mask] = 1.0  # facing right
        mask = mask | right_mask

        # Top face (visible on the top)
        top_mask = (
            (x >= center_x - half)
            & (x <= center_x + half)
            & (y >= center_y + half - 0.3)
            & (y <= center_y + half + 0.3)
        )
        z[top_mask] = torch.where(
            y[top_mask] < center_y + half,
            center_z + half - (center_y + half - y[top_mask]),
            center_z - half + (y[top_mask] - (center_y + half)),
        )
        normals[1, top_mask] = 1.0  # facing up
        mask = mask | top_mask

        return z, mask, normals

    def compute_radiance(self, normals, mask):
        dot_product = (normals * self.I.view(3, 1, 1)).sum(dim=0)
        radiance = self.albedo * torch.clamp(dot_product, min=0.0)
        radiance[~mask] = 0.0
        return radiance

    def run(self, x, y):
        z, mask, normals = self.generate_surface(x, y)
        radiance = self.compute_radiance(normals, mask)
        return radiance, mask, normals


# Create and visualize a proper cube
if __name__ == "__main__":
    # Create coordinate grid
    x = torch.linspace(-1.5, 1.5, 256)
    y = torch.linspace(-1.5, 1.5, 256)
    X, Y = torch.meshgrid(x, y, indexing="xy")

    # Create cube with light from the corner
    light_dir = torch.tensor([1.0, 1.0, 1.0])
    cube = SimpleCube(
        center=[0.0, 0.0, 0.0], side_length=1.0, light_direction=light_dir
    )

    # Generate rendering
    radiance, mask, normals = cube.run(X, Y)

    # Visualize results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Radiance (shading)
    im1 = axes[0].imshow(
        radiance.numpy(), cmap="gray", extent=[-1.5, 1.5, -1.5, 1.5], origin="lower"
    )
    axes[0].set_title("Cube Radiance (Shading)")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    plt.colorbar(im1, ax=axes[0])

    # Mask (visible surface)
    im2 = axes[1].imshow(
        mask.numpy(), cmap="gray", extent=[-1.5, 1.5, -1.5, 1.5], origin="lower"
    )
    axes[1].set_title("Visible Surface Mask")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")
    plt.colorbar(im2, ax=axes[1])

    # Normal visualization (RGB)
    normal_vis = (normals.numpy().transpose(1, 2, 0) + 1) / 2
    normal_vis[~mask.numpy()] = 0
    im3 = axes[2].imshow(normal_vis, extent=[-1.5, 1.5, -1.5, 1.5], origin="lower")
    axes[2].set_title("Surface Normals (RGB)")
    axes[2].set_xlabel("X")
    axes[2].set_ylabel("Y")

    plt.tight_layout()
    plt.show()

    print(f"Light direction: {cube.I}")
    print(f"Radiance range: [{radiance.min():.3f}, {radiance.max():.3f}]")
    print(f"Visible pixels: {mask.sum().item()}/{mask.numel()}")

