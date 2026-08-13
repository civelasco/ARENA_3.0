# %%
import os
import sys
from functools import partial
from pathlib import Path
from typing import Callable

import einops
import plotly.express as px
import plotly.graph_objects as go
import torch as t
from IPython.display import display
from ipywidgets import interact
from jaxtyping import Bool, Float
from torch import Tensor
from tqdm import tqdm

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part1_ray_tracing"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part1_ray_tracing.tests as tests
from part1_ray_tracing.utils import (
    render_lines_with_plotly,
    setup_widget_fig_ray,
    setup_widget_fig_triangle,
)
from plotly_utils import imshow

MAIN = __name__ == "__main__"
# %%
def make_rays_1d(num_pixels: int, y_limit: float) -> Tensor:
    """
    num_pixels: The number of pixels in the y dimension. Since there is one ray per pixel, this is
        also the number of rays.
    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both endpoints.

    Returns: shape (num_pixels, num_points=2, num_dim=3) where the num_points dimension contains
        (origin, direction) and the num_dim dimension contains xyz.

    Example of make_rays_1d(9, 1.0): [
        [[0, 0, 0], [1, -1.0, 0]],
        [[0, 0, 0], [1, -0.75, 0]],
        [[0, 0, 0], [1, -0.5, 0]],
        ...
        [[0, 0, 0], [1, 0.75, 0]],
        [[0, 0, 0], [1, 1, 0]],
    ]
    """
    rays = t.zeros((num_pixels,2,3))
    rays[:,1,0] = t.ones(num_pixels)
    rays[:,1,1] = t.linspace(-1,1,num_pixels)
    return rays


rays1d = make_rays_1d(9, 10.0)
fig = render_lines_with_plotly(rays1d)
# %%
def intersect_ray_1d(ray: Float[Tensor, "points dims"], segment: Float[Tensor, "points dims"]) -> bool:
    """
    ray: shape (n_points=2, n_dim=3)  # O, D points
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points

    Return True if the ray intersects the segment.
    """
    ray = ray[:,:2]
    seg = segment[:,:2]
    segdif = seg[0]-seg[1]
    odif   = seg[0]-ray[0]
    matrix = t.stack((ray[1],segdif), dim = -1)
    try:
        intersect = t.linalg.solve(matrix,odif)
    except RuntimeError:
        return False
    u = intersect[0].item()
    v = intersect[1].item()
    return (u >= 0.0) and (v >= 0.0) and (v <= 1.0)



tests.test_intersect_ray_1d(intersect_ray_1d)
tests.test_intersect_ray_1d_special_case(intersect_ray_1d)
# %%
def intersect_rays_1d(
    rays: Float[Tensor, "nrays 2 3"], segments: Float[Tensor, "nsegments 2 3"]
) -> Bool[Tensor, " nrays"]:
    """
    For each ray, return True if it intersects any segment.
    """
    rays = rays[...,:2]
    nrays = len(rays)
    segs = segments[...,:2]
    nsegs = len(segs)

    segdif = segs[:,0]-segs[:,1]
    segdif = einops.repeat(segdif, "i j -> n i j", n = nrays)
    rays_mat = einops.repeat(rays[:,1], "i j -> i n j", n = nsegs)
    mat = t.stack((rays_mat,segdif), dim = -1)

    dets = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    mat[is_singular] = t.eye(2)

    odif = segs[:,0]-einops.repeat(rays[:,0], "i j -> i n j", n = nsegs)
    intersect = t.linalg.solve(mat,odif)
    u = intersect[...,0]
    v = intersect[...,1]
    return ((u>=0.0) & (v>=0.0) & (v<=1.0) &~is_singular).any(dim = -1)


tests.test_intersect_rays_1d(intersect_rays_1d)
tests.test_intersect_rays_1d_special_case(intersect_rays_1d)

# %%
def make_rays_2d(num_pixels_y: int, num_pixels_z: int, y_limit: float, z_limit: float) -> Float[Tensor, "nrays 2 3"]:
    """
    num_pixels_y: The number of pixels in the y dimension
    num_pixels_z: The number of pixels in the z dimension

    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both.
    z_limit: At x=1, the rays should extend from -z_limit to +z_limit, inclusive of both.

    Returns: shape (num_rays=num_pixels_y * num_pixels_z, num_points=2, num_dims=3).
    """
    rays = t.zeros((num_pixels_y*num_pixels_z,2,3))
    extent_y = t.linspace(-y_limit,y_limit,num_pixels_y)
    extent_y = einops.repeat(extent_y, "i -> (i nz)", nz = num_pixels_z)
    extent_z = t.linspace(-z_limit,z_limit,num_pixels_z)
    extent_z = einops.repeat(extent_z, "i -> (ny i)", ny = num_pixels_y)

    nrays = num_pixels_y*num_pixels_z
    rays = t.zeros((nrays,2,3))
    rays[:,1,0] = t.ones(nrays)
    rays[:,1,1] = extent_y
    rays[:,1,2] = extent_z
    return rays
    


rays_2d = make_rays_2d(10, 10, 0.3, 0.3)
render_lines_with_plotly(rays_2d)

# %%
Point = Float[Tensor, "points=3"]


def triangle_ray_intersects(A: Point, B: Point, C: Point, O: Point, D: Point) -> bool:
    """
    A: shape (3,), one vertex of the triangle
    B: shape (3,), second vertex of the triangle
    C: shape (3,), third vertex of the triangle
    O: shape (3,), origin point
    D: shape (3,), direction point

    Return True if the ray and the triangle intersect.
    """
    matrix = t.stack((-D, B-A, C-A), dim=-1)
    odif   = O-A
    try:
        intersect = t.linalg.solve(matrix,odif)
    except RuntimeError:
        return False
    s,u,v = intersect
    return (s >= 0) and (u >= 0) and (v >= 0) and (u+v <= 1)

tests.test_triangle_ray_intersects(triangle_ray_intersects)

# %%
def raytrace_triangle(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangle: Float[Tensor, "trianglePoints=3 dims=3"],
) -> Bool[Tensor, " nrays"]:
    """
    For each ray, return True if the triangle intersects that ray.
    """
    nrays    = rays.size(0)
    triangle = einops.repeat(triangle,'i j -> nr i j', nr = nrays)
    O,D = rays.unbind(1)
    A,B,C = triangle.unbind(1)
    mat  = t.stack((-D,B-A,C-A),dim=-1)
    dets = mat.det()
    is_singular = (dets < 1e-8)
    mat[is_singular] = t.eye(3)
    odif = O-A
    intersect = t.linalg.solve(mat,odif)
    s,u,v = intersect.unbind(1)
    return ((s >= 0.0) & (u >= 0.0) & (v >= 0.0) & (u+v <= 1.0) &~is_singular)


A = t.tensor([1, 0.0, -0.5])
B = t.tensor([1, -0.5, 0.0])
C = t.tensor([1, 0.5, 0.5])
num_pixels_y = num_pixels_z = 30
y_limit = z_limit = 0.5

# Plot triangle & rays
test_triangle = t.stack([A, B, C], dim=0)
rays2d = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
triangle_lines = t.stack([A, B, C, A, B, C], dim=0).reshape(-1, 2, 3)
render_lines_with_plotly(rays2d, triangle_lines)

# Calculate and display intersections
intersects = raytrace_triangle(rays2d, test_triangle)
img = intersects.reshape(num_pixels_y, num_pixels_z).int()
imshow(img, origin="lower", width=600, title="Triangle (as intersected by rays)")

# %%
triangles = t.load(section_dir / "pikachu.pt", weights_only=True)
# %%
def raytrace_mesh(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangles: Float[Tensor, "ntriangles trianglePoints=3 dims=3"],
) -> Float[Tensor, " nrays"]:
    """
    For each ray, return the distance to the closest intersecting triangle, or infinity.
    """
    nrays = rays.size(0)
    ntriangles = triangles.size(0)
    rays = einops.repeat(rays, "nr p d -> nt nr p d", nt = ntriangles)
    triangles = einops.repeat(triangles, "nt p d -> nt nr p d", nr = nrays)
    O,D = rays.unbind(2)
    A,B,C = triangles.unbind(2)
    mat  = t.stack((-D,B-A,C-A),dim=-1)
    dets = mat.det()
    is_singular = (dets.abs() < 1e-8)
    mat[is_singular] = t.eye(3)
    odif = O-A
    intersect = t.linalg.solve(mat,odif)
    s,u,v = intersect.unbind(-1)
    s *= D[...,0]
    intersecting = ((u >= 0.0) & (v >= 0.0) & (u+v <= 1.0) &~is_singular)
    s[~intersecting] = float("inf")
    return einops.reduce(s,"i j -> j", "min")




num_pixels_y = 120
num_pixels_z = 120
y_limit = z_limit = 1

rays = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
rays[:, 0] = t.tensor([-2, 0.0, 0.0])
dists = raytrace_mesh(rays, triangles)
intersects = t.isfinite(dists).view(num_pixels_y, num_pixels_z)
dists_square = dists.view(num_pixels_y, num_pixels_z)
img = t.stack([intersects, dists_square], dim=0)

fig = px.imshow(img, facet_col=0, origin="lower", color_continuous_scale="magma", width=1000)
fig.update_layout(coloraxis_showscale=False)
for i, text in enumerate(["Intersects", "Distance"]):
    fig.layout.annotations[i]["text"] = text
fig.show()

# %%
def rotation_matrix(theta: Float[Tensor, ""]) -> Float[Tensor, "rows cols"]:
    """
    Creates a rotation matrix representing a counterclockwise rotation of `theta` around the y-axis.
    """
    sint = t.sin(theta)
    cost = t.cos(theta)
    ry = t.zeros((3,3))
    ry[0,0] = ry[2,2] = cost
    ry[0,2] = sint; ry[2,0] = -sint
    ry[1,1] = 1
    return ry


tests.test_rotation_matrix(rotation_matrix)
# %%
def raytrace_mesh_video(
    rays: Float[Tensor, "nrays points dim"],
    triangles: Float[Tensor, "ntriangles points dims"],
    rotation_matrix: Callable[[float], Float[Tensor, "rows cols"]],
    raytrace_function: Callable,
    num_frames: int,
) -> Bool[Tensor, "nframes nrays"]:
    """
    Creates a stack of raytracing results, rotating the triangles by `rotation_matrix` each frame.
    """
    result = []
    theta = t.tensor(2 * t.pi) / num_frames
    R = rotation_matrix(theta)
    for theta in tqdm(range(num_frames)):
        triangles = triangles @ R
        result.append(raytrace_function(rays, triangles))
        t.cuda.empty_cache()  # clears GPU memory (this line will be more important later on!)
    return t.stack(result, dim=0)


def display_video(distances: Float[Tensor, "frames y z"]):
    """
    Displays video of raytracing results, using Plotly. `distances` is a tensor where the [i, y, z]
    element is distance to the closest triangle for the i-th frame & the [y, z]-th ray in our 2D
    grid of rays.
    """
    px.imshow(
        distances,
        animation_frame=0,
        origin="lower",
        zmin=0.0,
        zmax=distances[distances.isfinite()].quantile(0.99).item(),
        color_continuous_scale="viridis_r",  # "Brwnyl"
    ).update_layout(coloraxis_showscale=False, width=550, height=600, title="Raytrace mesh video").show()


num_pixels_y = 250
num_pixels_z = 250
y_limit = z_limit = 0.8
num_frames = 50

rays = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
rays[:, 0] = t.tensor([-3.0, 0.0, 0.0])
dists = raytrace_mesh_video(rays, triangles, rotation_matrix, raytrace_mesh, num_frames)
dists = einops.rearrange(dists, "frames (y z) -> frames y z", y=num_pixels_y)

display_video(dists)
# %%
def raytrace_mesh_gpu(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangles: Float[Tensor, "ntriangles trianglePoints=3 dims=3"],
) -> Float[Tensor, " nrays"]:
    """
    For each ray, return the distance to the closest intersecting triangle, or infinity.

    All computations should be performed on the GPU.
    """
    #device = "mps" if t.backends.mps.is_available() else "cpu"
    #rays = rays.to(device)
    #triangles = triangles.to(device)
    nrays = rays.size(0)
    ntriangles = triangles.size(0)
    rays = einops.repeat(rays, "nr p d -> nt nr p d", nt = ntriangles)
    triangles = einops.repeat(triangles, "nt p d -> nt nr p d", nr = nrays)
    O,D = rays.unbind(2)
    A,B,C = triangles.unbind(2)
    mat  = t.stack((-D,B-A,C-A),dim=-1)
    dets = mat.det()
    is_singular = (dets.abs() < 1e-8)
    mat[is_singular] = t.eye(3)
    odif = O-A
    intersect = t.linalg.solve(mat,odif)
    s,u,v = intersect.unbind(-1)
    s *= D[...,0]
    intersecting = ((u >= 0.0) & (v >= 0.0) & (u+v <= 1.0) &~is_singular)
    s[~intersecting] = float("inf")
    return einops.reduce(s,"i j -> j", "min").to("cpu")


dists = raytrace_mesh_video(rays, triangles, rotation_matrix, raytrace_mesh_gpu, num_frames)
dists = einops.rearrange(dists, "frames (y z) -> frames y z", y=num_pixels_y)
display_video(dists)
# %%
def raytrace_mesh_lambert(
    rays: Float[Tensor, "nrays points=2 dims=3"],
    triangles: Float[Tensor, "ntriangles points=3 dims=3"],
    light: Float[Tensor, "dims=3"],
    ambient_intensity: float,
    device: str = "cuda",
) -> Float[Tensor, " nrays"]:
    """
    For each ray, return the intensity of light hitting the triangle it intersects with (or zero if
    no intersection).

    Args:
        rays:   A tensor of rays, with shape `[nrays, 2, 3]`.
        triangles:  A tensor of triangles, with shape `[ntriangles, 3, 3]`.
        light:  A tensor representing the light vector, with shape `[3]`. We compute the intensity
                as the dot product of the triangle normals & the light vector, then set it to be
                zero if the sign is negative.
        ambient_intensity:  A float representing the ambient intensity. This is the minimum
                            brightness for a triangle, to differentiate it from the black background
                            (rays that don't hit any triangle).
        device: The device to perform the computation on.

    Returns:
        A tensor of intensities for each of the rays, flattened over the [y, z] dimensions. The
        values are zero when there is no intersection, and `ambient_intensity + intensity` when
        there is an interesection (where `intensity` is the dot product of the triangle's normal
        vector and the light vector, truncated at zero).
    """
    nrays = rays.size(0)
    ntriangles = triangles.size(0)
    rays = einops.repeat(rays, "nr p d -> nt nr p d", nt = ntriangles)
    triangles = einops.repeat(triangles, "nt p d -> nt nr p d", nr = nrays)
    O,D = rays.unbind(2)
    A,B,C = triangles.unbind(2)
    mat  = t.stack((-D,B-A,C-A),dim=-1)
    dets = mat.det()
    is_singular = (dets.abs() < 1e-8)
    mat[is_singular] = t.eye(3)
    odif = O-A
    intersect = t.linalg.solve(mat,odif)
    s,u,v = intersect.unbind(-1)
    s *= D[...,0]
    intersecting = ((s>= 0.0) & (u >= 0.0) & (v >= 0.0) & (u+v <= 1.0) &~is_singular)
    s[~intersecting] = float("inf")
    mindist,mintriang = s.min(dim=0)
    normals = t.cross((B-A)[:,0],(C-A)[:,0],dim=-1)
    normals = normals/normals.norm(dim=-1,keepdim=True)
    intensity_per_triangle = einops.einsum(normals,light,"nt dims, dims -> nt")
    intensity_per_triangle_sign = t.where(intensity_per_triangle>0,intensity_per_triangle,0.0)
    intensity = intensity_per_triangle_sign[mintriang]+ambient_intensity
    intensity = t.where(mindist.isfinite(),intensity,0.0)
    return intensity

def display_video_with_lighting(intensity: Float[Tensor, "frames y z"]):
    """
    Displays video of raytracing results, using Plotly. `distances` is a tensor where the [i, y, z]
    element is the lighting intensity based on the angle of light & the surface of the triangle
    which this ray hits first.
    """
    px.imshow(
        intensity,
        animation_frame=0,
        origin="lower",
        color_continuous_scale="magma",
    ).update_layout(coloraxis_showscale=False, width=550, height=600, title="Raytrace mesh video (lighting)").show()


ambient_intensity = 0.5
light = t.tensor([0.0, -1.0, 1.0])
raytrace_function = partial(
    raytrace_mesh_lambert,
    ambient_intensity=ambient_intensity,
    light=light,
)

intensity = raytrace_mesh_video(rays, triangles, rotation_matrix, raytrace_function, num_frames)
intensity = einops.rearrange(intensity, "frames (y z) -> frames y z", y=num_pixels_y)
display_video_with_lighting(intensity)

# %%
