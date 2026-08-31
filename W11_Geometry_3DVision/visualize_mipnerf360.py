import os
import struct
import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


CAMERA_MODEL_IDS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}

DEFAULT_DATA_ROOT = Path("data/360_v2")


def read_bytes(fid, num_bytes, fmt):
    data = fid.read(num_bytes)
    return struct.unpack("<" + fmt, data)


def qvec_to_rotmat(qvec):
    """
    COLMAP quaternion format: [qw, qx, qy, qz]
    This returns world-to-camera rotation R.
    """
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2 * qy**2 - 2 * qz**2, 2 * qx * qy - 2 * qw * qz, 2 * qx * qz + 2 * qw * qy],
        [2 * qx * qy + 2 * qw * qz, 1 - 2 * qx**2 - 2 * qz**2, 2 * qy * qz - 2 * qw * qx],
        [2 * qx * qz - 2 * qw * qy, 2 * qy * qz + 2 * qw * qx, 1 - 2 * qx**2 - 2 * qy**2],
    ])


def read_cameras_binary(path):
    cameras = {}

    with open(path, "rb") as fid:
        num_cameras = read_bytes(fid, 8, "Q")[0]

        for _ in range(num_cameras):
            camera_id, model_id, width, height = read_bytes(fid, 24, "iiQQ")

            if model_id not in CAMERA_MODEL_IDS:
                raise ValueError(f"Unknown camera model id: {model_id}")

            model_name, num_params = CAMERA_MODEL_IDS[model_id]
            params = np.array(read_bytes(fid, 8 * num_params, "d" * num_params))

            cameras[camera_id] = {
                "model": model_name,
                "width": width,
                "height": height,
                "params": params,
            }

    return cameras


def read_images_binary(path):
    images = {}

    with open(path, "rb") as fid:
        num_images = read_bytes(fid, 8, "Q")[0]

        for _ in range(num_images):
            image_properties = read_bytes(fid, 64, "idddddddi")

            image_id = image_properties[0]
            qvec = np.array(image_properties[1:5])
            tvec = np.array(image_properties[5:8])
            camera_id = image_properties[8]

            name = b""
            current_char = fid.read(1)
            while current_char != b"\x00":
                name += current_char
                current_char = fid.read(1)
            image_name = name.decode("utf-8")

            num_points2d = read_bytes(fid, 8, "Q")[0]
            fid.read(24 * num_points2d)

            images[image_id] = {
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": image_name,
            }

    return images


def read_points3d_binary(path):
    xyzs = []
    rgbs = []

    with open(path, "rb") as fid:
        num_points = read_bytes(fid, 8, "Q")[0]

        for _ in range(num_points):
            point_properties = read_bytes(fid, 43, "QdddBBBd")

            xyz = np.array(point_properties[1:4])
            rgb = np.array(point_properties[4:7])

            track_length = read_bytes(fid, 8, "Q")[0]
            fid.read(8 * track_length)

            xyzs.append(xyz)
            rgbs.append(rgb)

    return np.array(xyzs), np.array(rgbs)


def get_intrinsics(camera):
    """
    Return fx, fy, cx, cy.
    Distortion parameters are ignored for frustum visualization.
    """
    model = camera["model"]
    params = camera["params"]

    if model in [
        "SIMPLE_PINHOLE",
        "SIMPLE_RADIAL",
        "RADIAL",
        "SIMPLE_RADIAL_FISHEYE",
        "RADIAL_FISHEYE",
    ]:
        f = params[0]
        cx = params[1]
        cy = params[2]
        fx = f
        fy = f

    elif model in [
        "PINHOLE",
        "OPENCV",
        "OPENCV_FISHEYE",
        "FULL_OPENCV",
        "FOV",
        "THIN_PRISM_FISHEYE",
    ]:
        fx = params[0]
        fy = params[1]
        cx = params[2]
        cy = params[3]

    else:
        raise ValueError(f"Unsupported camera model: {model}")

    return fx, fy, cx, cy


def camera_to_world_from_colmap(qvec, tvec):
    """
    COLMAP stores world-to-camera:
        x_cam = R * x_world + t

    Camera center in world:
        C = -R.T @ t

    Camera-to-world transform:
        R_cw = R.T
        t_cw = C
    """
    R_wc = qvec_to_rotmat(qvec)
    R_cw = R_wc.T
    t_cw = -R_cw @ tvec

    T_cw = np.eye(4)
    T_cw[:3, :3] = R_cw
    T_cw[:3, 3] = t_cw

    return T_cw


def make_camera_frustum(camera, T_cw, scale):
    width = camera["width"]
    height = camera["height"]
    fx, fy, cx, cy = get_intrinsics(camera)

    corners_px = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height],
    ], dtype=np.float64)

    corners_cam = []
    for u, v in corners_px:
        x = (u - cx) / fx * scale
        y = (v - cy) / fy * scale
        z = scale
        corners_cam.append([x, y, z])

    corners_cam = np.array(corners_cam)
    center_cam = np.array([[0.0, 0.0, 0.0]])

    points_cam = np.vstack([center_cam, corners_cam])
    points_world = (T_cw[:3, :3] @ points_cam.T).T + T_cw[:3, 3]

    # index 0 is camera center, 1-4 are image-plane corners
    edges = [
        (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 2), (2, 3), (3, 4), (4, 1),
    ]

    return points_world, edges


def resolve_scene_dir(scene_arg, data_root):
    scene_path = Path(scene_arg)
    if scene_path.exists():
        return scene_path

    candidate = data_root / scene_arg
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        f"Could not find scene '{scene_arg}'. "
        f"Tried '{scene_path}' and '{candidate}'."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene",
        type=str,
        required=True,
        help="Scene name like 'bonsai' or a full path to a Mip-NeRF 360 scene folder.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=str(DEFAULT_DATA_ROOT),
        help="Base directory used when --scene is provided as a scene name.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output HTML file. Defaults to output/<scene_name>_colmap_visualization.html.",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=80000,
        help="Maximum number of COLMAP sparse points to visualize.",
    )
    parser.add_argument(
        "--every_nth_camera",
        type=int,
        default=1,
        help="Visualize every N-th camera.",
    )
    parser.add_argument(
        "--camera_scale",
        type=float,
        default=None,
        help="Camera frustum scale. If not set, it is estimated from point cloud size.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    scene_dir = resolve_scene_dir(args.scene, data_root)
    scene_name = scene_dir.name
    output_path = Path(args.output) if args.output else Path("output") / f"{scene_name}_colmap_visualization.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sparse_dir = scene_dir / "sparse" / "0"
    if not sparse_dir.exists():
        sparse_dir = scene_dir / "sparse"

    cameras_path = sparse_dir / "cameras.bin"
    images_path = sparse_dir / "images.bin"
    points3d_path = sparse_dir / "points3D.bin"

    if not cameras_path.exists():
        raise FileNotFoundError(f"Cannot find {cameras_path}")
    if not images_path.exists():
        raise FileNotFoundError(f"Cannot find {images_path}")
    if not points3d_path.exists():
        raise FileNotFoundError(f"Cannot find {points3d_path}")

    print(f"Reading COLMAP model from: {sparse_dir}")

    cameras = read_cameras_binary(cameras_path)
    images = read_images_binary(images_path)
    xyzs, rgbs = read_points3d_binary(points3d_path)

    import pdb; pdb.set_trace()

    print(f"Loaded {len(cameras)} cameras")
    print(f"Loaded {len(images)} registered images")
    print(f"Loaded {len(xyzs)} sparse 3D points")

    if len(xyzs) > args.max_points:
        idx = np.random.choice(len(xyzs), args.max_points, replace=False)
        xyzs_vis = xyzs[idx]
        rgbs_vis = rgbs[idx]
    else:
        xyzs_vis = xyzs
        rgbs_vis = rgbs

    bbox_min = xyzs_vis.min(axis=0)
    bbox_max = xyzs_vis.max(axis=0)
    scene_diag = np.linalg.norm(bbox_max - bbox_min)

    camera_scale = args.camera_scale
    if camera_scale is None:
        camera_scale = scene_diag * 0.035

    point_colors = [
        f"rgb({int(r)}, {int(g)}, {int(b)})"
        for r, g, b in rgbs_vis
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=xyzs_vis[:, 0],
        y=xyzs_vis[:, 1],
        z=xyzs_vis[:, 2],
        mode="markers",
        marker=dict(
            size=1.5,
            color=point_colors,
            opacity=0.75,
        ),
        name="COLMAP sparse points",
    ))

    cam_centers = []
    line_x, line_y, line_z = [], [], []

    sorted_images = sorted(images.items(), key=lambda x: x[0])

    for i, (_, image) in enumerate(sorted_images):
        if i % args.every_nth_camera != 0:
            continue

        camera = cameras[image["camera_id"]]
        T_cw = camera_to_world_from_colmap(image["qvec"], image["tvec"])

        cam_centers.append(T_cw[:3, 3])

        frustum_points, edges = make_camera_frustum(camera, T_cw, camera_scale)

        for a, b in edges:
            p1 = frustum_points[a]
            p2 = frustum_points[b]

            line_x += [p1[0], p2[0], None]
            line_y += [p1[1], p2[1], None]
            line_z += [p1[2], p2[2], None]

    cam_centers = np.array(cam_centers)

    fig.add_trace(go.Scatter3d(
        x=cam_centers[:, 0],
        y=cam_centers[:, 1],
        z=cam_centers[:, 2],
        mode="markers",
        marker=dict(size=3),
        name="Camera centers",
    ))

    fig.add_trace(go.Scatter3d(
        x=line_x,
        y=line_y,
        z=line_z,
        mode="lines",
        line=dict(width=2),
        name="Camera frustums",
    ))

    fig.update_layout(
        title=f"Mip-NeRF 360 {scene_name}: COLMAP Cameras and Sparse 3D Points",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=True,
    )

    fig.write_html(output_path)
    print(f"Saved visualization to: {output_path}")


if __name__ == "__main__":
    main()
