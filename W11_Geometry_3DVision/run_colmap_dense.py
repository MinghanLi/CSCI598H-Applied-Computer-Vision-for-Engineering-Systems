import argparse
import shutil
import subprocess
from pathlib import Path


DEFAULT_DATA_ROOT = Path("data/360_v2")
DEFAULT_WORKSPACE_ROOT = Path("output/dense")


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


def find_sparse_dir(scene_dir):
    sparse_zero = scene_dir / "sparse" / "0"
    if sparse_zero.exists():
        return sparse_zero

    sparse_dir = scene_dir / "sparse"
    if sparse_dir.exists():
        return sparse_dir

    raise FileNotFoundError(f"Could not find sparse COLMAP model under '{scene_dir}'.")


def run_command(cmd, dry_run):
    print("Running:")
    print(" ".join(str(part) for part in cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def copy_depth_maps(source_dir, destination_dir, dry_run):
    if not source_dir.exists():
        raise FileNotFoundError(f"Depth map directory not found: {source_dir}")

    depth_map_files = sorted(source_dir.glob("*"))
    if not depth_map_files:
        raise FileNotFoundError(f"No depth maps found in: {source_dir}")

    print(f"Saving {len(depth_map_files)} depth map files to: {destination_dir}")
    if dry_run:
        return

    destination_dir.mkdir(parents=True, exist_ok=True)
    for path in depth_map_files:
        if path.is_file():
            shutil.copy2(path, destination_dir / path.name)


def main():
    parser = argparse.ArgumentParser(
        description="Step 2: run COLMAP dense reconstruction from an existing sparse model."
    )
    parser.add_argument(
        "--scene",
        required=True,
        help="Scene name like 'bonsai' or a full path to a scene folder.",
    )
    parser.add_argument(
        "--data_root",
        default=str(DEFAULT_DATA_ROOT),
        help="Base directory used when --scene is provided as a scene name.",
    )
    parser.add_argument(
        "--workspace_root",
        default=str(DEFAULT_WORKSPACE_ROOT),
        help="Base output directory for dense reconstruction results.",
    )
    parser.add_argument(
        "--image_dir",
        default="images",
        help="Image subdirectory inside the scene folder.",
    )
    parser.add_argument(
        "--colmap_bin",
        default="colmap",
        help="Path to the COLMAP executable, for example '/opt/homebrew/bin/colmap'.",
    )
    parser.add_argument(
        "--fusion_output",
        default="fused.ply",
        help="Filename for the fused dense point cloud.",
    )
    parser.add_argument(
        "--depth_output",
        default="saved_depth_maps",
        help="Directory name used to store copied depth maps.",
    )
    parser.add_argument(
        "--geom_consistency",
        action="store_true",
        help="Enable geometric consistency during PatchMatch stereo.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print COLMAP commands without executing them.",
    )
    args = parser.parse_args()

    colmap_path = shutil.which(args.colmap_bin)
    if colmap_path is None:
        raise SystemExit(
            "COLMAP executable not found.\n"
            "Install COLMAP and make sure it is on your PATH, or pass the full binary path with:\n"
            "  python3 run_colmap_dense.py --scene bonsai --colmap_bin /opt/homebrew/bin/colmap"
        )
    args.colmap_bin = colmap_path

    scene_dir = resolve_scene_dir(args.scene, Path(args.data_root))
    scene_name = scene_dir.name
    sparse_dir = find_sparse_dir(scene_dir)
    image_dir = scene_dir / args.image_dir

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    workspace_dir = Path(args.workspace_root) / scene_name
    workspace_dir.mkdir(parents=True, exist_ok=True)

    fusion_output_path = workspace_dir / args.fusion_output
    depth_output_dir = workspace_dir / args.depth_output

    print(f"Scene: {scene_name}")
    print(f"Images: {image_dir}")
    print(f"Sparse model: {sparse_dir}")
    print(f"Dense workspace: {workspace_dir}")

    undistort_cmd = [
        args.colmap_bin,
        "image_undistorter",
        "--image_path",
        str(image_dir),
        "--input_path",
        str(sparse_dir),
        "--output_path",
        str(workspace_dir),
        "--output_type",
        "COLMAP",
    ]

    patch_match_cmd = [
        args.colmap_bin,
        "patch_match_stereo",
        "--workspace_path",
        str(workspace_dir),
        "--workspace_format",
        "COLMAP",
        "--PatchMatchStereo.geom_consistency",
        "true" if args.geom_consistency else "false",
    ]

    fusion_cmd = [
        args.colmap_bin,
        "stereo_fusion",
        "--workspace_path",
        str(workspace_dir),
        "--workspace_format",
        "COLMAP",
        "--input_type",
        "geometric" if args.geom_consistency else "photometric",
        "--output_path",
        str(fusion_output_path),
    ]

    run_command(undistort_cmd, args.dry_run)
    run_command(patch_match_cmd, args.dry_run)
    run_command(fusion_cmd, args.dry_run)

    depth_maps_dir = workspace_dir / "stereo" / "depth_maps"
    copy_depth_maps(depth_maps_dir, depth_output_dir, args.dry_run)

    print(f"Saved fused dense point cloud to: {fusion_output_path}")
    print(f"Saved copied depth maps to: {depth_output_dir}")


if __name__ == "__main__":
    main()
