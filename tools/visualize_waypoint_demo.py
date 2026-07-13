"""Visualize and validate one RLBench live-waypoint demonstration."""

import argparse
from io import BytesIO
import pickle
from pathlib import Path
from typing import Optional

import imageio.v2 as imageio
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CAMERAS = ("front", "left_shoulder", "overhead", "right_shoulder", "wrist")
DEPTH_SCALE = 2**24 - 1
DEFAULT_EPISODE = Path(
    "/home/pyun/Docker/RM01/Agent/Code/RLBench/data/waypoint_demos/"
    "basketball_in_hoop/variation0/episodes/episode0"
)
TIMES_NEW_ROMAN_FALLBACK = Path(
    "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode_dir", type=Path, default=DEFAULT_EPISODE)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--point_stride", type=int, default=8)
    parser.add_argument("--gif_duration_ms", type=int, default=50)
    parser.add_argument("--max_frames", type=int, default=None)
    return parser.parse_args()


def numeric_png_ids(directory: Path):
    ids = []
    for path in directory.glob("*.png"):
        if path.stem.isdigit():
            ids.append(int(path.stem))
    return sorted(ids)


def decode_metric_depth(path: Path, near: float, far: float) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint32)
    encoded = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
    depth_norm = encoded.astype(np.float32) / DEPTH_SCALE
    return float(near) + depth_norm * (float(far) - float(near))


def load_pickle(path: Path):
    with path.open("rb") as file:
        return pickle.load(file)


def validate_episode(episode_dir: Path):
    if not episode_dir.is_dir():
        raise FileNotFoundError(f"Episode directory does not exist: {episode_dir}")

    frame_sets = {}
    for camera in CAMERAS:
        for modality in ("rgb", "depth", "mask"):
            key = f"{camera}_{modality}"
            frame_sets[key] = numeric_png_ids(episode_dir / key)
    frame_sets["camera_poses"] = sorted(
        int(path.stem)
        for path in (episode_dir / "camera_poses").glob("*.pkl")
        if path.stem.isdigit()
    )
    expected = frame_sets["front_rgb"]
    if not expected:
        raise ValueError("No front RGB frames were found.")
    for name, frame_ids in frame_sets.items():
        if frame_ids != expected:
            raise ValueError(
                f"Frame IDs for {name} do not match front_rgb: "
                f"{len(frame_ids)} vs {len(expected)}."
            )
    if expected != list(range(expected[-1] + 1)):
        raise ValueError(f"Frame IDs must be continuous, got {expected[:5]} ...")

    low_dim = load_pickle(episode_dir / "low_dim_obs.pkl")
    object_poses = load_pickle(episode_dir / "object_poses.pkl")
    if len(low_dim) != len(expected):
        raise ValueError(
            f"low_dim_obs has {len(low_dim)} frames; expected {len(expected)}."
        )
    if len(object_poses) != len(expected):
        raise ValueError(
            f"object_poses has {len(object_poses)} frames; expected {len(expected)}."
        )
    return expected, low_dim, object_poses


def world_to_camera(world_xyz: np.ndarray, extrinsics: np.ndarray):
    rotation = np.asarray(extrinsics, dtype=np.float64)[:3, :3]
    translation = np.asarray(extrinsics, dtype=np.float64)[:3, 3]
    return (np.asarray(world_xyz, dtype=np.float64) - translation) @ rotation


def project_world_points(world_xyz: np.ndarray, camera):
    camera_xyz = world_to_camera(world_xyz, camera["extrinsics"])
    intrinsics = np.asarray(camera["intrinsics"], dtype=np.float64)
    z = camera_xyz[:, 2]
    valid = z > 1e-6
    u = np.zeros(len(camera_xyz), dtype=np.float64)
    v = np.zeros(len(camera_xyz), dtype=np.float64)
    u[valid] = intrinsics[0, 0] * camera_xyz[valid, 0] / z[valid] + intrinsics[0, 2]
    v[valid] = intrinsics[1, 1] * camera_xyz[valid, 1] / z[valid] + intrinsics[1, 2]
    return u, v, z, valid


def depth_to_world(depth: np.ndarray, camera, stride: int):
    height, width = depth.shape
    ys, xs = np.mgrid[0:height:stride, 0:width:stride]
    z = depth[::stride, ::stride]
    intrinsics = np.asarray(camera["intrinsics"], dtype=np.float64)
    camera_xyz = np.stack(
        [
            (xs - intrinsics[0, 2]) * z / intrinsics[0, 0],
            (ys - intrinsics[1, 2]) * z / intrinsics[1, 1],
            z,
        ],
        axis=-1,
    ).reshape(-1, 3)
    valid = np.isfinite(camera_xyz).all(axis=1) & (camera_xyz[:, 2] > 1e-6)
    rotation = np.asarray(camera["extrinsics"], dtype=np.float64)[:3, :3]
    translation = np.asarray(camera["extrinsics"], dtype=np.float64)[:3, 3]
    world_xyz = camera_xyz @ rotation.T + translation
    return world_xyz[valid], valid.reshape(z.shape)


def pose_axes(pose: np.ndarray, axis_length: float):
    pose = np.asarray(pose, dtype=np.float64)
    position = pose[:3]
    x, y, z, w = pose[3:7]
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    endpoints = position[None] + rotation.T * axis_length
    return np.vstack([position, endpoints])


def draw_pose_on_image(image: np.ndarray, pose: np.ndarray, camera, depth: np.ndarray,
                       axis_length: float, label: Optional[str] = None,
                       line_width: int = 2, draw_origin: bool = False,
                       occlusion_tolerance: float = 0.025,
                       label_font_size: Optional[int] = None):
    axes = pose_axes(pose, axis_length)
    u, v, z, valid = project_world_points(axes, camera)
    height, width = image.shape[:2]
    visible = valid.copy()
    for index in range(len(axes)):
        if not visible[index]:
            continue
        x_pixel, y_pixel = int(round(u[index])), int(round(v[index]))
        if not (0 <= x_pixel < width and 0 <= y_pixel < height):
            visible[index] = False
            continue
        visible[index] = (
            z[index] <= float(depth[y_pixel, x_pixel]) + occlusion_tolerance
        )

    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    for axis, color in enumerate(((255, 0, 0), (0, 255, 0), (0, 128, 255)), start=1):
        if visible[0] and visible[axis]:
            draw.line(
                [(int(round(u[0])), int(round(v[0]))),
                 (int(round(u[axis])), int(round(v[axis])))],
                fill=color,
                width=line_width,
            )
    if draw_origin and visible[0]:
        x_origin, y_origin = int(round(u[0])), int(round(v[0]))
        radius = max(2, line_width)
        draw.ellipse(
            [
                (x_origin - radius, y_origin - radius),
                (x_origin + radius, y_origin + radius),
            ],
            outline="white",
            width=line_width,
        )
    if label and visible[0]:
        font = None
        if label_font_size is not None and TIMES_NEW_ROMAN_FALLBACK.exists():
            font = ImageFont.truetype(
                str(TIMES_NEW_ROMAN_FALLBACK), size=label_font_size
            )
        draw.text(
            (int(round(u[0])) + 3, int(round(v[0])) + 3),
            label,
            fill="white",
            font=font,
        )
    return np.asarray(pil_image)


def make_mask_legend(mask: np.ndarray, max_entries: int = 8):
    colors, counts = np.unique(mask.reshape(-1, 3), axis=0, return_counts=True)
    order = np.argsort(counts)[::-1]
    entries = []
    for index in order:
        color = colors[index]
        if np.all(color == 0):
            continue
        handle = int(color[0]) * 65536 + int(color[1]) * 256 + int(color[2])
        entries.append(f"{handle}: #{color[0]:02x}{color[1]:02x}{color[2]:02x}")
        if len(entries) == max_entries:
            break
    return " | ".join(entries) if entries else "background only"


def should_visualize_object(object_name: str, metadata) -> bool:
    return not (
        object_name.startswith("waypoint")
        or object_name == "success"
        or metadata["type"] == "PROXIMITY_SENSOR"
    )


def reproject_point_cloud(world_xyz: np.ndarray, rgb: np.ndarray, camera, shape):
    height, width = shape[:2]
    u, v, z, valid = project_world_points(world_xyz, camera)
    x = np.rint(u).astype(np.int64)
    y = np.rint(v).astype(np.int64)
    valid &= (x >= 0) & (x < width) & (y >= 0) & (y < height)
    output = np.zeros((height, width, 3), dtype=np.uint8)
    if not np.any(valid):
        return output

    flat = y[valid] * width + x[valid]
    depths = z[valid]
    colors = rgb[valid]
    order = np.lexsort((depths, flat))
    ordered_flat = flat[order]
    keep = np.r_[True, ordered_flat[1:] != ordered_flat[:-1]]
    output.reshape(-1, 3)[ordered_flat[keep]] = colors[order][keep]
    return output


def render_frame(frame_id, low_dim, object_poses, episode_dir: Path,
                 point_stride: int):
    cameras = load_pickle(episode_dir / "camera_poses" / f"{frame_id}.pkl")
    rgbs, depths, masks = {}, {}, {}
    fused_xyz, fused_rgb = [], []
    fused_xyz_dense, fused_rgb_dense = [], []
    for name in CAMERAS:
        rgb = np.asarray(Image.open(episode_dir / f"{name}_rgb" / f"{frame_id}.png").convert("RGB"))
        depth = decode_metric_depth(
            episode_dir / f"{name}_depth" / f"{frame_id}.png",
            cameras[name]["near"],
            cameras[name]["far"],
        )
        mask = np.asarray(Image.open(episode_dir / f"{name}_mask" / f"{frame_id}.png").convert("RGB"))
        if not np.isfinite(depth).all():
            raise ValueError(f"Non-finite depth in {name}, frame {frame_id}.")
        rgbs[name], depths[name], masks[name] = rgb, depth, mask
        dense_world_xyz, dense_valid_grid = depth_to_world(
            depth, cameras[name], stride=1
        )
        fused_xyz_dense.append(dense_world_xyz)
        fused_rgb_dense.append(rgb[dense_valid_grid])
        world_xyz, valid_grid = depth_to_world(depth, cameras[name], point_stride)
        fused_xyz.append(world_xyz)
        fused_rgb.append(rgb[::point_stride, ::point_stride][valid_grid])

    fused_xyz = np.concatenate(fused_xyz, axis=0)
    fused_rgb = np.concatenate(fused_rgb, axis=0)
    fused_xyz_dense = np.concatenate(fused_xyz_dense, axis=0)
    fused_rgb_dense = np.concatenate(fused_rgb_dense, axis=0)
    gripper_pose = np.asarray(low_dim[frame_id].gripper_pose, dtype=np.float64)
    frame_objects = object_poses[frame_id]
    visual_objects = {
        name: metadata
        for name, metadata in frame_objects.items()
        if should_visualize_object(name, metadata)
    }

    figure = plt.figure(figsize=(25, 20), constrained_layout=True)
    grid = figure.add_gridspec(4, 5, height_ratios=(1, 1, 1.35, 1))
    figure.suptitle(
        f"frame {frame_id:04d} | points: {len(fused_xyz):,} | "
        f"objects: {len(visual_objects)}/{len(frame_objects)}",
        fontsize=16,
    )

    for column, name in enumerate(CAMERAS):
        axis = figure.add_subplot(grid[0, column])
        axis.imshow(rgbs[name])
        axis.set_title(name)
        axis.axis("off")

        axis = figure.add_subplot(grid[1, column])
        axis.imshow(masks[name])
        handle_count = len(
            [entry for entry in make_mask_legend(masks[name]).split(" | ")
             if entry != "background only"]
        )
        axis.set_title(f"{name} mask ({handle_count}+ handles)", fontsize=9)
        axis.axis("off")

    axis_3d = figure.add_subplot(grid[2, :4], projection="3d")
    axis_3d.scatter(
        fused_xyz[:, 0], fused_xyz[:, 1], fused_xyz[:, 2],
        c=fused_rgb / 255.0, s=0.6, alpha=0.65,
    )
    for pose, label, scale in [(gripper_pose, "gripper", 0.07)]:
        axes = pose_axes(pose, scale)
        axis_3d.plot(axes[[0, 1], 0], axes[[0, 1], 1], axes[[0, 1], 2], "r-", label=label)
        axis_3d.plot(axes[[0, 2], 0], axes[[0, 2], 1], axes[[0, 2], 2], "g-")
        axis_3d.plot(axes[[0, 3], 0], axes[[0, 3], 1], axes[[0, 3], 2], "b-")
    for metadata in visual_objects.values():
        axes = pose_axes(np.asarray(metadata["pose"]), 0.025)
        axis_3d.plot(axes[[0, 1], 0], axes[[0, 1], 1], axes[[0, 1], 2], "r-", alpha=0.65)
        axis_3d.plot(axes[[0, 2], 0], axes[[0, 2], 1], axes[[0, 2], 2], "g-", alpha=0.65)
        axis_3d.plot(axes[[0, 3], 0], axes[[0, 3], 1], axes[[0, 3], 2], "b-", alpha=0.65)
    axis_3d.set_title("Fused RGB point cloud reconstructed from metric depth")
    axis_3d.set_xlabel("world x")
    axis_3d.set_ylabel("world y")
    axis_3d.set_zlabel("world z")
    zoom_half_range = 0.75
    axis_3d.set_xlim(
        gripper_pose[0] - zoom_half_range, gripper_pose[0] + zoom_half_range
    )
    axis_3d.set_ylim(
        gripper_pose[1] - zoom_half_range, gripper_pose[1] + zoom_half_range
    )
    axis_3d.set_zlim(
        gripper_pose[2] - zoom_half_range, gripper_pose[2] + zoom_half_range
    )
    axis_3d.set_box_aspect((1, 1, 0.7))
    axis_3d.view_init(elev=26, azim=-68)

    axis_open = figure.add_subplot(grid[2, 4])
    gripper_open_history = np.asarray(
        [float(low_dim[index].gripper_open) for index in range(frame_id + 1)]
    )
    axis_open.plot(
        np.arange(frame_id + 1),
        gripper_open_history,
        color="tab:purple",
        linewidth=2,
    )
    axis_open.scatter(
        [frame_id],
        [gripper_open_history[-1]],
        color="tab:red",
        s=35,
        zorder=3,
    )
    axis_open.set_xlim(0, max(1, frame_id))
    axis_open.set_ylim(-0.05, 1.05)
    axis_open.set_xlabel("frame")
    axis_open.set_ylabel("gripper open")
    axis_open.set_title("Gripper openness")
    axis_open.grid(alpha=0.3)

    for column, name in enumerate(CAMERAS):
        reprojection = reproject_point_cloud(
            fused_xyz_dense, fused_rgb_dense, cameras[name], rgbs[name].shape
        )
        reprojection = draw_pose_on_image(
            reprojection, gripper_pose, cameras[name], depths[name], 0.07, "gripper"
        )
        for object_name, metadata in visual_objects.items():
            reprojection = draw_pose_on_image(
                reprojection, np.asarray(metadata["pose"]), cameras[name],
                depths[name], 0.06, line_width=3, draw_origin=True,
                occlusion_tolerance=0.05,
                label=object_name,
                label_font_size=10,
            )
        axis = figure.add_subplot(grid[3, column])
        axis.imshow(reprojection)
        axis.set_title(f"{name} reprojected")
        axis.axis("off")

    image_buffer = BytesIO()
    figure.savefig(image_buffer, dpi=120, format="png")
    plt.close(figure)
    image_buffer.seek(0)
    image = np.asarray(Image.open(image_buffer).convert("RGB")).copy()
    image_buffer.close()
    return image


def output_gif_path(episode_dir: Path, point_stride: int, frame_stride: int):
    episodes_dir = episode_dir.parent
    variation_dir = episodes_dir.parent
    task_dir = variation_dir.parent
    if episodes_dir.name != "episodes":
        raise ValueError(f"Expected an episodes directory, got: {episodes_dir}")
    if not variation_dir.name.startswith("variation"):
        raise ValueError(f"Expected variation<N>, got: {variation_dir}")
    if not episode_dir.name.startswith("episode"):
        raise ValueError(f"Expected episode<N>, got: {episode_dir}")

    variation = variation_dir.name[len("variation"):]
    episode = episode_dir.name[len("episode"):]
    data_root_dir = task_dir.parent
    return (
        data_root_dir / "visualization" /
        f"{task_dir.name}_{variation}_{episode}_{point_stride}_{frame_stride}.gif"
    )


def main():
    args = parse_args()
    if args.frame_stride < 1 or args.point_stride < 1:
        raise ValueError("frame_stride and point_stride must be at least 1.")
    frame_ids, low_dim, object_poses = validate_episode(args.episode_dir)
    selected_ids = frame_ids[::args.frame_stride]
    if args.max_frames is not None:
        selected_ids = selected_ids[:args.max_frames]
    gif_path = output_gif_path(
        args.episode_dir, args.point_stride, args.frame_stride
    )
    gif_path.parent.mkdir(parents=True, exist_ok=True)

    with imageio.get_writer(
            gif_path, mode="I", duration=args.gif_duration_ms / 1000.0, loop=0
    ) as writer:
        for frame_id in selected_ids:
            writer.append_data(render_frame(
                frame_id, low_dim, object_poses, args.episode_dir,
                args.point_stride,
            ))
            print(f"Rendered frame {frame_id}")
    print(f"Saved {len(selected_ids)} frames to {gif_path}")


if __name__ == "__main__":
    main()
