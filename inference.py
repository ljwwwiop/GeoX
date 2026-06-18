"""
minivggt Inference Script

This script performs 3D reconstruction from images using the minivggt model.
It supports:
- Loading images and optional depth maps and camera parameters
- Running inference to predict depth and camera poses
- Visualizing results in 3D using viser
- Exporting results to GLB format
- Multi-frame temporal visualization:
  * Single frame: Pass a folder containing surround-view images (e.g., /path/to/frame_13)
  * Temporal sequence: Pass a parent folder containing multiple frame_XX subfolders
    (e.g., /path/to/ddad_images with frame_13/ and frame_44/)
  * Each frame is independently inferred and maintains its own coordinate system
  * All frames are visualized together in a single canvas for comparison
"""

import os, pdb
import argparse
import threading
import time
from typing import List, Optional
import open3d as o3d

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import viser
import viser.transforms as viser_tf
from safetensors.torch import load_file
from tqdm import tqdm

from geox.models.geox import GeoX
from geox.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from geox.utils.misc import select_first_batch
from geox.utils.pose_enc import pose_encoding_to_extri_intri
from visual_util import (
    apply_sky_segmentation,
    get_world_points_from_depth,
    load_images_and_cameras,
    predictions_to_glb,
)



def viser_wrapper(
    pred_dict: dict,
    port: int = 8080,
    init_conf_threshold: float = 20.0,  # represents percentage (e.g., 20 means filter lowest 20%)
    use_point_map: bool = False,
    background_mode: bool = False,
    mask_sky: bool = False,
    mask_black_bg: bool = False,
    mask_white_bg: bool = False,
    image_folder: Optional[str] = None,
    batch_sizes: Optional[List[int]] = None,
    ):
    """
    Visualize predicted 3D points and camera poses with viser.

    Args:
        pred_dict (dict):
            {
                "images": (S, 3, H, W)   - Input images,
                "world_points": (S, H, W, 3),
                "world_points_conf": (S, H, W),
                "depth": (S, H, W, 1),
                "depth_conf": (S, H, W),
                "extrinsic": (S, 3, 4),
                "intrinsic": (S, 3, 3),
            }
        port (int): Port number for the viser server.
        init_conf_threshold (float): Initial percentage of low-confidence points to filter out.
        use_point_map (bool): Whether to visualize world_points or use depth-based points.
        background_mode (bool): Whether to run the server in background thread.
        mask_sky (bool): Whether to apply sky segmentation to filter out sky points.
        mask_black_bg (bool): Whether to mask out black background pixels.
        mask_white_bg (bool): Whether to mask out white background pixels.
        image_folder (str): Path to the folder containing input images.
        batch_sizes (List[int]): List indicating how many frames belong to each batch (e.g., [6, 6] for 2 batches of 6 frames each).
    """
    print(f"Starting viser server on port {port}")
    # pdb.set_trace()

    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    # Unpack prediction dict
    images = pred_dict["images"]  # (S, 3, H, W)
    depth_map = pred_dict["depth"]  # (S, H, W, 1)
    depth_conf = pred_dict["depth_conf"]  # (S, H, W)
    extrinsics_cam = pred_dict["extrinsic"]  # (S, 3, 4)
    intrinsics_cam = pred_dict["intrinsic"]  # (S, 3, 3)

    # Compute world points from depth if not using the precomputed point map
    if use_point_map and "world_points" in pred_dict:
        # Use precomputed world points if available
        world_points = pred_dict["world_points"]  # (S, H, W, 3)
        conf = pred_dict.get("world_points_conf", depth_conf)  # (S, H, W)
    else:
        # Compute world points by unprojecting depth map
        world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)
        conf = depth_conf

    # Apply sky segmentation if enabled
    if mask_sky and image_folder is not None:
        conf = apply_sky_segmentation(conf, image_folder)
        
    # Convert images from (S, 3, H, W) to (S, H, W, 3)
    # Then flatten everything for the point cloud
    colors = images.transpose(0, 2, 3, 1)  # now (S, H, W, 3)
    S, H, W, _ = world_points.shape

    # Flatten
    points = world_points.reshape(-1, 3)
    colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)
    conf_flat = conf.reshape(-1)

    cam_to_world_mat = closed_form_inverse_se3(extrinsics_cam)  # shape (S, 4, 4) typically
    # For convenience, we store only (3,4) portion
    cam_to_world = cam_to_world_mat[:, :3, :]

    # Handle multi-batch: each batch maintains its own coordinate system
    if batch_sizes is not None and len(batch_sizes) > 1:
        print(f"\n{'='*60}")
        print(f"Multi-batch mode: {len(batch_sizes)} batches with independent coordinate systems")
        print(f"Batch sizes: {batch_sizes}")
        print(f"{'='*60}\n")

        # Store batch indices
        batch_indices = np.concatenate([
            np.full(batch_size * H * W, batch_idx)
            for batch_idx, batch_size in enumerate(batch_sizes)
        ])

        # Process each batch independently - each gets centered to its own scene center
        points_centered = np.zeros_like(points)
        cam_to_world_adjusted = cam_to_world.copy()

        frame_offset = 0
        point_offset = 0

        for batch_idx, batch_size in enumerate(batch_sizes):
            # Extract points for this batch
            batch_point_start = point_offset
            batch_point_end = point_offset + batch_size * H * W
            batch_points = points[batch_point_start:batch_point_end]

            # Compute this batch's scene center
            batch_scene_center = np.mean(batch_points, axis=0)

            # Center this batch's points around its own scene center
            batch_points_centered = batch_points - batch_scene_center

            # Store centered points (no additional offset - each batch stays in its own coordinate)
            points_centered[batch_point_start:batch_point_end] = batch_points_centered

            # Adjust camera positions for this batch
            batch_frame_start = frame_offset
            batch_frame_end = frame_offset + batch_size
            cam_to_world_adjusted[batch_frame_start:batch_frame_end, :, -1] -= batch_scene_center

            print(f"Batch {batch_idx}: {batch_size} frames, scene center = {batch_scene_center}")

            frame_offset += batch_size
            point_offset += batch_size * H * W

        cam_to_world = cam_to_world_adjusted

    else:
        # Single batch mode: center at origin
        batch_indices = None
        scene_center = np.mean(points, axis=0)
        points_centered = points - scene_center
        cam_to_world[..., -1] -= scene_center
        print(f"Single batch mode: scene center = {scene_center}")

    # Store frame indices so we can filter by frame
    frame_indices = np.repeat(np.arange(S), H * W)

    # Build the viser GUI
    gui_show_frames = server.gui.add_checkbox("Show Cameras", initial_value=True)

    # Now the slider represents percentage of points to filter out
    gui_points_conf = server.gui.add_slider(
        "Confidence Percent", min=0, max=100, step=0.1, initial_value=init_conf_threshold
    )

    # Add batch selector if multiple batches are present
    if batch_sizes is not None and len(batch_sizes) > 1:
        batch_options = ["All"] + [f"Batch {i}" for i in range(len(batch_sizes))]
        gui_batch_selector = server.gui.add_dropdown(
            "Show Batch", options=batch_options, initial_value="All"
        )
    else:
        gui_batch_selector = None

    gui_frame_selector = server.gui.add_dropdown(
        "Show Points from Frames", options=["All"] + [str(i) for i in range(S)], initial_value="All"
    )

    # Create the main point cloud handle
    # Compute the threshold value as the given percentile
    init_threshold_val = np.percentile(conf_flat, init_conf_threshold)
    init_conf_mask = (conf_flat >= init_threshold_val) & (conf_flat > 0.1)
    
    # Apply black background mask if enabled
    if mask_black_bg:
        black_bg_mask = colors_flat.sum(axis=1) >= 16
        init_conf_mask = init_conf_mask & black_bg_mask
    
    # Apply white background mask if enabled
    if mask_white_bg:
        white_bg_mask = ~((colors_flat[:, 0] > 240) & (colors_flat[:, 1] > 240) & (colors_flat[:, 2] > 240))
        init_conf_mask = init_conf_mask & white_bg_mask
    
    point_cloud = server.scene.add_point_cloud(
        name="viser_pcd",
        points=points_centered[init_conf_mask],
        colors=colors_flat[init_conf_mask],
        point_size=0.008,  # Changed from 0.001 to 0.01 for denser appearance
        point_shape="circle",
    )

    # We will store references to frames & frustums so we can toggle visibility
    frames: List[viser.FrameHandle] = []
    frustums: List[viser.CameraFrustumHandle] = []

    def visualize_frames(extrinsics: np.ndarray, images_: np.ndarray) -> None:
        """
        Add camera frames and frustums to the scene.
        extrinsics: (S, 3, 4)
        images_:    (S, 3, H, W)
        """
        # Clear any existing frames or frustums
        for f in frames:
            f.remove()
        frames.clear()
        for fr in frustums:
            fr.remove()
        frustums.clear()

        # Optionally attach a callback that sets the viewpoint to the chosen camera
        def attach_callback(frustum: viser.CameraFrustumHandle, frame: viser.FrameHandle) -> None:
            @frustum.on_click
            def _(_) -> None:
                for client in server.get_clients().values():
                    client.camera.wxyz = frame.wxyz
                    client.camera.position = frame.position

        img_ids = range(S)
        for img_id in tqdm(img_ids):
            cam2world_3x4 = extrinsics[img_id]
            T_world_camera = viser_tf.SE3.from_matrix(cam2world_3x4)

            # Add a small frame axis
            frame_axis = server.scene.add_frame(
                f"frame_{img_id}",
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.05,
                axes_radius=0.002,
                origin_radius=0.002,
            )
            frames.append(frame_axis)

            # Convert the image for the frustum
            img = images_[img_id]  # shape (3, H, W)
            img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
            h, w = img.shape[:2]

            # If you want correct FOV from intrinsics, do something like:
            # fx = intrinsics_cam[img_id, 0, 0]
            # fov = 2 * np.arctan2(h/2, fx)
            # For demonstration, we pick a simple approximate FOV:
            fy = 1.1 * h
            fov = 2 * np.arctan2(h / 2, fy)

            # Add the frustum
            frustum_cam = server.scene.add_camera_frustum(
                f"frame_{img_id}/frustum", fov=fov, aspect=w / h, scale=0.05, image=img, line_width=1.0
            )
            frustums.append(frustum_cam)
            attach_callback(frustum_cam, frame_axis)

    def update_point_cloud() -> None:
        """Update the point cloud based on current GUI selections."""
        # Here we compute the threshold value based on the current percentage
        current_percentage = gui_points_conf.value
        threshold_val = np.percentile(conf_flat, current_percentage)

        print(f"Threshold absolute value: {threshold_val}, percentage: {current_percentage}%")

        conf_mask = (conf_flat >= threshold_val) & (conf_flat > 1e-5)

        # Apply black background mask if enabled
        if mask_black_bg:
            black_bg_mask = colors_flat.sum(axis=1) >= 16
            conf_mask = conf_mask & black_bg_mask

        # Apply white background mask if enabled
        if mask_white_bg:
            white_bg_mask = ~((colors_flat[:, 0] > 240) & (colors_flat[:, 1] > 240) & (colors_flat[:, 2] > 240))
            conf_mask = conf_mask & white_bg_mask

        # Apply batch filter if multiple batches exist
        if gui_batch_selector is not None and batch_indices is not None:
            if gui_batch_selector.value == "All":
                batch_mask = np.ones_like(conf_mask, dtype=bool)
            else:
                selected_batch_idx = int(gui_batch_selector.value.split()[-1])
                batch_mask = batch_indices == selected_batch_idx
        else:
            batch_mask = np.ones_like(conf_mask, dtype=bool)

        if gui_frame_selector.value == "All":
            frame_mask = np.ones_like(conf_mask, dtype=bool)
        else:
            selected_idx = int(gui_frame_selector.value)
            frame_mask = frame_indices == selected_idx

        combined_mask = conf_mask & frame_mask & batch_mask
        point_cloud.points = points_centered[combined_mask]
        point_cloud.colors = colors_flat[combined_mask]

    @gui_points_conf.on_update
    def _(_) -> None:
        update_point_cloud()

    @gui_frame_selector.on_update
    def _(_) -> None:
        update_point_cloud()

    if gui_batch_selector is not None:
        @gui_batch_selector.on_update
        def _(_) -> None:
            update_point_cloud()

    @gui_show_frames.on_update
    def _(_) -> None:
        """Toggle visibility of camera frames and frustums."""
        for f in frames:
            f.visible = gui_show_frames.value
        for fr in frustums:
            fr.visible = gui_show_frames.value

    # Add the camera frames to the scene
    visualize_frames(cam_to_world, images)

    print("Starting viser server...")
    # If background_mode is True, spawn a daemon thread so the main thread can continue.
    if background_mode:

        def server_loop():
            while True:
                time.sleep(0.001)

        thread = threading.Thread(target=server_loop, daemon=True)
        thread.start()
    else:
        while True:
            time.sleep(0.01)

    return server

# Command-line argument parser
parser = argparse.ArgumentParser(
    description="minivggt demo with viser for 3D visualization"
)

# Input data arguments
parser.add_argument("--image_folder",type=str,help="Path to folder containing images")
parser.add_argument("--camera_folder",type=str,default=None,help="Path to folder containing camera files (.txt)")

# Processing options
parser.add_argument("--use_point_map",action="store_true",help="Use point map instead of depth-based points")
parser.add_argument("--mask_sky",action="store_true",help="Apply sky segmentation to filter out sky points")
parser.add_argument("--mask_black_bg",action="store_true",help="Mask out black background pixels")
parser.add_argument("--mask_white_bg",action="store_true",help="Mask out white background pixels")
parser.add_argument("--target_size",type=int,default=518,help="Target size for the images")

# Visualization options
parser.add_argument("--background_mode",action="store_true",help="Run the viser server in background mode")
parser.add_argument("--port",type=int,default=8082,help="Port number for the viser server")
parser.add_argument("--conf_threshold",type=float,default=10.0,help="Initial percentage of low-confidence points to filter out")

# Export options
parser.add_argument("--save_glb",action="store_true",help="Save the output as a GLB file")

def main():
    """
    Main function for the minivggt demo with viser for 3D visualization.

    This function:
    1. Loads the minivggt model
    2. Processes input images from the specified folder (single or multiple batches)
    3. Runs inference to generate 3D points and camera poses
    4. Optionally applies sky segmentation to filter out sky points
    5. Visualizes the results using viser
    """
    # pdb.set_trace()
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize and load minivggt model
    print("Initializing and loading minivggt model...")

    ##============== MINI-SMALL MODEL ===========
    model = GeoX()

    ckpt_path = "./checkpoints/model.safetensors"

    model_params_mb = (sum(p.numel() for p in model.parameters())) / (1e6)
    print(f"Models Params Size ===>>  {model_params_mb:.2f} MB")

    state_dict = load_file(ckpt_path)
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()

    # =========================================================
    # Detect if input path is a single frame or parent folder containing multiple frames
    # =========================================================
    image_folder = args.image_folder
    frame_folders = []
    batch_sizes = []

    # Check if the input folder contains subdirectories named "frame_XX"
    if os.path.isdir(image_folder):
        # Look for frame_XX subdirectories (exclude sky_masks and other auxiliary folders)
        potential_frames = sorted([
            d for d in os.listdir(image_folder)
            if os.path.isdir(os.path.join(image_folder, d))
            and d.startswith("frame_")
            and "sky_mask" not in d.lower()  # Exclude sky_masks directories
        ])

        if potential_frames:
            # Parent folder with multiple frames detected
            print(f"\n{'='*60}")
            print(f"Multiple frames detected in {image_folder}")
            print(f"Found {len(potential_frames)} frames: {', '.join(potential_frames)}")
            print(f"{'='*60}\n")

            for frame_name in potential_frames:
                frame_path = os.path.join(image_folder, frame_name)
                frame_folders.append(frame_path)
        else:
            # Single folder (either single frame or direct image folder)
            print(f"\n{'='*60}")
            print(f"Single batch mode: loading from {image_folder}")
            print(f"{'='*60}\n")
            frame_folders.append(image_folder)
    else:
        print(f"Error: {image_folder} is not a valid directory")
        return

    # Load input data from all detected frames
    all_predictions = []
    all_frame_names = []

    for frame_idx, frame_folder in enumerate(frame_folders):
        print(f"\n{'='*60}")
        print(f"Processing frame {frame_idx + 1}/{len(frame_folders)}: {os.path.basename(frame_folder)}")
        print(f"{'='*60}")

        # Determine camera folder for this frame
        if args.camera_folder is not None:
            frame_camera_folder = os.path.join(args.camera_folder, os.path.basename(frame_folder)) \
                if len(frame_folders) > 1 else args.camera_folder
        else:
            frame_camera_folder = None

        print(f"Loading images from {frame_folder}...")
        if frame_camera_folder is not None:
            print(f"Loading cameras from {frame_camera_folder}...")

        # Load this frame's data
        images, extrinsics, intrinsics, _, _, _, camera_indices = \
            load_images_and_cameras(
                frame_folder,
                frame_camera_folder,
                None,
                args.target_size
            )

        # Prepare model inputs for this frame
        inputs = {
            'images': images.to(device),
            'extrinsics': extrinsics.to(device),
            'intrinsics': intrinsics.to(device),
            'camera_gt_index': camera_indices
        }

        print(f"Running inference for frame {frame_idx + 1}...")
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            predictions = model.inference(**inputs)

        torch.cuda.synchronize()
        end_time = time.perf_counter()
        inference_time = (end_time - start_time) * 1000
        print(f"⏱️  Inference time for frame {frame_idx + 1}: {inference_time:.2f}ms")

        # Convert pose encoding to camera matrices
        print("Converting pose encoding to extrinsic and intrinsic matrices...")
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            predictions["pose_enc"],
            images.shape[-2:]
        )
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic
        predictions["images"] = images

        # Store predictions for this frame
        all_predictions.append(predictions)
        all_frame_names.append(os.path.basename(frame_folder))

    # Combine predictions for visualization
    print(f"\n{'='*60}")
    print(f"Combining {len(all_predictions)} frame(s) for visualization...")
    if len(all_predictions) > 1:
        print(f"Each frame maintains its own coordinate system")
    print(f"{'='*60}\n")

    # Merge all predictions into a single dict for visualization
    # Each prediction has batch dimension (B=1), we need to squeeze it first
    combined_predictions = {
        "images": torch.cat([p["images"].squeeze(0) if p["images"].dim() > 3 else p["images"] for p in all_predictions], dim=0),
        "depth": torch.cat([p["depth"].squeeze(0) if p["depth"].dim() > 4 else p["depth"] for p in all_predictions], dim=0),
        "depth_conf": torch.cat([p["depth_conf"].squeeze(0) if p["depth_conf"].dim() > 3 else p["depth_conf"] for p in all_predictions], dim=0),
        "extrinsic": torch.cat([p["extrinsic"].squeeze(0) if p["extrinsic"].dim() > 3 else p["extrinsic"] for p in all_predictions], dim=0),
        "intrinsic": torch.cat([p["intrinsic"].squeeze(0) if p["intrinsic"].dim() > 3 else p["intrinsic"] for p in all_predictions], dim=0),
    }

    # Handle optional world_points if present
    if "world_points" in all_predictions[0]:
        combined_predictions["world_points"] = torch.cat([p["world_points"].squeeze(0) if p["world_points"].dim() > 4 else p["world_points"] for p in all_predictions], dim=0)
    if "world_points_conf" in all_predictions[0]:
        combined_predictions["world_points_conf"] = torch.cat([p["world_points_conf"].squeeze(0) if p["world_points_conf"].dim() > 3 else p["world_points_conf"] for p in all_predictions], dim=0)

    # Store batch sizes for GUI filtering (after squeeze, each prediction contributes S cameras)
    batch_sizes = [p["images"].squeeze(0).shape[0] if p["images"].dim() > 3 else p["images"].shape[0] for p in all_predictions]

    # Export to GLB if requested
    if args.save_glb:
        print("Exporting scene to GLB...")
        predictions_0 = select_first_batch(combined_predictions)
        get_world_points_from_depth(predictions_0)
        glbscene = predictions_to_glb(
            predictions_0,
            conf_thres=0.0,
            filter_by_frames='All',
            mask_white_bg=False,
            show_cam=True,
            mask_sky=False,
            target_dir=frame_folders[0],
            prediction_mode="Predicted Depth",
        )
        glb_path = os.path.join(frame_folders[0], 'scene.glb')
        glbscene.export(file_obj=glb_path)
        print(f"Saved GLB file to {glb_path}")

    # Convert predictions to numpy for visualization
    print("Processing model outputs...")
    for key in combined_predictions.keys():
        if isinstance(combined_predictions[key], torch.Tensor):
            # Don't squeeze - keep the batch dimension for multi-frame visualization
            combined_predictions[key] = combined_predictions[key].cpu().numpy()

    # Print visualization mode
    if args.use_point_map:
        print("Visualizing 3D points from point map")
    else:
        print("Visualizing 3D points by unprojecting depth map by cameras")

    if args.mask_sky:
        print("Sky segmentation enabled - will filter out sky points")

    if args.mask_black_bg:
        print("Black background masking enabled - will filter out black background points")

    if args.mask_white_bg:
        print("White background masking enabled - will filter out white background points")

    # Start visualization server
    print("Starting viser visualization...")
    viser_server = viser_wrapper(
        combined_predictions,
        port=args.port,
        init_conf_threshold=args.conf_threshold,
        use_point_map=args.use_point_map,
        background_mode=args.background_mode,
        mask_sky=args.mask_sky,
        mask_black_bg=args.mask_black_bg,
        mask_white_bg=args.mask_white_bg,
        image_folder=frame_folders[0] if len(frame_folders) == 1 else args.image_folder,
        batch_sizes=batch_sizes if len(batch_sizes) > 1 else None,
    )
    print("Visualization complete")


if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=7

'''
=== Single Frame Example ===
CUDA_VISIBLE_DEVICES=2 python inference.py \
    --image_folder /lpai/volumes/mind-vla-ali-sh-mix/lianjiawei/clean_codebase/minivggt/exp/ddad_images/imgs \
    --port 8082

=== RGB+Sky Mask Example ===
CUDA_VISIBLE_DEVICES=2 python inference.py \
    --image_folder /lpai/volumes/mind-vla-ali-sh-mix/lianjiawei/clean_codebase/minivggt/exp/ddad_images/imgs \
    --port 8082 --mask_sky

=== RGB+Pose Example ===
CUDA_VISIBLE_DEVICES=2 python inference.py \
    --image_folder /lpai/volumes/mind-vla-ali-sh-mix/lianjiawei/clean_codebase/minivggt/exp/ddad_images \
    --camera_folder /lpai/volumes/mind-vla-ali-sh-mix/lianjiawei/clean_codebase/minivggt/exp/ddad_cameras \
    --port 8083

'''


