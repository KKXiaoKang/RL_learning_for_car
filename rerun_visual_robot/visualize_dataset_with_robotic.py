#!/usr/bin/env python
"""
Enhanced Dataset Visualization using Robotic Class

This script uses the Robotic class to visualize LeRobotDataset episodes with
improved data handling, robot state management, and 3D visualization.

The Robotic class provides:
- Structured data extraction from datasets
- Camera pose and image visualization  
- Joint state management and forward kinematics
- Integrated robot model configuration

Usage:
    python3 rerun_visual_robot/visualize_dataset_with_robotic.py \
        --repo-id /path/to/dataset \
        --episode-index 10

Example:
    python3 rerun_visual_robot/visualize_dataset_with_robotic.py \
        --repo-id /home/lab/RL/lerobot_data/rl_graspbox_increase_0818_vision_random/ \
        --episode-index 10
"""

import argparse
import gc
import logging
import time
from pathlib import Path
from typing import Iterator, Dict, Any, Optional
import os

import numpy as np
import rerun as rr
import torch
import torch.utils.data
import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# Import the Robotic system
from common.robot_system import (
    Robotic, create_kuavo_robot, create_dual_arm_robot,
    LeRobotDatasetWrapper, ModelMode, TorsoConfig
)

# Import the Robotic-Rerun adapter
from robotic_rerun_adapter import RoboticRerunAdapter, create_adapter_for_dataset

# Import the original visualization functions for camera poses and robot visualization
from visualize_dataset_robotics import (
    _visualize_camera_poses, log_robot_visualization, 
    initialize_robot_visualizer, _configure_robot_view
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EpisodeSampler(torch.utils.data.Sampler):
    """Sampler for extracting a specific episode from dataset"""
    
    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self) -> Iterator:
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)


def to_hwc_uint8_numpy(chw_float32_torch: torch.Tensor) -> np.ndarray:
    """Convert CHW float32 tensor to HWC uint8 numpy array"""
    assert chw_float32_torch.dtype == torch.float32
    assert chw_float32_torch.ndim == 3
    c, h, w = chw_float32_torch.shape
    assert c < h and c < w, f"expect channel first images, but instead {chw_float32_torch.shape}"
    hwc_uint8_numpy = (chw_float32_torch * 255).type(torch.uint8).permute(1, 2, 0).numpy()
    return hwc_uint8_numpy


def create_robot_adapter_for_dataset(repo_id: str, root: Optional[Path] = None) -> RoboticRerunAdapter:
    """
    Create and configure robot adapter for the given dataset
    
    Args:
        repo_id: Dataset repository ID
        root: Root directory for dataset
        
    Returns:
        Configured RoboticRerunAdapter instance
    """
    # Create adapter using convenience function
    adapter = create_adapter_for_dataset(repo_id, str(root) if root else None)
    
    # Load dataset to get metadata
    dataset = adapter.robot.dataset_wrapper.load_dataset()
    
    # Configure cameras based on dataset
    adapter.robot.camera_system.cameras.clear()  # Clear default cameras
    
    for camera_key in dataset.meta.camera_keys:
        camera_name = camera_key.replace('observation.images.', '').replace('.', '_')
        adapter.robot.add_camera(camera_name, camera_type='RGB')
        logger.info(f"Added camera: {camera_name} (from {camera_key})")
    
    logger.info(f"Robot configured with {len(adapter.robot.get_all_cameras())} cameras")
    logger.info(adapter.get_robot_info_summary())
    
    return adapter





def visualize_dataset_with_robotic(
    adapter: RoboticRerunAdapter,
    episode_index: int,
    batch_size: int = 32,
    num_workers: int = 0,
    mode: str = "local",
    web_port: int = 9090,
    ws_port: int = 9087,
    save: bool = False,
    output_dir: Optional[Path] = None,
) -> Optional[Path]:
    """
    Visualize dataset episode using Robotic adapter
    
    Args:
        adapter: Configured RoboticRerunAdapter instance
        episode_index: Episode index to visualize
        batch_size: DataLoader batch size
        num_workers: Number of DataLoader workers
        mode: Visualization mode ('local' or 'distant')
        web_port: Web port for distant mode
        ws_port: WebSocket port for distant mode
        save: Whether to save .rrd file
        output_dir: Output directory for saving
        
    Returns:
        Path to saved .rrd file if save=True, else None
    """
    if save:
        assert output_dir is not None, "Set output directory with --output-dir"
    
    # Get dataset from robot
    robot = adapter.robot
    dataset = robot.dataset_wrapper.dataset
    repo_id = dataset.repo_id
    
    logger.info("Setting up dataloader")
    episode_sampler = EpisodeSampler(dataset, episode_index)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        sampler=episode_sampler,
    )
    
    logger.info("Starting Rerun")
    
    if mode not in ["local", "distant"]:
        raise ValueError(f"Invalid mode: {mode}")
    
    spawn_local_viewer = mode == "local" and not save
    rr.init(f"{repo_id}/episode_{episode_index}", spawn=spawn_local_viewer)
    
    # Manually call garbage collector to avoid hanging
    gc.collect()
    
    # Configure robot-centric view settings - same as original
    rr.log("robot_view", rr.ViewCoordinates.RIGHT_HAND_Y_UP)
    
    # Configure a good viewing angle for robot observation
    # This creates the viewer camera frame (not a robot camera)
    rr.log("robot_view/viewer_camera", 
           rr.ViewCoordinates(
               xyz=rr.ViewCoordinates.RUB  # Right, Up, Back
           ))
    
    logger.info("Robot view configured. Adjust camera position in Rerun viewer for optimal robot observation.")
    
    if mode == "distant":
        rr.serve(open_browser=False, web_port=web_port, ws_port=ws_port)
    
    logger.info("Processing dataset frames")
    
    # Get episode info
    episode_info = robot.dataset_wrapper.get_episode_info(episode_index)
    logger.info(f"Episode {episode_index}: {episode_info}")
    
    for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
        # Iterate over batch items
        for i in range(len(batch["index"])):
            frame_idx = batch["frame_index"][i].item()
            timestamp = batch["timestamp"][i].item()
            
            # Set rerun timeline
            rr.set_time("frame_index", sequence=frame_idx)
            rr.set_time("timestamp", timestamp=timestamp)
            
            # Display each camera image (like original)
            for key in dataset.meta.camera_keys:
                # TODO(rcadene): add `.compress()`? is it lossless?
                rr.log(key, rr.Image(to_hwc_uint8_numpy(batch[key][i])))
            
            # Create data item for robot processing
            data_item = {key: batch[key][i] for key in batch.keys()}
            
            # Load robot state from data
            robot.load_state_from_data(data_item)
            
            # NOTE: Commented out the new robot camera visualization to avoid duplicate camera frame creation
            # robot.update_camera_poses_from_robot_state()
            # robot.visualize_camera_poses_in_rerun()
            
            # Use original camera pose visualization (this is the one we want to keep)
            _visualize_camera_poses(dataset.meta.camera_keys, batch, i)
            
            # Extract and visualize robot arm joints (like original)
            if "observation.state" in batch:
                state_data = batch["observation.state"][i].numpy()
                
                # Try to extract arm joints based on different possible state configurations
                arm_joints = None
                if len(state_data) >= 28:
                    # For WBC mode: arm joints are typically at indices 6-19 (14 joints total)
                    arm_joints = state_data[6:20]
                elif len(state_data) >= 21:
                    # For other configurations, arm joints might be at different positions
                    for start_idx in range(len(state_data) - 13):
                        potential_joints = state_data[start_idx:start_idx+14]
                        # Check if values are in reasonable joint angle range (-π to π)
                        if np.all(np.abs(potential_joints) < 4.0):  # Reasonable joint angle range
                            arm_joints = potential_joints
                            break
                
                if arm_joints is not None and len(arm_joints) == 14:
                    try:
                        # Use the original robot visualization function
                        log_robot_visualization(arm_joints, frame_idx)
                    except Exception as e:
                        logger.warning(f"Failed to visualize robot at frame {frame_idx}: {e}")
                else:
                    # Log a message about state structure for debugging
                    if i == 0:  # Only log once per batch to avoid spam
                        logger.info(f"State shape: {state_data.shape}, cannot extract 14 arm joints for robot visualization")
                
                # Log state data as scalars
                for dim_idx, val in enumerate(batch["observation.state"][i]):
                    rr.log(f"state/{dim_idx}", rr.Scalars(val.item()))
            
            # Log action data
            if "action" in batch:
                for dim_idx, val in enumerate(batch["action"][i]):
                    rr.log(f"action/{dim_idx}", rr.Scalars(val.item()))
            
            # Log episode metrics
            if "next.done" in batch:
                rr.log("next.done", rr.Scalars(batch["next.done"][i].item()))
            
            if "next.reward" in batch:
                rr.log("next.reward", rr.Scalars(batch["next.reward"][i].item()))
            
            if "next.success" in batch:
                rr.log("next.success", rr.Scalars(batch["next.success"][i].item()))
    
    # Save or serve
    if mode == "local" and save:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        repo_id_str = repo_id.replace("/", "_")
        rrd_path = output_dir / f"{repo_id_str}_episode_{episode_index}.rrd"
        rr.save(rrd_path)
        logger.info(f"Saved visualization to: {rrd_path}")
        return rrd_path
    
    elif mode == "distant":
        try:
            logger.info(f"Serving on web port {web_port}, websocket port {ws_port}")
            logger.info(f"Connect with: rerun ws://localhost:{ws_port}")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Ctrl-C received. Exiting.")
    
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Visualize LeRobotDataset using enhanced Robotic class"
    )
    
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository ID or path to LeRobotDataset (e.g., '/path/to/dataset' or 'lerobot/pusht')"
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        required=True,
        help="Episode index to visualize"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root directory for local datasets"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for saving .rrd files"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="DataLoader batch size"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="local",
        choices=["local", "distant"],
        help="Visualization mode"
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=9090,
        help="Web port for distant mode"
    )
    parser.add_argument(
        "--ws-port",
        type=int,
        default=9087,
        help="WebSocket port for distant mode"
    )
    parser.add_argument(
        "--save",
        type=int,
        default=0,
        help="Save .rrd file (1) or spawn viewer (0)"
    )
    parser.add_argument(
        "--tolerance-s",
        type=float,
        default=1e-4,
        help="Timestamp tolerance for dataset loading"
    )
    
    args = parser.parse_args()
    
    try:
        # Create and configure robot adapter for dataset
        logger.info(f"Creating robot adapter for dataset: {args.repo_id}")
        adapter = create_robot_adapter_for_dataset(args.repo_id, args.root)
        
        # Validate robot configuration
        issues = adapter.robot.validate_configuration()
        if issues:
            logger.warning(f"Robot configuration issues: {issues}")
        
        # Visualize dataset
        result = visualize_dataset_with_robotic(
            adapter=adapter,
            episode_index=args.episode_index,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            mode=args.mode,
            web_port=args.web_port,
            ws_port=args.ws_port,
            save=bool(args.save),
            output_dir=args.output_dir
        )
        
        if result:
            logger.info(f"Visualization completed: {result}")
        else:
            logger.info("Visualization completed")
            
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
