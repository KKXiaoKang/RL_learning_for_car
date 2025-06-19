#!/usr/bin/env python3
import torch
import torch.nn as nn
import os
import argparse
import gymnasium as gym
from algo.policies import Actor

class ActorForExport(nn.Module):
    """
    A wrapper for the SAC Actor model to make it compatible with ONNX export.
    This wrapper ensures that only the deterministic action is returned.
    """
    def __init__(self, actor_model: Actor):
        super().__init__()
        self.actor = actor_model

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # For inference/export, we use deterministic mode.
        # The original actor returns (action, log_prob, penalty). We only need the action.
        action, _, _ = self.actor(obs, deterministic=True)
        return action

def main():
    """Main function to convert a .pth model to .onnx."""
    parser = argparse.ArgumentParser(description="Convert a PyTorch SAC model to ONNX.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained PyTorch model (.pth) file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the ONNX model (.onnx) file.")
    args = parser.parse_args()

    pytorch_model_path = args.model_path
    onnx_output_path = args.output_path

    if not os.path.exists(pytorch_model_path):
        print(f"Error: Model file not found at {pytorch_model_path}")
        return

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Model Definition ---
    # The observation and action spaces must match the ones used during training.
    # From sac.py: obs space is (21,), action space is (2,).
    observation_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(21,))
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))

    # Instantiate the Actor model
    actor = Actor(observation_space, action_space).to(device)

    # --- Load Weights ---
    print(f"Loading weights from {pytorch_model_path}...")
    checkpoint = torch.load(pytorch_model_path, map_location=device)
    
    # The saved checkpoint contains the state_dict for the entire SACPolicy.
    # We need to extract the state_dict for the Actor, which has keys prefixed with "actor."
    policy_state_dict = checkpoint.get('policy_state_dict')
    if policy_state_dict is None:
        print("Error: 'policy_state_dict' not found in the checkpoint file.")
        return

    # Filter for actor weights and remove the 'actor.' prefix from keys.
    actor_state_dict = {k.replace('actor.', ''): v for k, v in policy_state_dict.items() if k.startswith('actor.')}
    
    if not actor_state_dict:
        print("Error: No actor weights found in the 'policy_state_dict'. Check key prefixes.")
        return
        
    actor.load_state_dict(actor_state_dict)
    actor.eval()
    print("Actor weights loaded successfully.")

    # --- ONNX Export ---
    # Wrap the actor for export
    export_model = ActorForExport(actor).to(device)
    export_model.eval()

    # Create a dummy input with the correct shape (batch_size, obs_dim)
    dummy_input = torch.randn(1, observation_space.shape[0], device=device)

    print(f"Exporting model to {onnx_output_path}...")
    try:
        # Create the directory for the output file if it doesn't exist
        os.makedirs(os.path.dirname(onnx_output_path), exist_ok=True)
        
        torch.onnx.export(
            export_model,
            dummy_input,
            onnx_output_path,
            verbose=False,
            input_names=["observation"],
            output_names=["action"],
            opset_version=12,
            dynamic_axes={
                "observation": {0: "batch_size"},
                "action": {0: "batch_size"},
            },
        )
        print("Model exported successfully.")
        print(f"You can now use the ONNX model at: {onnx_output_path}")
    except Exception as e:
        print(f"An error occurred during ONNX export: {e}")

if __name__ == "__main__":
    main()
