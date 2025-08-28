#!/usr/bin/env python3
"""
Utilities for loading and applying MLP BC warm-up parameters to SAC Actor networks.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from safetensors import safe_open


class WarmupParameterLoader:
    """
    Utility class for loading warm-up parameters from MLP BC models and applying them to SAC Actor networks.
    
    This class handles:
    1. Loading parameters from safetensors files
    2. Mapping MLP BC parameter names to SAC Actor parameter names
    3. Filtering compatible parameters
    4. Applying parameters to the target model
    """
    
    def __init__(self):
        self.warmup_params = {}
        self.parameter_mapping = self._create_parameter_mapping()
    
    def _create_parameter_mapping(self) -> Dict[str, str]:
        """
        Create mapping from MLP BC parameter names to SAC Actor parameter names.
        
        Returns:
            Dictionary mapping MLP BC parameter names to SAC Actor parameter names
        """
        mapping = {
            # 编码器归一化参数
            "actor.encoder.input_normalization.observation_state_min": "encoder.input_normalization.observation_state_min",
            "actor.encoder.input_normalization.observation_state_max": "encoder.input_normalization.observation_state_max",
            
            # 状态编码器参数
            "actor.encoder.state_encoder.0.weight": "encoder.state_encoder.0.weight",
            "actor.encoder.state_encoder.0.bias": "encoder.state_encoder.0.bias",
            "actor.encoder.state_encoder.1.weight": "encoder.state_encoder.1.weight",
            "actor.encoder.state_encoder.1.bias": "encoder.state_encoder.1.bias",
            
            # 主干网络参数
            "actor.network.net.0.weight": "network.net.0.weight",
            "actor.network.net.0.bias": "network.net.0.bias",
            "actor.network.net.1.weight": "network.net.1.weight",
            "actor.network.net.1.bias": "network.net.1.bias",
            "actor.network.net.3.weight": "network.net.3.weight",
            "actor.network.net.3.bias": "network.net.3.bias",
            "actor.network.net.4.weight": "network.net.4.weight",
            "actor.network.net.4.bias": "network.net.4.bias",
            
            # 输出层参数
            "actor.mean_layer.weight": "mean_layer.weight",
            "actor.mean_layer.bias": "mean_layer.bias",
            "actor.std_layer.weight": "std_layer.weight",
            "actor.std_layer.bias": "std_layer.bias",
        }
        return mapping
    
    def load_warmup_parameters(self, warmup_model_path: str) -> bool:
        """
        Load warm-up parameters from a safetensors file.
        
        Args:
            warmup_model_path: Path to the warm-up model file (.safetensors)
            
        Returns:
            True if loading succeeded, False otherwise
        """
        if not os.path.exists(warmup_model_path):
            logging.error(f"Warm-up model file not found: {warmup_model_path}")
            return False
        
        try:
            # Load parameters from safetensors file
            warmup_params = {}
            with safe_open(warmup_model_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    warmup_params[key] = f.get_tensor(key)
            
            self.warmup_params = warmup_params
            logging.info(f"Successfully loaded {len(warmup_params)} warm-up parameters from {warmup_model_path}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to load warm-up parameters: {e}")
            return False
    
    def get_compatible_parameters(self, target_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Get compatible parameters that can be transferred to the target model.
        
        Args:
            target_state_dict: State dict of the target SAC Actor model
            
        Returns:
            Dictionary of compatible parameters ready for loading
        """
        compatible_params = {}
        
        if not self.warmup_params:
            logging.warning("No warm-up parameters loaded")
            return compatible_params
        
        transfer_count = 0
        shape_mismatch_count = 0
        missing_count = 0
        
        # Check each mapping
        for warmup_key, target_key in self.parameter_mapping.items():
            if warmup_key in self.warmup_params and target_key in target_state_dict:
                warmup_param = self.warmup_params[warmup_key]
                target_param = target_state_dict[target_key]
                
                # Check shape compatibility
                if warmup_param.shape == target_param.shape:
                    compatible_params[target_key] = warmup_param.clone()
                    transfer_count += 1
                    logging.debug(f"✅ Compatible: {warmup_key} → {target_key} {warmup_param.shape}")
                else:
                    shape_mismatch_count += 1
                    logging.warning(f"❌ Shape mismatch: {warmup_key} {warmup_param.shape} vs {target_key} {target_param.shape}")
            else:
                missing_count += 1
                if warmup_key not in self.warmup_params:
                    logging.debug(f"⚠️ Missing in warm-up: {warmup_key}")
                if target_key not in target_state_dict:
                    logging.debug(f"⚠️ Missing in target: {target_key}")
        
        logging.info(f"Parameter compatibility summary:")
        logging.info(f"  ✅ Compatible parameters: {transfer_count}")
        logging.info(f"  ❌ Shape mismatches: {shape_mismatch_count}")
        logging.info(f"  ⚠️ Missing parameters: {missing_count}")
        
        return compatible_params
    
    def apply_warmup_parameters(self, target_model: torch.nn.Module, 
                              strict: bool = False, 
                              freeze_loaded_params: bool = False) -> bool:
        """
        Apply warm-up parameters to the target SAC Actor model.
        
        Args:
            target_model: Target SAC Actor model to apply parameters to
            strict: If True, require all mapped parameters to be compatible
            freeze_loaded_params: If True, freeze parameters that were loaded from warm-up
            
        Returns:
            True if application succeeded, False otherwise
        """
        try:
            # Get current state dict
            current_state_dict = target_model.state_dict()
            
            # Get compatible parameters
            compatible_params = self.get_compatible_parameters(current_state_dict)
            
            if not compatible_params:
                logging.warning("No compatible parameters found for warm-up")
                return False
            
            # Update state dict with compatible parameters
            updated_state_dict = current_state_dict.copy()
            updated_state_dict.update(compatible_params)
            
            # Load the updated state dict
            target_model.load_state_dict(updated_state_dict, strict=strict)
            
            # Optionally freeze loaded parameters
            if freeze_loaded_params:
                self._freeze_parameters(target_model, compatible_params.keys())
            
            logging.info(f"✅ Successfully applied {len(compatible_params)} warm-up parameters to target model")
            return True
            
        except Exception as e:
            logging.error(f"Failed to apply warm-up parameters: {e}")
            return False
    
    def _freeze_parameters(self, model: torch.nn.Module, param_names: list):
        """
        Freeze specific parameters in the model.
        
        Args:
            model: Target model
            param_names: List of parameter names to freeze
        """
        frozen_count = 0
        for name, param in model.named_parameters():
            if name in param_names:
                param.requires_grad = False
                frozen_count += 1
                logging.debug(f"🔒 Frozen parameter: {name}")
        
        logging.info(f"🔒 Frozen {frozen_count} warm-up parameters")
    
    def get_warmup_info(self) -> Dict[str, Any]:
        """
        Get information about loaded warm-up parameters.
        
        Returns:
            Dictionary containing warm-up parameter information
        """
        if not self.warmup_params:
            return {"loaded": False, "param_count": 0}
        
        total_params = sum(p.numel() for p in self.warmup_params.values())
        
        return {
            "loaded": True,
            "param_count": len(self.warmup_params),
            "total_parameters": total_params,
            "parameter_names": list(self.warmup_params.keys()),
            "mapping_count": len(self.parameter_mapping)
        }


def load_warmup_parameters_for_actor(actor_model: torch.nn.Module, 
                                   warmup_model_path: str,
                                   strict: bool = False,
                                   freeze_loaded_params: bool = False) -> bool:
    """
    Convenience function to load warm-up parameters for a SAC Actor model.
    
    Args:
        actor_model: SAC Actor model to load parameters into
        warmup_model_path: Path to the warm-up model file
        strict: If True, require all mapped parameters to be compatible
        freeze_loaded_params: If True, freeze parameters that were loaded from warm-up
        
    Returns:
        True if loading succeeded, False otherwise
    """
    loader = WarmupParameterLoader()
    
    # Load warm-up parameters
    if not loader.load_warmup_parameters(warmup_model_path):
        return False
    
    # Apply parameters to actor model
    return loader.apply_warmup_parameters(
        target_model=actor_model, 
        strict=strict, 
        freeze_loaded_params=freeze_loaded_params
    )


def validate_warmup_model_path(warmup_model_path: Optional[str]) -> Optional[str]:
    """
    Validate and resolve warm-up model path.
    
    Args:
        warmup_model_path: Path to warm-up model (can be relative or absolute)
        
    Returns:
        Absolute path if valid, None otherwise
    """
    if not warmup_model_path:
        return None
    
    # Convert to Path object for easier handling
    path = Path(warmup_model_path)
    
    # If relative path, assume it's relative to the project root
    if not path.is_absolute():
        # Try to find project root by looking for common markers
        current_dir = Path.cwd()
        project_markers = ['lerobot', 'pyproject.toml', 'setup.py']
        
        for marker in project_markers:
            if (current_dir / marker).exists():
                path = current_dir / warmup_model_path
                break
        else:
            # If no markers found, use current directory
            path = current_dir / warmup_model_path
    
    # Check if file exists
    if path.exists() and path.is_file():
        return str(path.absolute())
    else:
        logging.error(f"Warm-up model file not found: {path}")
        return None
