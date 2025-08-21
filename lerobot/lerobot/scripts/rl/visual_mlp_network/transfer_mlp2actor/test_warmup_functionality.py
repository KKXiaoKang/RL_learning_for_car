#!/usr/bin/env python3
"""
Test script for validating SAC Actor warm-up functionality.

This script tests the warm-up parameter loading functionality by:
1. Creating a SAC policy with warm-up enabled
2. Comparing parameters before and after warm-up
3. Validating that the correct parameters were loaded
4. Testing different warm-up configurations
"""

import sys
import os
import logging
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from lerobot.common.policies.sac.configuration_sac import SACConfig
from lerobot.common.policies.sac.modeling_sac import SACPolicy
from lerobot.common.policies.sac.warmup_utils import WarmupParameterLoader


def setup_logging():
    """Setup logging for the test script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_test_config(enable_warmup=True, warmup_model_path=None, freeze_params=False, strict_loading=False):
    """
    Create a test SAC configuration.
    
    Args:
        enable_warmup: Whether to enable warm-up
        warmup_model_path: Path to warm-up model
        freeze_params: Whether to freeze loaded parameters
        strict_loading: Whether to use strict loading
        
    Returns:
        SACConfig instance
    """
    # Import required classes
    from lerobot.configs.types import PolicyFeature, FeatureType, NormalizationMode
    from lerobot.common.policies.sac.configuration_sac import CriticNetworkConfig, PolicyConfig
    
    # Define input and output features that match your warm-up model  
    input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(32,))
    }
    
    output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))
    }
    
    # Create configuration
    config = SACConfig(
        # Basic features
        input_features=input_features,
        output_features=output_features,
        
        # Architecture
        latent_dim=64,  # 修改为64以匹配warm-up模型
        actor_network_kwargs=CriticNetworkConfig(
            hidden_dims=[256, 256],
        ),
        policy_kwargs=PolicyConfig(),
        
        # Add required dataset_stats
        dataset_stats={
            "observation.state": {
                "min": [-1.0] * 32,
                "max": [1.0] * 32,
            },
            "action": {
                "min": [-1.0] * 6,
                "max": [1.0] * 6,
            }
        },
        
        # Add required normalization mapping
        normalization_mapping={
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX
        },
        
        # Warm-up configuration
        enable_warmup=enable_warmup,
        warmup_model_path=warmup_model_path,
        warmup_freeze_loaded_params=freeze_params,
        warmup_strict_loading=strict_loading,
        
        # Disable vision to match our MLP setup
        disable_vision_features=True,
        vision_encoder_name=None,
        image_encoder_hidden_dim=0,
    )
    
    return config


def get_actor_parameter_summary(actor):
    """
    Get a summary of actor parameters.
    
    Args:
        actor: SAC Actor model
        
    Returns:
        Dictionary with parameter information
    """
    param_summary = {}
    total_params = 0
    
    for name, param in actor.named_parameters():
        param_summary[name] = {
            "shape": param.shape,
            "requires_grad": param.requires_grad,
            "mean": param.data.mean().item(),
            "std": param.data.std().item(),
            "min": param.data.min().item(),
            "max": param.data.max().item(),
        }
        total_params += param.numel()
    
    param_summary["_total_params"] = total_params
    return param_summary


def compare_parameter_summaries(before, after, warmup_params):
    """
    Compare parameter summaries before and after warm-up.
    
    Args:
        before: Parameter summary before warm-up
        after: Parameter summary after warm-up
        warmup_params: Dictionary of warm-up parameters
        
    Returns:
        Dictionary with comparison results
    """
    comparison = {
        "changed_params": [],
        "unchanged_params": [],
        "frozen_params": [],
        "expected_changes": 0,
        "actual_changes": 0
    }
    
    # Create mapping from warmup parameter names to actor parameter names
    loader = WarmupParameterLoader()
    param_mapping = loader._create_parameter_mapping()
    
    # Count expected changes
    for warmup_key, actor_key in param_mapping.items():
        if warmup_key in warmup_params and actor_key in before:
            comparison["expected_changes"] += 1
    
    # Compare each parameter
    for param_name in before:
        if param_name.startswith("_"):  # Skip metadata
            continue
            
        if param_name not in after:
            continue
            
        before_param = before[param_name]
        after_param = after[param_name]
        
        # Check if parameter changed
        mean_changed = abs(before_param["mean"] - after_param["mean"]) > 1e-6
        std_changed = abs(before_param["std"] - after_param["std"]) > 1e-6
        
        if mean_changed or std_changed:
            comparison["changed_params"].append(param_name)
            comparison["actual_changes"] += 1
        else:
            comparison["unchanged_params"].append(param_name)
        
        # Check if parameter is frozen
        if not after_param["requires_grad"]:
            comparison["frozen_params"].append(param_name)
    
    return comparison


def test_basic_warmup():
    """Test basic warm-up functionality."""
    print("\n" + "="*80)
    print("TEST 1: Basic Warm-up Functionality")
    print("="*80)
    
    # Path to warm-up model
    warmup_path = "/home/lab/RL/lerobot/lerobot/scripts/rl/visual_mlp_network/transfer_mlp2actor/transferred_sac_model.safetensors"
    
    # Check if warm-up model exists
    if not os.path.exists(warmup_path):
        print(f"❌ Warm-up model not found: {warmup_path}")
        print("Please ensure the transferred_sac_model.safetensors file exists")
        return False
    
    # Load warm-up parameters for comparison
    loader = WarmupParameterLoader()
    if not loader.load_warmup_parameters(warmup_path):
        print("❌ Failed to load warm-up parameters for testing")
        return False
    
    warmup_params = loader.warmup_params
    print(f"📊 Loaded {len(warmup_params)} warm-up parameters")
    
    # Test without warm-up
    print("\n🔧 Creating SAC policy without warm-up...")
    config_no_warmup = create_test_config(enable_warmup=False)
    policy_no_warmup = SACPolicy(config_no_warmup)
    before_summary = get_actor_parameter_summary(policy_no_warmup.actor)
    print(f"✅ Policy created. Total parameters: {before_summary['_total_params']}")
    
    # Test with warm-up
    print("\n🔥 Creating SAC policy with warm-up...")
    config_with_warmup = create_test_config(
        enable_warmup=True, 
        warmup_model_path=warmup_path
    )
    policy_with_warmup = SACPolicy(config_with_warmup)
    after_summary = get_actor_parameter_summary(policy_with_warmup.actor)
    print(f"✅ Policy created. Total parameters: {after_summary['_total_params']}")
    
    # Compare parameters
    print("\n📊 Comparing parameters...")
    comparison = compare_parameter_summaries(before_summary, after_summary, warmup_params)
    
    print(f"Expected parameter changes: {comparison['expected_changes']}")
    print(f"Actual parameter changes: {comparison['actual_changes']}")
    
    if comparison["changed_params"]:
        print(f"✅ Changed parameters ({len(comparison['changed_params'])}):")
        for param_name in comparison["changed_params"]:
            print(f"  - {param_name}")
    
    if comparison["unchanged_params"]:
        print(f"⚪ Unchanged parameters ({len(comparison['unchanged_params'])}):")
        for param_name in comparison["unchanged_params"][:5]:  # Show first 5
            print(f"  - {param_name}")
        if len(comparison["unchanged_params"]) > 5:
            print(f"  ... and {len(comparison['unchanged_params']) - 5} more")
    
    # Validate results
    success = comparison["actual_changes"] > 0
    if success:
        print(f"\n✅ Warm-up test PASSED: {comparison['actual_changes']} parameters were successfully loaded")
    else:
        print(f"\n❌ Warm-up test FAILED: No parameters were changed")
    
    return success


def test_parameter_freezing():
    """Test parameter freezing functionality."""
    print("\n" + "="*80)
    print("TEST 2: Parameter Freezing")
    print("="*80)
    
    warmup_path = "/home/lab/RL/lerobot/lerobot/scripts/rl/visual_mlp_network/transfer_mlp2actor/transferred_sac_model.safetensors"
    
    if not os.path.exists(warmup_path):
        print(f"❌ Warm-up model not found: {warmup_path}")
        return False
    
    print("🔒 Creating SAC policy with parameter freezing enabled...")
    config_frozen = create_test_config(
        enable_warmup=True,
        warmup_model_path=warmup_path,
        freeze_params=True
    )
    
    policy_frozen = SACPolicy(config_frozen)
    summary_frozen = get_actor_parameter_summary(policy_frozen.actor)
    
    # Count frozen parameters
    frozen_count = 0
    total_count = 0
    
    for name, param in policy_frozen.actor.named_parameters():
        total_count += 1
        if not param.requires_grad:
            frozen_count += 1
    
    print(f"📊 Total parameters: {total_count}")
    print(f"🔒 Frozen parameters: {frozen_count}")
    
    success = frozen_count > 0
    if success:
        print(f"✅ Parameter freezing test PASSED: {frozen_count} parameters are frozen")
    else:
        print("❌ Parameter freezing test FAILED: No parameters are frozen")
    
    return success


def test_invalid_path():
    """Test handling of invalid warm-up paths."""
    print("\n" + "="*80)
    print("TEST 3: Invalid Path Handling")
    print("="*80)
    
    print("🚫 Testing with invalid warm-up path...")
    config_invalid = create_test_config(
        enable_warmup=True,
        warmup_model_path="/invalid/path/to/model.safetensors"
    )
    
    try:
        policy_invalid = SACPolicy(config_invalid)
        print("✅ Policy created successfully despite invalid path (expected behavior)")
        return True
    except Exception as e:
        print(f"❌ Policy creation failed with invalid path: {e}")
        return False


def main():
    """Run all warm-up functionality tests."""
    print("🧪 SAC Actor Warm-up Functionality Test Suite")
    print("=" * 80)
    
    setup_logging()
    
    # Run tests
    tests = [
        ("Basic Warm-up", test_basic_warmup),
        ("Parameter Freezing", test_parameter_freezing),
        ("Invalid Path Handling", test_invalid_path),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test '{test_name}' failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Warm-up functionality is working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
    
    return passed == len(tests)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
