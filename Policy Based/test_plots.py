"""
Simple test script to verify plot saving functionality.
Run this to test if plots are being saved correctly.
"""

import numpy as np
import os
import sys

# Add the parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.visualization import plot_training_curves, plot_comparative_analysis

def test_plot_saving():
    """Test if plots are saved correctly."""
    
    # Create dummy reward data
    np.random.seed(42)
    episodes = 100
    
    rewards_dict = {
        'Actor-Critic': np.cumsum(np.random.randn(episodes) * 0.5 + 0.1).tolist(),
        'REINFORCE': np.cumsum(np.random.randn(episodes) * 0.5 + 0.05).tolist(),
        'REINFORCE+Baseline': np.cumsum(np.random.randn(episodes) * 0.5 + 0.08).tolist()
    }
    
    # Configuration
    config = {
        'figsize': (10, 5),
        'grid': True,
        'title': 'Test Training Curves',
        'smooth_window': 10
    }
    
    # Determine save directory
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
    
    plots_dir = os.path.join(script_dir, 'plots')
    
    print(f"Testing plot saving...")
    print(f"Save directory: {plots_dir}")
    print(f"Current working directory: {os.getcwd()}")
    print("-" * 50)
    
    # Test training curves
    print("\n1. Testing training curves plot...")
    plot_training_curves(rewards_dict, config, save_dir=plots_dir)
    
    # Test comparative analysis
    print("\n2. Testing comparative analysis plot...")
    plot_comparative_analysis(rewards_dict, config, save_dir=plots_dir)
    
    # Check if files exist
    print("\n" + "="*50)
    print("Checking saved files:")
    print("="*50)
    
    files_to_check = ['training_curves.png', 'comparative_analysis.png']
    for filename in files_to_check:
        filepath = os.path.join(plots_dir, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"✓ {filename}: EXISTS ({size:,} bytes)")
        else:
            print(f"✗ {filename}: NOT FOUND")
    
    print("="*50)
    print("\nTest completed!")
    print(f"Check the plots folder at: {plots_dir}")

if __name__ == '__main__':
    test_plot_saving()
