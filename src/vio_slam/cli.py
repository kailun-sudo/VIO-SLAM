#!/usr/bin/env python3
"""
Command line interface for VIO-SLAM.
"""

import argparse
import os
import sys
import logging
from pathlib import Path

from .slam_pipeline import SLAMPipeline
from .utils.config import load_config, get_default_config


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='VIO-SLAM: Visual-Inertial Odometry SLAM',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input arguments
    parser.add_argument(
        '--data_path', '-d', 
        type=str, required=True,
        help='Path to dataset directory (e.g., data/mav0)'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str, default=None,
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--dataset_type', '-t',
        type=str, default='euroc',
        choices=['euroc', 'tum', 'kitti'],
        help='Dataset type'
    )
    
    # Output arguments
    parser.add_argument(
        '--output', '-o',
        type=str, default='results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--save_trajectory',
        action='store_true',
        help='Save trajectory to file'
    )
    
    parser.add_argument(
        '--save_plots',
        action='store_true',
        help='Save visualization plots'
    )
    
    # Processing arguments
    parser.add_argument(
        '--downsample', '-s',
        type=int, default=None,
        help='Downsample factor (process every Nth frame)'
    )
    
    parser.add_argument(
        '--window_size', '-w',
        type=int, default=None,
        help='Sliding window size'
    )
    
    parser.add_argument(
        '--orb_features', '-f',
        type=int, default=None,
        help='Number of ORB features to detect'
    )
    
    # Control arguments
    parser.add_argument(
        '--no_loop_closure',
        action='store_true',
        help='Disable loop closure detection'
    )
    
    parser.add_argument(
        '--no_visualization',
        action='store_true',
        help='Disable visualization'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Validate input path
    if not os.path.exists(args.data_path):
        logger.error(f"Data path does not exist: {args.data_path}")
        sys.exit(1)
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        config = get_default_config()
        logger.info("Using default configuration")
    
    # Override config with command line arguments
    if args.downsample is not None:
        config['dataset']['downsample_factor'] = args.downsample
    
    if args.window_size is not None:
        config['slam']['window_size'] = args.window_size
        
    if args.orb_features is not None:
        config['slam']['orb_features'] = args.orb_features
    
    if args.no_loop_closure:
        config['slam']['loop_closure']['enabled'] = False
    
    if args.no_visualization:
        config['visualization']['show_trajectory'] = False
        config['visualization']['show_features'] = False
    
    if args.save_plots:
        config['visualization']['save_plots'] = True
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    try:
        # Initialize SLAM pipeline
        logger.info("Initializing SLAM pipeline...")
        slam = SLAMPipeline(config=config)
        
        # Load dataset
        logger.info(f"Loading {args.dataset_type} dataset from {args.data_path}")
        slam.load_dataset(args.data_path, args.dataset_type)
        
        # Run SLAM
        logger.info("Running SLAM pipeline...")
        trajectory = slam.run()
        
        # Save results
        if args.save_trajectory or config.get('output', {}).get('save_trajectory', False):
            trajectory_path = os.path.join(args.output, 'trajectory.pkl')
            slam.save_results(trajectory_path)
            
            # Also save as text file
            txt_path = os.path.join(args.output, 'trajectory.txt')
            import numpy as np
            np.savetxt(txt_path, trajectory, fmt='%.6f', 
                      header='x y z', comments='# ')
            logger.info(f"Trajectory saved to {txt_path}")
        
        # Save plots
        if args.save_plots or config['visualization']['save_plots']:
            plot_path = os.path.join(args.output, 'trajectory.png')
            slam.visualize_trajectory(save_path=plot_path)
        
        # Print statistics
        stats = slam.get_statistics()
        logger.info("SLAM Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        logger.info("SLAM processing completed successfully!")
        
    except Exception as e:
        logger.error(f"SLAM processing failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()