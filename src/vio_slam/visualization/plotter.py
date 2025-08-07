"""
Trajectory visualization and plotting utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class TrajectoryPlotter:
    """
    Utility class for visualizing SLAM trajectories and results.
    """
    
    def __init__(self, figsize: tuple = (10, 8)):
        """
        Initialize trajectory plotter.
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        logger.debug("Initialized trajectory plotter")
    
    def plot_trajectory(self, trajectory: np.ndarray, 
                       ground_truth: Optional[np.ndarray] = None,
                       title: str = "SLAM Trajectory",
                       save_path: Optional[str] = None,
                       show_plot: bool = True):
        """
        Plot 2D trajectory.
        
        Args:
            trajectory: Array of shape (N, 3) with [x, y, z] positions
            ground_truth: Optional ground truth trajectory
            title: Plot title
            save_path: Optional path to save plot
            show_plot: Whether to display plot
        """
        if len(trajectory) == 0:
            logger.warning("Empty trajectory provided")
            return
        
        plt.figure(figsize=self.figsize)
        
        # Plot main trajectory
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', 
                linewidth=2, label='SLAM Trajectory')
        plt.scatter(trajectory[0, 0], trajectory[0, 1], 
                   c='green', s=100, marker='o', label='Start', zorder=5)
        plt.scatter(trajectory[-1, 0], trajectory[-1, 1], 
                   c='red', s=100, marker='s', label='End', zorder=5)
        
        # Plot ground truth if available
        if ground_truth is not None and len(ground_truth) > 0:
            plt.plot(ground_truth[:, 0], ground_truth[:, 1], 'r--',
                    linewidth=1, alpha=0.7, label='Ground Truth')
        
        plt.title(title)
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Trajectory plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_trajectory_3d(self, trajectory: np.ndarray,
                          ground_truth: Optional[np.ndarray] = None,
                          title: str = "3D SLAM Trajectory",
                          save_path: Optional[str] = None,
                          show_plot: bool = True):
        """
        Plot 3D trajectory.
        
        Args:
            trajectory: Array of shape (N, 3) with [x, y, z] positions
            ground_truth: Optional ground truth trajectory
            title: Plot title
            save_path: Optional path to save plot
            show_plot: Whether to display plot
        """
        if len(trajectory) == 0:
            logger.warning("Empty trajectory provided")
            return
        
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot main trajectory
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
               'b-', linewidth=2, label='SLAM Trajectory')
        ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2],
                  c='green', s=100, marker='o', label='Start')
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2],
                  c='red', s=100, marker='s', label='End')
        
        # Plot ground truth if available
        if ground_truth is not None and len(ground_truth) > 0:
            ax.plot(ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2],
                   'r--', linewidth=1, alpha=0.7, label='Ground Truth')
        
        ax.set_title(title)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"3D trajectory plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_trajectory_comparison(self, trajectories: dict,
                                  title: str = "Trajectory Comparison",
                                  save_path: Optional[str] = None,
                                  show_plot: bool = True):
        """
        Plot multiple trajectories for comparison.
        
        Args:
            trajectories: Dictionary with {name: trajectory_array} pairs
            title: Plot title
            save_path: Optional path to save plot
            show_plot: Whether to display plot
        """
        plt.figure(figsize=self.figsize)
        
        colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k']
        linestyles = ['-', '--', '-.', ':']
        
        for i, (name, traj) in enumerate(trajectories.items()):
            if len(traj) == 0:
                continue
                
            color = colors[i % len(colors)]
            linestyle = linestyles[i % len(linestyles)]
            
            plt.plot(traj[:, 0], traj[:, 1], 
                    color=color, linestyle=linestyle,
                    linewidth=2, label=name)
            
            # Mark start and end points
            plt.scatter(traj[0, 0], traj[0, 1], 
                       c=color, s=50, marker='o', alpha=0.7)
            plt.scatter(traj[-1, 0], traj[-1, 1], 
                       c=color, s=50, marker='s', alpha=0.7)
        
        plt.title(title)
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_error_statistics(self, errors: np.ndarray,
                             error_type: str = "Translation Error",
                             save_path: Optional[str] = None,
                             show_plot: bool = True):
        """
        Plot error statistics.
        
        Args:
            errors: Array of error values
            error_type: Type of error being plotted
            save_path: Optional path to save plot
            show_plot: Whether to display plot
        """
        if len(errors) == 0:
            logger.warning("No error data provided")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Time series plot
        ax1.plot(errors, 'b-', linewidth=1)
        ax1.set_title(f'{error_type} vs Time')
        ax1.set_xlabel('Frame Index')
        ax1.set_ylabel(f'{error_type} (m)')
        ax1.grid(True, alpha=0.3)
        
        # Histogram
        ax2.hist(errors, bins=30, alpha=0.7, edgecolor='black')
        ax2.set_title(f'{error_type} Distribution')
        ax2.set_xlabel(f'{error_type} (m)')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        max_error = np.max(errors)
        
        stats_text = f'Mean: {mean_error:.3f}m\nStd: {std_error:.3f}m\nMax: {max_error:.3f}m'
        ax2.text(0.7, 0.9, stats_text, transform=ax2.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                verticalalignment='top')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Error statistics plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()