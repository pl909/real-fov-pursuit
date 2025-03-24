#!/usr/bin/env python3

"""Visualize trained pursuit-evasion models."""

import os
import time
import argparse
import numpy as np
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.utils.utils import sync
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.BaseAviary import ObservationType
from gym_pybullet_drones.envs.MultiPursuitAviary import MultiPursuitAviary

# Default arguments
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

# Specify the path to your models
DEFAULT_MODELS_PATH = "results/pursuit-evasion-03.24.2025_12.20.23"

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Visualize pursuit-evasion trained models')
    parser.add_argument('--gui', default=DEFAULT_GUI, type=bool, 
                        help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video', default=DEFAULT_RECORD_VIDEO, type=bool, 
                        help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot', default=DEFAULT_PLOT, type=bool, 
                        help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str, 
                        help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--models_path', default=DEFAULT_MODELS_PATH, type=str, 
                        help='Path to trained models folder (default: "results/pursuit-evasion...")', metavar='')
    parser.add_argument('--debug', default=DEFAULT_USER_DEBUG_GUI, type=bool,
                        help='Whether to use PyBullet debug GUI (default: False)', metavar='')
    ARGS = parser.parse_args()

    # Load the trained models
    print(f"\n[INFO] Loading trained models from '{ARGS.models_path}'...\n")
    
    pursuer_model_path = os.path.join(ARGS.models_path, "pursuer_model.zip")
    evader_model_path = os.path.join(ARGS.models_path, "evader_model.zip")
    
    if not os.path.exists(pursuer_model_path) or not os.path.exists(evader_model_path):
        raise FileNotFoundError(f"Could not find model files in {ARGS.models_path}")
    
    # Load models
    pursuer_model = PPO.load(pursuer_model_path)
    evader_model = PPO.load(evader_model_path)
    
    print("[INFO] Models loaded successfully!")
    
    # Create environment for visualization
    env = MultiPursuitAviary(
        gui=ARGS.gui,
        obs=ObservationType.KIN,
        user_debug_gui=ARGS.debug,
        record=ARGS.record_video
    )
    
    # Setup logger
    logger = Logger(
        logging_freq_hz=int(env.CTRL_FREQ),
        num_drones=2,
        output_folder=ARGS.output_folder,
        colab=DEFAULT_COLAB
    )
    
    # Run visualization
    obs, info = env.reset(seed=42, options={})
    start = time.time()
    
    print("\n[INFO] Starting visualization with trained policies...\n")
    print("Press 'h' in PyBullet GUI for keyboard shortcuts")
    
    episode_counter = 0
    total_steps = 0
    capture_count = 0
    
    try:
        for i in range(10000):  # Large number for continuous visualization
            # Get actions from both policies
            pursuer_action, _ = pursuer_model.predict(obs[0], deterministic=True)
            evader_action, _ = evader_model.predict(obs[1], deterministic=True)
            
            # Combine actions and step the environment
            action = np.array([pursuer_action, evader_action])
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Log data
            for d in range(2):
                logger.log(drone=d,
                           timestamp=i/env.CTRL_FREQ,
                           state=np.hstack([
                               env.pos[d], 
                               np.zeros(4),  # Placeholder for quaternions
                               env.vel[d],
                               env.rpy[d],
                               action[d]
                           ]),
                           control=np.zeros(12))
            
            # Calculate distance between drones
            distance = np.linalg.norm(env.pos[0] - env.pos[1])
            
            # Count captures (when distance is very close)
            if distance < 0.3:
                capture_count += 1
            
            # Display info
            total_steps += 1
            if i % 10 == 0:  # Every 10 steps
                print(f"Step {i} | Episode {episode_counter} | Distance: {distance:.2f}m | "
                      f"Pursuer reward: {reward[0]:.2f} | Evader reward: {reward[1]:.2f}")
            
            # Maintain the desired frequency
            sync(i, start, env.CTRL_TIMESTEP)
            
            # Reset if episode terminates
            if terminated[0] or truncated[0]:  # If any agent terminates
                print(f"\n[INFO] Episode {episode_counter} ended after {i - episode_counter * env.EPISODE_LEN_SEC * env.CTRL_FREQ} steps")
                print(f"Pursuer reward: {reward[0]:.2f} | Evader reward: {reward[1]:.2f}")
                
                # Reset environment
                obs, info = env.reset(seed=np.random.randint(0, 10000), options={})
                episode_counter += 1
                
                print(f"\n[INFO] Starting episode {episode_counter}...\n")
    
    except KeyboardInterrupt:
        print("\n[INFO] Simulation interrupted by user")
    
    finally:
        # Close the environment
        env.close()
        
        # Print overall statistics
        print("\n[INFO] Simulation statistics:")
        print(f"Total episodes: {episode_counter}")
        print(f"Total steps: {total_steps}")
        print(f"Capture events: {capture_count}")
        
        # Plot if requested
        if ARGS.plot:
            logger.plot() 