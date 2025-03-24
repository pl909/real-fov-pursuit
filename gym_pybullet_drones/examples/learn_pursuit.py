"""Script demonstrating the use of MultiPursuitAviary for pursuit-evasion learning.

This script trains policies for both the pursuer and evader drones in an adversarial setting.
The pursuer tries to catch the evader, while the evader tries to maintain distance.

Example
-------
In a terminal, run as:

    $ python learn_pursuit.py

Notes
-----`
This example uses stable-baselines3 to train two PPO policies:
1. A pursuer policy that tries to minimize distance to the evader
2. An evader policy that tries to maximize distance from the pursuer
"""
import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.MultiPursuitAviary import MultiPursuitAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

DEFAULT_OBS = ObservationType('kin')  # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('pid')  # 'rpm', 'pid', 'vel', 'one_d_rpm', 'one_d_pid'
DEFAULT_TRAIN_STEPS = 50000  # Lower for testing, increase for better performance

# Check for GPU availability and set PyTorch to use it
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {DEVICE}")

# Set number of threads for PyTorch based on available CPUs
if DEVICE == "cpu":
    torch.set_num_threads(os.cpu_count())
    print(f"[INFO] Set PyTorch to use {os.cpu_count()} CPU threads")

class PursuitEvasionWrapper(gym.Wrapper):
    """Wrapper to select either pursuer or evader perspective in MultiPursuitAviary.
    
    This wrapper allows training individual policies for either the pursuer or evader
    by exposing only one agent's observation and reward to the learning algorithm.
    """
    
    def __init__(self, env, agent_idx=0, curriculum_phase=0):
        """Initialize the wrapper.
        
        Parameters
        ----------
        env : gym.Env
            The environment to wrap.
        agent_idx : int
            The index of the agent to extract (0=pursuer, 1=evader).
        curriculum_phase : int
            The current phase of the curriculum.
        """
        super().__init__(env)
        self.agent_idx = agent_idx
        self.curriculum_phase = curriculum_phase
        
        # Get a sample observation to determine the actual observation size
        sample_obs, _ = env.reset()
        obs_size = sample_obs[agent_idx].shape[0]
        
        # Adjust observation and action spaces to be single-agent with correct dimensions
        if isinstance(self.observation_space, gym.spaces.Box):
            # Use the correct observation size from the sample
            low = np.full(obs_size, -np.inf)
            high = np.full(obs_size, np.inf)
            
            # Fill in the first 12 elements with the original observation space bounds
            low[:12] = self.observation_space.low[agent_idx, :12] 
            high[:12] = self.observation_space.high[agent_idx, :12]
            
            # Set the bounds for action buffer part (from 12 to obs_size-4)
            if obs_size > 16:  # if we have action buffer
                low[12:obs_size-4] = -1.0
                high[12:obs_size-4] = 1.0
                
            # Set bounds for relative position and distance (last 4 elements)
            low[obs_size-4:obs_size-1] = -np.inf  # Relative position can be any value
            high[obs_size-4:obs_size-1] = np.inf
            low[obs_size-1] = 0.0  # Distance is always positive
            high[obs_size-1] = np.inf
            
            self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
            
        if isinstance(self.action_space, gym.spaces.Box):
            low = self.action_space.low[agent_idx]
            high = self.action_space.high[agent_idx]
            self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
    
    def reset(self, **kwargs):
        """Reset the environment and return only the selected agent's observation."""
        # Adjust randomization based on curriculum phase
        random_distance = min(4.0 + self.curriculum_phase * 0.5, 8.0)
        
        # We need to pass this to the underlying environment
        if hasattr(self.env, 'random_distance'):
            self.env.random_distance = random_distance
            
        obs, info = self.env.reset(**kwargs)
        return obs[self.agent_idx], info
    
    def step(self, action):
        """Take a step with the selected agent's action.
        
        The other agent's action is sampled from a fixed or pretrained policy.
        """
        # Create a full action vector for all agents
        full_action = np.zeros((2, action.shape[0]))
        full_action[self.agent_idx] = action
        
        # For the other agent, use a simple heuristic or a pretrained policy
        if self.agent_idx == 0:  # We're controlling the pursuer, evader does simple evasion
            evader_obs = self.env._computeObs()[1]  # Get evader's observation
            # Simple evasion policy: move away from pursuer
            relative_pos = evader_obs[-4:-1]  # Relative position of pursuer from evader
            if np.linalg.norm(relative_pos) > 0:
                evasion_direction = -relative_pos / np.linalg.norm(relative_pos)
                full_action[1] = evasion_direction * 0.5  # Scale for PID control
        else:  # We're controlling the evader, pursuer does simple pursuit
            pursuer_obs = self.env._computeObs()[0]  # Get pursuer's observation
            # Simple pursuit policy: move toward evader
            relative_pos = pursuer_obs[-4:-1]  # Relative position of evader from pursuer
            if np.linalg.norm(relative_pos) > 0:
                pursuit_direction = relative_pos / np.linalg.norm(relative_pos)
                full_action[0] = pursuit_direction * 0.5  # Scale for PID control
        
        # Take the step with full actions
        obs, reward, terminated, truncated, info = self.env.step(full_action)
        
        # Debug logging for early termination
        if terminated[self.agent_idx] or truncated[self.agent_idx]:
            if self.env.step_counter < 5:  # Early termination
                states = np.array([self.env._getDroneStateVector(i) for i in range(2)])
                print(f"[DEBUG] Early termination for agent {self.agent_idx}:")
                print(f"  - Position: {self.env.pos}")
                print(f"  - Roll/Pitch: {states[:,7:9]}")
                print(f"  - Terminated: {terminated}, Truncated: {truncated}")
        
        # Return only the selected agent's perspective
        return obs[self.agent_idx], reward[self.agent_idx], terminated[self.agent_idx], truncated[self.agent_idx], info

def run(output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, plot=True, 
        colab=DEFAULT_COLAB, record_video=DEFAULT_RECORD_VIDEO, 
        train_steps=DEFAULT_TRAIN_STEPS):
    """Main training and evaluation function.
    
    Parameters
    ----------
    output_folder : str
        Folder to save results in.
    gui : bool
        Whether to use PyBullet GUI during evaluation.
    plot : bool
        Whether to generate plots after evaluation.
    colab : bool
        Whether running in Google Colab.
    record_video : bool
        Whether to record evaluation videos.
    train_steps : int
        Number of training steps.
    """
    # Create folder for results
    time_str = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    filename = os.path.join(output_folder, 'pursuit-evasion-'+time_str)
    if not os.path.exists(filename):
        os.makedirs(filename+'/')
    
    # Create environments for pursuer and evader training
    print("\n[INFO] Creating training environments...")
    
    # Override the episode length in MultiPursuitAviary for training
    # This will be applied when creating the environments below
    MultiPursuitAviary.EPISODE_LEN_SEC = 30  # Increase from default 15 to 30 seconds
    
    def make_env_pursuer():
        env = MultiPursuitAviary(
            obs=DEFAULT_OBS, 
            act=DEFAULT_ACT,
            random_init=True,
            min_distance=4.0,
            max_distance=8.0
        )
        return PursuitEvasionWrapper(env, agent_idx=0)
    
    def make_env_evader():
        env = MultiPursuitAviary(
            obs=DEFAULT_OBS, 
            act=DEFAULT_ACT,
            random_init=True,
            min_distance=4.0,
            max_distance=8.0
        )
        return PursuitEvasionWrapper(env, agent_idx=1)
    
    # Create vectorized environments
    pursuer_env = make_vec_env(make_env_pursuer, n_envs=1, seed=0)
    evader_env = make_vec_env(make_env_evader, n_envs=1, seed=1)
    
    # Add these evaluation environments after the training environments
    eval_scenarios = [
        # Scenario 1: Default starting positions
        {"initial_xyzs": np.array([[0.0, 0.0, 0.5], [3.0, 3.0, 0.5]])},
        # Scenario 2: Medium distance
        {"initial_xyzs": np.array([[0.0, 0.0, 0.5], [4.0, 0.0, 0.5]])},
        # Scenario 3: Long distance
        {"initial_xyzs": np.array([[0.0, 0.0, 0.5], [0.0, 6.0, 0.5]])},
    ]

    # Create evaluation environments with consistent scenarios
    eval_envs_pursuer = []
    eval_envs_evader = []

    for scenario in eval_scenarios:
        eval_envs_pursuer.append(
            PursuitEvasionWrapper(
                MultiPursuitAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT, **scenario), 
                agent_idx=0
            )
        )
        eval_envs_evader.append(
            PursuitEvasionWrapper(
                MultiPursuitAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT, **scenario), 
                agent_idx=1
            )
        )
    
    # Check the environments' spaces
    print('[INFO] Pursuer action space:', pursuer_env.action_space)
    print('[INFO] Pursuer observation space:', pursuer_env.observation_space)
    print('[INFO] Evader action space:', evader_env.action_space)
    print('[INFO] Evader observation space:', evader_env.observation_space)
    
    # Create models for pursuer and evader
    pursuer_model = PPO('MlpPolicy', 
                       pursuer_env, 
                       verbose=1,
                       learning_rate=0.0001,      # Lower learning rate for stability
                       n_steps=1024,              # Shorter trajectories
                       batch_size=64,
                       n_epochs=5,                # Fewer epochs to avoid overfitting
                       gamma=0.99,
                       ent_coef=0.01,
                       clip_range=0.1,            # Smaller clip range for stability
                       device=DEVICE,
                       policy_kwargs=dict(
                           net_arch=dict(pi=[128, 128], vf=[128, 128])
                       ))
    
    evader_model = PPO('MlpPolicy', 
                      evader_env, 
                      verbose=1,
                      learning_rate=0.0001,
                      n_steps=1024,
                      batch_size=64,
                      n_epochs=5,
                      gamma=0.99,
                      device=DEVICE,
                      policy_kwargs=dict(
                          net_arch=dict(pi=[128, 128], vf=[128, 128])
                      ))
    
    # Replace sequential training with alternating phases
    alternating_steps = 10000  # 10K steps per phase
    phases = train_steps // alternating_steps

    for phase in range(phases):
        print(f"\n[INFO] Training phase {phase+1}/{phases}")
        
        # Train pursuer first
        print("\n[INFO] Training pursuer model...\n")
        pursuer_model.learn(total_timesteps=alternating_steps)
        
        # Then train evader against updated pursuer
        print("\n[INFO] Training evader model...\n")
        evader_model.learn(total_timesteps=alternating_steps)
    
    # Save the models
    pursuer_model.save(filename+'/pursuer_model.zip')
    evader_model.save(filename+'/evader_model.zip')
    print(f"Models saved to {filename}")
    
    # Evaluate the models
    print("\n[INFO] Evaluating pursuer model...\n")
    mean_reward_pursuer, _ = evaluate_policy(pursuer_model, eval_envs_pursuer[0], n_eval_episodes=10)
    print(f"Mean pursuer reward: {mean_reward_pursuer}")
    
    print("\n[INFO] Evaluating evader model...\n")
    mean_reward_evader, _ = evaluate_policy(evader_model, eval_envs_evader[0], n_eval_episodes=10)
    print(f"Mean evader reward: {mean_reward_evader}")

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Pursuit-evasion reinforcement learning example')
    parser.add_argument('--gui', default=DEFAULT_GUI, type=str2bool, 
                        help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video', default=DEFAULT_RECORD_VIDEO, type=str2bool, 
                        help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str, 
                        help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab', default=DEFAULT_COLAB, type=bool, 
                        help='Whether example is being run by a notebook (default: "False")', metavar='')
    parser.add_argument('--train_steps', default=DEFAULT_TRAIN_STEPS, type=int, 
                        help='Number of training steps (default: 50000)', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
