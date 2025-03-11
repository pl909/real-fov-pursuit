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

DEFAULT_GUI = False
DEFAULT_RECORD_VIDEO = True
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
    
    def __init__(self, env, agent_idx=0):
        """Initialize the wrapper.
        
        Parameters
        ----------
        env : gym.Env
            The environment to wrap.
        agent_idx : int
            The index of the agent to extract (0=pursuer, 1=evader).
        """
        super().__init__(env)
        self.agent_idx = agent_idx
        
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
    
    def make_env_pursuer():
        env = MultiPursuitAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)
        return PursuitEvasionWrapper(env, agent_idx=0)
    
    def make_env_evader():
        env = MultiPursuitAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)
        return PursuitEvasionWrapper(env, agent_idx=1)
    
    # Create vectorized environments
    pursuer_env = make_vec_env(make_env_pursuer, n_envs=1, seed=0)
    evader_env = make_vec_env(make_env_evader, n_envs=1, seed=1)
    
    # Create evaluation environments
    eval_env_pursuer = PursuitEvasionWrapper(MultiPursuitAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT), agent_idx=0)
    eval_env_evader = PursuitEvasionWrapper(MultiPursuitAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT), agent_idx=1)
    
    # Check the environments' spaces
    print('[INFO] Pursuer action space:', pursuer_env.action_space)
    print('[INFO] Pursuer observation space:', pursuer_env.observation_space)
    print('[INFO] Evader action space:', evader_env.action_space)
    print('[INFO] Evader observation space:', evader_env.observation_space)
    
    # Create models for pursuer and evader
    pursuer_model = PPO('MlpPolicy', 
                       pursuer_env, 
                       verbose=1,
                       learning_rate=0.0003,       # Lower from 0.0005
                       n_steps=2048,
                       batch_size=64,
                       n_epochs=10,                # Reduce from 15
                       gamma=0.99,
                       ent_coef=0.01,
                       clip_range=0.15,            # Reduce from default 0.2
                       device=DEVICE,
                       policy_kwargs=dict(
                           net_arch=[dict(pi=[128, 128], vf=[128, 128])]
                       ))
    
    evader_model = PPO('MlpPolicy', 
                      evader_env, 
                      verbose=1,
                      learning_rate=0.0003,
                      n_steps=2048,
                      batch_size=64,
                      n_epochs=10,
                      gamma=0.99,
                      device=DEVICE)              # Use GPU if available
    
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
    mean_reward_pursuer, _ = evaluate_policy(pursuer_model, eval_env_pursuer, n_eval_episodes=10)
    print(f"Mean pursuer reward: {mean_reward_pursuer}")
    
    print("\n[INFO] Evaluating evader model...\n")
    mean_reward_evader, _ = evaluate_policy(evader_model, eval_env_evader, n_eval_episodes=10)
    print(f"Mean evader reward: {mean_reward_evader}")
    
    # Create environment for visualization with trained policies
    test_env = MultiPursuitAviary(gui=gui, obs=DEFAULT_OBS, act=DEFAULT_ACT, record=record_video)
    logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ), num_drones=2, output_folder=output_folder, colab=colab)
    
    # Run visualization
    obs, info = test_env.reset(seed=42, options={})
    start = time.time()
    
    print("\n[INFO] Starting visualization with trained policies...\n")
    for i in range((test_env.EPISODE_LEN_SEC+2)*test_env.CTRL_FREQ):
        # Get actions from both policies
        pursuer_action, _ = pursuer_model.predict(obs[0], deterministic=True)
        evader_action, _ = evader_model.predict(obs[1], deterministic=True)
        
        # Combine actions and step the environment
        action = np.array([pursuer_action, evader_action])
        obs, reward, terminated, truncated, info = test_env.step(action)
        
        # Log data
        if DEFAULT_OBS == ObservationType.KIN:
            for d in range(2):
                logger.log(drone=d,
                           timestamp=i/test_env.CTRL_FREQ,
                           state=np.hstack([obs[d][0:3],
                                            np.zeros(4),
                                            obs[d][3:12],
                                            action[d]]),
                           control=np.zeros(12))
        
        # Render
        test_env.render()
        
        # Print some information
        distance = np.linalg.norm(test_env.pos[0] - test_env.pos[1])
        print(f"Step {i}: Distance: {distance:.2f}, Pursuer reward: {reward[0]:.2f}, Evader reward: {reward[1]:.2f}")
        
        # Maintain the desired frequency
        sync(i, start, test_env.CTRL_TIMESTEP)
        
        # Reset if episode termination
        if terminated[0] or truncated[0]:  # If any agent terminates
            print("\n[INFO] Episode terminated, resetting...\n")
            obs, info = test_env.reset(seed=42, options={})
    
    # Close the environment
    test_env.close()
    
    # Plot if requested
    if plot and DEFAULT_OBS == ObservationType.KIN:
        logger.plot()

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
