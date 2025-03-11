import numpy as np

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class MultiPursuitAviary(BaseRLAviary):
    """Multi-agent RL environment: Pursuit-Evasion scenario."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM
                 ):
        """Initialization of a multi-agent RL environment for pursuit-evasion.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        # Always have 2 drones - pursuer (idx 0) and evader (idx 1)
        self.EPISODE_LEN_SEC = 15
        
        # Set default initial positions if not provided
        if initial_xyzs is None:
            initial_xyzs = np.array([
                [0.0, 0.0, 0.5],    # Pursuer starting position
                [3.0, 3.0, 0.5]     # Evader starting position
            ])
        
        # Fixed value for number of drones in this environment
        super().__init__(drone_model=drone_model,
                         num_drones=2,  # Always 2 drones for pursuer and evader
                         neighbourhood_radius=np.inf,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record, 
                         obs=obs,
                         act=act
                         )
        # Define indices for clarity
        self.PURSUER_IDX = 0
        self.EVADER_IDX = 1
        
        # Define max distance for normalization and reward scaling
        self.MAX_DIST = 10.0  # Maximum expected distance between drones
        
        # Arena limits (for truncation)
        self.ARENA_SIZE = 5.0  # 5m x 5m arena

    ################################################################################
    
    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        ndarray
            Observation for each drone based on the parent class method,
            with added relative position information.
        """
        # Get basic observation from parent class
        obs = super()._computeObs()
        
        # Extract positions for both drones
        pursuer_pos = np.array(self.pos[self.PURSUER_IDX])
        evader_pos = np.array(self.pos[self.EVADER_IDX])
        
        # Calculate relative positions
        relative_pos_pursuer = evader_pos - pursuer_pos  # What pursuer sees (vector pointing to evader)
        relative_pos_evader = pursuer_pos - evader_pos  # What evader sees (vector pointing to pursuer)
        
        # Calculate distance between drones
        distance = np.linalg.norm(relative_pos_pursuer)
        
        # Add relative position and distance to observation
        if self.OBS_TYPE == ObservationType.KIN:
            # For the pursuer (idx 0): Add relative position to evader
            extended_obs_pursuer = np.hstack([obs[self.PURSUER_IDX], relative_pos_pursuer, [distance]])
            
            # For the evader (idx 1): Add relative position to pursuer
            extended_obs_evader = np.hstack([obs[self.EVADER_IDX], relative_pos_evader, [distance]])
            
            # Return the extended observations for both drones
            return np.array([extended_obs_pursuer, extended_obs_evader])
        else:
            # For other observation types (like RGB), return the parent observation
            return obs

    ################################################################################
    
    def _computeReward(self):
        """Computes the current reward value for each drone.

        Returns
        -------
        dict
            Rewards for each drone. Pursuer gets higher reward for being closer to evader,
            while evader gets higher reward for being further from pursuer.
        """
        # Get positions for both drones
        pursuer_pos = self.pos[self.PURSUER_IDX]
        evader_pos = self.pos[self.EVADER_IDX]
        
        # Calculate the Euclidean distance between drones
        distance = np.linalg.norm(pursuer_pos - evader_pos)
        
        # Get previous distance (from last step) - store this in self at each step
        prev_distance = getattr(self, 'prev_distance', distance)
        self.prev_distance = distance
        
        # Calculate closing velocity (positive when pursuer is getting closer)
        closing_velocity = prev_distance - distance
        
        # Normalize distance to [0, 1] range using MAX_DIST
        normalized_dist = min(distance / self.MAX_DIST, 1.0)
        
        # Compute rewards with shaping
        # Pursuer gets reward for getting closer AND for closing velocity
        pursuer_reward = 1.0 - normalized_dist + 0.3 * max(0, closing_velocity)
        
        # Evader reward remains the same
        evader_reward = normalized_dist
        
        return {
            self.PURSUER_IDX: pursuer_reward,
            self.EVADER_IDX: evader_reward
        }

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current terminated value.

        Returns
        -------
        dict
            Whether the episode is terminated for each drone.
            Episode ends if pursuer catches evader (within catch_radius).
        """
        # Get positions for both drones
        pursuer_pos = self.pos[self.PURSUER_IDX]
        evader_pos = self.pos[self.EVADER_IDX]
        
        # Calculate the Euclidean distance between drones
        distance = np.linalg.norm(pursuer_pos - evader_pos)
        
        # Define catching radius - if pursuer gets this close, it catches the evader
        catch_radius = 0.3  # 30cm
        
        # Terminate if pursuer catches evader
        caught = distance < catch_radius
        
        # Both agents terminate together
        return {
            self.PURSUER_IDX: caught,
            self.EVADER_IDX: caught
        }

    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        dict
            Whether the episode timed out or drones went out of bounds.
        """
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        
        # Check if any drone is out of bounds
        out_of_bounds = False
        for i in range(self.NUM_DRONES):
            if (abs(states[i][0]) > self.ARENA_SIZE or  # X-axis bounds
                abs(states[i][1]) > self.ARENA_SIZE or  # Y-axis bounds
                states[i][2] > 3.0 or                   # Z-axis upper bound
                states[i][2] < 0.1 or                   # Z-axis lower bound
                abs(states[i][7]) > 0.7 or              # Roll angle bounds
                abs(states[i][8]) > 0.7                 # Pitch angle bounds
               ):
                out_of_bounds = True
        
        # Check for timeout
        timeout = self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC
        
        # Both agents truncate together
        truncated = out_of_bounds or timeout
        return {
            self.PURSUER_IDX: truncated,
            self.EVADER_IDX: truncated
        }

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Returns
        -------
        dict
            Additional information about the pursuit-evasion scenario.
        """
        # Get positions for both drones
        pursuer_pos = self.pos[self.PURSUER_IDX]
        evader_pos = self.pos[self.EVADER_IDX]
        
        # Calculate the Euclidean distance between drones
        distance = np.linalg.norm(pursuer_pos - evader_pos)
        
        return {
            "pursuer_position": pursuer_pos,
            "evader_position": evader_pos,
            "distance": distance,
            "time": self.step_counter/self.PYB_FREQ
        } 