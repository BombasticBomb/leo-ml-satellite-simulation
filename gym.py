import gymnasium as gym
from gymnasium import spaces
import numpy as np
# Core Basilisk simulation architecture
from Basilisk.architecture import bskLogging
from Basilisk.simulation import spacecraft, simpleNav, simulationArchTypes
from Basilisk.utilities import SimulationBaseClass, macros, orbitalMotion

class BasiliskOrbitalAvoidanceEnv(gym.Env):
    """Custom Environment for Orbital Collision Avoidance using Basilisk"""
    metadata = {"render_modes": []}

    def __init__(self):
        super(BasiliskOrbitalAvoidanceEnv, self).__init__()
        
        # Action Space: Continuous thrust vector [Thrust_X, Thrust_Y, Thrust_Z] in Newtons
        # Limiting max thrust to 0.5N (typical for electric/chemical thrusters on smallsats)
        self.action_space = spaces.Box(low=-0.5, high=0.5, shape=(3,), dtype=np.float32)
        
        # Observation Space: Relative position and velocity [dx, dy, dz, dvx, dvy, dvz]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        
        self.sim_time = 0.0
        self.dt = 10.0  # Physics time step size in seconds
        self.max_duration = 86400.0  # 1 Day max horizon per episode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 1. Initialize Basilisk Base Engine
        self.sim = SimulationBaseClass.SimBaseClass()
        self.proc = self.sim.CreateNewProcess("DynamicsProcess")
        self.task = self.sim.CreateNewTask("DynamicsTask", macros.sec2nano(self.dt))
        self.proc.addTask(self.task)
        
        # 2. Setup Primary Spacecraft (The RL Agent)
        self.scObject = spacecraft.Spacecraft()
        self.scObject.ModelTag = "agentSat"
        self.scObject.hub.mHub = 500.0  # Mass in kg
        self.sim.AddModelToTask("DynamicsTask", self.scObject)
        
        # 3. Setup Earth Gravity Field
        self.gravFactory = simIncludeGravBody.gravBodyFactory()
        earth = self.gravFactory.createEarth()
        earth.isCentralBody = True
        self.scObject.gravField.gravBodies.append(earth)
        
        # 4. Set Initial States (Sampled from your dataset configurations)
        # Using standard LEO parameters as baseline example
        oe = orbitalMotion.ClassicElements()
        oe.a = 7000e3      # Semi-major axis (meters)
        oe.e = 0.001       # Eccentricity
        oe.i = 51.6 * macros.D2R
        rN, vN = orbitalMotion.elem2rv(earth.mu, oe)
        self.scObject.hub.r_CN_NInit = rN
        self.scObject.hub.v_CN_NInit = vN
        
        # 5. Define Secondary Object (The Threat / Debris)
        # For simulation simplicity, we track debris relative trajectories mathematically 
        # or can initialize a second Basilisk scObject tracking a colliding trajectory
        self.debris_r = rN + np.array([2000.0, 500.0, -100.0]) # 2km offset close approach
        self.debris_v = vN + np.array([0.0, 10.0, 2.0])
        
        self.sim_time = 0.0
        self.sim.InitializeSimulation()
        
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        # Extract current state vectors from the Basilisk engine structure
        agent_r = np.array(self.scObject.hub.r_CN_N)
        agent_v = np.array(self.scObject.hub.v_CN_N)
        
        # Compute relative state tensors
        rel_pos = self.debris_r - agent_r
        rel_vel = self.debris_v - agent_v
        
        return np.concatenate([rel_pos, rel_vel]).astype(np.float32)

    def step(self, action):
        # 1. Apply Action: Convert RL thrust matrix into Basilisk external force vectors
        # In a complete build, link this to Basilisk's externalForceInertial module
        thrust_force = action.tolist() 
        
        # 2. Advance Basilisk physics engine by self.dt
        self.sim_time += self.dt
        self.sim.ConfigureStopTime(macros.sec2nano(self.sim_time))
        self.sim.ExecuteSimulation()
        
        # Fake debris propagation along unperturbed path for example simplification
        self.debris_r += self.debris_v * self.dt 
        
        # 3. Calculate Rewards & Collision Penalties
        obs = self._get_obs()
        rel_distance = np.linalg.norm(obs[0:3])
        
        # Reward design: Penalize getting too close, reward preserving fuel (small actions)
        reward = 0.0
        terminated = False
        
        if rel_distance < 100.0:  # Collision threshold: 100 meters
            reward = -1000.0      # Catastrophic failure penalty
            terminated = True
        else:
            reward = 0.1 * rel_distance - 0.01 * np.linalg.norm(action) # Stay far, burn less
            
        truncated = self.sim_time >= self.max_duration
        
        return obs, reward, terminated, truncated, {}
