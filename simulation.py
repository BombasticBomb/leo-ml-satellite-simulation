from bsk_rl import sats, scene, obs, act
from Basilisk.utilities import macros, orbitalMotion, simIncludeGravBody
from Basilisk.simulation import exponentialAtmosphere, spacecraft
import os

#Choose a satellite profile, preferably a cubesat, and assign it mass, corss sectional area, fuel, etc
class SimulationSatellite(sats.Satellite):

    #Configuration
    mass = 12 #in kg
    inertia = [0.2, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.3] # kg*m^2
    panelArea = 0.1  # m^2 cross-sectional area for drag
    dragCoeff = 2.2  # calculate and input later

    #Initial Orbital Elements (550 km altitude, circular, 53-degree inclination)
    oe = [
        6378137.0 + 550000.0,  # Semi-major axis (meters)
        0.0001,                 # Eccentricity (near-circular)
        53.0 * macros.D2R,      # Inclination
        12.0 * macros.D2R,      # Right Ascension of the Ascending Node (RAAN)
        0.0,                    # Argument of Perigee
        0.0                     # True Anomaly
    ]

    #Observational spec for neural network
    observation_spec = [
        obs.SatProperties(dict(prop="r_BN_P", norm=orbitalMotion.REQ_EARTH * 1e3)),
        obs.SatProperties(dict(prop="v_BN_P", norm=7616.5)),
        obs.Fuel(),
    ]

    action_spec = [
        act.ContinuousThrust(n_h=3)
    ]

class RigorousEnvironment(scene.GeneralAstrodynamicsEnv):
    #Custom environment with proper astrodynamics
    def configure(self):
        super().configure()

        #Apply J2 perburtations using oblated Earth
        earth = self.grav_factory.createEarth()
        earth.isCentralBody = True

        #Path to gravitational field data
        earthPath = simIncludeGravBody.Path()
        earth.useSphericalHarmonicsGravityModel(earthPath + "/grav_data/GGM03S.txt", 2) # Degree 2 captures J2

        # 2. Attach an atmospheric density model for realistic drag calculations
        uniAtmosphere = exponentialAtmosphere.ExponentialAtmosphere()
        uniAtmosphere.ModelTag = "exponentialAtmosphere"
        uniAtmosphere.addSpacecraftToModel(self.satellites[0].scObject.scStateOutMsg)
        # Standard scale height parameters for Earth's upper atmosphere
        uniAtmosphere.baseDensity = 1.217  
        self.sim.AddModelToTask(self.task_name, uniAtmosphere)

def run_continuous_simulation(enable_vizard=True):
    # Initialize the rigorous environment with Vizard streaming enabled
    env = RigorousEnvironment(
        satellites=[RigorousEnvironment(sat_id="LEOSat_Prime")],
        viz=enable_vizard,
        sim_rate=1.0 # 1 second time step
    )

    print("Initializing rigorous LEO simulation with J2 perturbations and drag...")
    obs, info = env.reset()

    print("Running continuous simulation loop. Open Vizard to observe the perturbed orbit.")
    
    # Run an extended simulation loop (e.g., multiple orbits; 5400 seconds is ~1 orbit)
    step_count = 0
    max_steps = 100 #Train later
    
    while step_count < max_steps:
        # Sample or pass control actions (zero thrust for baseline orbital decay / J2 drift tracking)
        action = env.action_space.sample() 
        obs, reward, terminated, truncated, info = env.step(action)
        
        step_count += 1
        if terminated or truncated:
            print("Environment reset triggered. Restarting episode...")
            obs, info = env.reset()

    print("Continuous simulation sequence completed.")

if __name__ == "__main__":
    run_continuous_simulation(enable_vizard=True)