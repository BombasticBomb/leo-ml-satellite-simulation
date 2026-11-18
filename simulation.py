from typing import ClassVar

import gymnasium as gym
from Basilisk.architecture import bskLogging
from Basilisk.utilities import macros, orbitalMotion, simIncludeGravBody
from Basilisk.simulation import radiationPressure

from bsk_rl import act, data, obs, sats, scene
from bsk_rl.sim import dyn, fsw

import numpy as np
import random



bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)




#Importing means it already includes J2 and Drag
class ScientificDynamics(dyn.ContinuousImagingDynModel):
    def __init__(self, satellite, sim_rate, **kwargs):
        super().__init__(satellite, sim_rate, **kwargs)

        #Initialize Basilisk Solar Radiation Pressure
        self.srpEffector = radiationPressure.RadiationPressure()
        self.srpEffector.ModelTag = f"{satellite.name}_SRP"
        self.srpEffector.setUseCannonballModel() #Appoximate by treating it as a round ball
        self.srpEffector.area = 0.1 #Tune later
        self.srpEffector.coefficientReflection = 1.2

        # Fetch the Sun's planet state message from the simulation world environment
        sun_index = self.world.sun_index
        sun_msg = self.world.gravFactory.spiceObject.planetStateOutMsgs[sun_index]
        self.srpEffector.sunEphmInMsg.subscribeTo(sun_msg)
        
        # Attach the SRP effector to the satellite's core spacecraft physical model
        self.scObject.addDynamicEffector(self.srpEffector)



class MySatellite(sats.AccessSatellite):
    # Physical and aerodynamic properties for scientific rigor
    mass = 11.0  # kg
    inertia = [0.2, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.3]  # kg*m^2
    panelArea = 0.1  # m^2 cross-sectional area for drag
    dragCoeff = 2.2  # Aerodynamic drag coefficient

    observation_spec: ClassVar[list[obs.Observation]] = [
        obs.SatProperties(
            dict(prop="storage_level_fraction"),
            dict(prop="battery_charge_fraction"),
            dict(prop="r_BN_P", norm=orbitalMotion.REQ_EARTH * 1e3),
            dict(prop="v_BN_P", norm=7616.5),
        ),
        obs.Eclipse(),
    ]

    action_spec: ClassVar[list[act.Action]] = [
        act.Scan(duration=60.0),
        act.Charge(duration=600.0),
    ]

    dyn_type = ScientificDynamics
    fsw_type = fsw.ContinuousImagingFSWModel

    def eci_data(self):
        sc_state_payload = self.dyn.scObject.scStateOutMsg.read()
        return np.array(sc_state_payload.r_BN_N), np.array(sc_state_payload.v_BN_N)



def main():

    #Setup Satellite Configuration
    oe_setup = orbitalMotion.ClassicElements()
    oe_setup.a = 6378137.0 + 550000.0  # Semi-major axis (meters)
    oe_setup.e = 0.0001                # Eccentricity (near-circular)
    oe_setup.i = 53.0 * macros.D2R     # Inclination (radians)
    oe_setup.Omega = 12.0 * macros.D2R # RAAN (radians)
    oe_setup.omega = 0.0               # Argument of Perigee (radians)
    oe_setup.f = 0.2 * macros.D2R                  # True Anomaly (radians)

    full_sat_args = MySatellite.default_sat_args(
        oe=lambda: oe_setup
    )
    satellite = MySatellite(name="Taquee's Satellite", sat_args=full_sat_args)

    env = gym.make(
        "SatelliteTasking-v1",
        satellite=satellite,
        scenario=scene.UniformNadirScanning(),
        rewarder=data.ScanningTimeReward(),
        time_limit=5700.0,
        log_level="WARNING",
        vizard_dir="./vizard_output",
        disable_env_checker=True,

        # Injecting the Vizard core configurations
        vizard_settings=dict(
            orbitLinesOn=1,
            trueTrajectoryLinesOn=1,
            spacecraftCSon=1,
            showLocationLabels=1,
            showLocationCommLines=1,
            storageDeviceGUIon=1,
            ambient=0.4
        ),
        
    )

    try:
        observation, info = env.reset(seed=1)
        print("Environment initialized with custom satellite properties successfully.")
        print("Initial observation:", observation)

        for step in range(100):
            observation, reward, terminated, truncated, info = env.step(0)

            print(
                f"Step {step + 1}: reward={reward}, "
                f"terminated={terminated}, "
                f"truncated={truncated}"
            )
            
            #Get ECI satellite state vectors
            satellite_data = env.unwrapped.satellites[0]
            r_sat, v_sat = satellite_data.eci_data()


            if terminated or truncated:
                break

            
        

        print("Custom property test completed successfully.")

    finally:
        try:
            env.close()
        except AttributeError:
            pass


#Pass in eci 3D vectors to compute for LVLH
def convert_eci_to_lvlh(r_sat, v_sat, r_deb, v_deb):

    #Relative eci
    r_eci = r_deb - r_sat
    v_eci = v_deb - v_sat

    #Radial
    norm = np.linalg.norm(r_sat)
    radial_hat = r_sat / norm

    #Cross Track
    w_hat = np.cross(r_sat, v_sat)
    norm = np.linalg.norm(w_hat)
    w_hat = w_hat / norm

    #Along-Track
    s_hat = np.cross(w_hat, radial_hat)
    norm = np.linalg.norm(s_hat)
    s_hat = s_hat / norm

    #Construct Rotation Matrix
    rotation = np.array([radial_hat, s_hat, w_hat])

    #Rotate position to LVLH
    r_final = rotation @ r_eci #Since it is column based

    #Correct velocity from coriolis effect
    ang_vel = np.cross(r_sat, v_sat) / pow(np.linalg.norm(r_sat), 2)
    v_corr = v_eci - (np.cross(ang_vel, r_eci))

    #Rotate velocity to LVLH
    v_final = rotation @ v_corr

    return r_final, v_final

def update_satellite_config():
    

if __name__ == "__main__":
    main()
