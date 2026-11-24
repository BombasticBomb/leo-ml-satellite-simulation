import numpy as np
import math
import pymap3d as pm
import pymsis
from datetime import datetime, timedelta
import logging
import satkit as sk

# Silence pymap3d and root logging warnings
logging.getLogger("pymap3d").setLevel(logging.ERROR)
logging.basicConfig(level=logging.ERROR)

#Constants
radius_earth_equatorial = 6378.1366
j2_constant = 1.08262668 * pow(10, -3)
earth_grav_param = 398600.435507
omega_earth = np.array(
    [0.0, 0.0, 7.2921159e-5] 
)  # Earth angular velocity (rad/s)


def catalog_row_to_eci(
    eccentricity,
    inclination,
    raan,
    mean_anomaly,
    mean_motion,
    bstar,
    epoch,
    argp=0.0,
):
    omm = {
        "EPOCH": epoch,
        "MEAN_MOTION": float(mean_motion),
        "ECCENTRICITY": float(eccentricity),
        "INCLINATION": float(inclination),
        "RA_OF_ASC_NODE": float(raan),
        "ARG_OF_PERICENTER": float(argp),
        "MEAN_ANOMALY": float(mean_anomaly),
        "BSTAR": float(bstar),
    }

    t_epoch = sk.time(epoch)

    # Get state in TEME frame at epoch, then rotate to inertial GCRF (ECI)
    pos_teme, vel_teme = sk.sgp4(omm, t_epoch)
    q_teme2gcrf = sk.frametransform.rotation(sk.frame.TEME, sk.frame.GCRF, t_epoch)

    return (q_teme2gcrf * pos_teme) / 1000.0, (q_teme2gcrf * vel_teme) / 1000.0

#Testing SGP4
date = datetime(2026, 8, 29, 8, 5, 41, 296128)
r_sat, v_sat = catalog_row_to_eci(
    eccentricity=0.2,
    inclination=53,
    raan=296.44,
    mean_anomaly=272.12,
    mean_motion=10,
    bstar=0.000112,
    epoch=date.isoformat(),
    argp=88.03,
    )


def twoBody(position):
    accel = (-earth_grav_param / math.pow((np.linalg.norm(position)), 3)) * position
    return accel

def j2(position):
    x = position[0]
    y = position[1]
    z = position[2]
    r = np.linalg.norm(position)
    constant = (-3 * j2_constant * earth_grav_param * pow(radius_earth_equatorial, 2)) / (pow(r, 5) * 2)
    j2x = constant * x * (1 - 5*(pow(z, 2) / pow(r, 2)))
    j2y = constant *  y * (1 - 5*(pow(z, 2) / pow(r, 2)))
    j2z = constant * z * (3 - 5*(pow(z, 2) / pow(r, 2)))
    accel = np.array([j2x, j2y, j2z])
    return accel

#This one is for efficiency
saved_density = 0
def atmospheric_drag(r_sat, v_sat, dt, dragCoeff, A, m, calculate):
    global saved_density
    f107 = 140
    f107a = 140
    ap = 15

    #Convert to Geodetic
    lat, long, alt = pm.eci2geodetic(
        r_sat[0] * 1000,
        r_sat[1] * 1000,
        r_sat[2] * 1000,
        dt
    )

    #Format inputs for pymsis (altitude in km)
    dates = np.array([dt], dtype="datetime64[s]")
    aps = np.array([[ap] * 7])

    #Calculate atmospheric density
    if calculate == True:
        density = pymsis.calculate(
            dates,
            long,
            lat, 
            (alt / 1000), #In km
            f107s=f107,
            f107as=f107a,
            aps = aps
        )
        density = float(density.ravel()[0]) #Take only atmospheric density
        saved_density = density
    else:
        density = saved_density

    #Calculate relative velocity for drag equation
    v_rel_atm = v_sat - np.cross(omega_earth, r_sat)
    drag = -0.5 * density * dragCoeff * (A / m) * (np.linalg.norm(v_rel_atm) * 1000) * (v_rel_atm * 1000)
    return drag / 1000.0




#Add SRP after done
def compute_total_acceleration(r_sat, v_sat, date, calculate):
    a_2b = twoBody(r_sat)
    a_j2 = j2(r_sat)
    a_drag = atmospheric_drag(r_sat, v_sat, date, 2.2, 0.055, 12, calculate) #Tune later
    return a_2b + a_j2 + a_drag



def rk4(r_sat, v_sat, date, dt):
    #The k is the 4 slope is estimates (ex slope of position is velocity)

    #K1
    k1_r = v_sat
    k1_v = compute_total_acceleration(r_sat, v_sat, date, True)

    #K2
    r2_sat = r_sat + (0.5 * dt * k1_r)
    v2_sat = v_sat + (0.5 * dt * k1_v)
    k2_r = v2_sat
    k2_v = compute_total_acceleration(r2_sat, v2_sat, date + timedelta(seconds=0.5 * dt), False)

    #K3
    r3_sat = r_sat + (0.5 * dt * k2_r)
    v3_sat = v_sat + (0.5 * dt * k2_v)
    k3_r = v3_sat
    k3_v = compute_total_acceleration(r3_sat, v3_sat, date + timedelta(seconds=0.5 * dt), False)

    #K4
    r4_sat = r_sat + (dt * k3_r)
    v4_sat = v_sat + (dt * k3_v)
    k4_r = v4_sat
    k4_v = compute_total_acceleration(r4_sat, v4_sat, date + timedelta(seconds=dt), False)

    #Combine and output
    output_r = r_sat + (dt / 6.0) * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r)
    output_v = v_sat + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)

    return output_r, output_v


dt = 1.0
steps = 10000
print(f"Starting Position (km): {r_sat}")
for step in range(steps):
    r_sat, v_sat = rk4(r_sat, v_sat, date, dt)
    date += timedelta(seconds=dt)
    if step % 10 == 0:
        print(f"Step {step}: Pos = {r_sat} km | Alt = {np.linalg.norm(r_sat) - radius_earth_equatorial:.2f} km")
