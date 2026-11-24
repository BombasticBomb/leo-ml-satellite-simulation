from beyond.frames.local import to_qsw
import sqlite3
import pandas as pd
import csv
from sgp4.api import jday

def hoots_part_1(perigee_A, eccentricity_A, perigee_B, eccentricity_B):
    altitude_band = 30 #Altitude band in km

    #Add Earth's radius as it is currently altitude
    perigee_A += 6371
    perigee_B += 6371

    #Calculate Apogee from Perigee & Eccentricity
    apogee_A = perigee_A * ((1+eccentricity_A)/(1-eccentricity_A))
    apogee_B = perigee_B * ((1+eccentricity_B)/(1-eccentricity_B))

    if ((perigee_A - apogee_B) > altitude_band) and ((perigee_B - apogee_A) > altitude_band):
        return True
    return False

def sgp_propagation():
    print("Hello")

def eci_to_lvlh(r_sat, v_sat, r_deb, v_deb):
    rel_eci = r_deb[:3] - r_deb[:3]

    
def toca(r_deb, v_deb):
    output  = -np.dot(r_deb, v_deb) / (np.linalg.norm(v_deb))**2
    return toca


df = pd.read_csv("dataset.csv")
#Debris Id
id = df["norad_id"].to_numpy()
#Perigee, converted to meters for Basilisk
perigee = df["perigee_km"].to_numpy()
perigee = perigee * 1000
#Rest of the keplerian elements
ecc = df["eccentricity"].to_numpy()
incln = df["inclination"].to_numpy()
raan = df["raan"].to_numpy()
mean_an = df["mean_anomaly"].to_numpy()
mean_mot = df["mean_motion"].to_numpy()
bstar = df["bstar"].to_numpy()
#Time in epoch
epoch = df["epoch"].to_numpy()

'''
epoch = pd.to_datetime(epoch, utc=True)

# 2. Vectorized conversion to two 1D float64 NumPy arrays: (jd, fr)

jd, fr = jday(
    epoch.year.to_numpy(),
    epoch.month.to_numpy(),
    epoch.day.to_numpy(),
    epoch.hour.to_numpy(),
    epoch.minute.to_numpy(),
    (epoch.second + epoch.microsecond / 1e6).to_numpy()
)

#2433281.5 is the Julian Date for December 31, 1949, at 00:00:00 UTC.
days_till_1949 = (jd + fr) - 2433281.5
'''

import numpy as np

# Replace with your actual array of angles (e.g., inclinations in degrees)
# angles_deg = df["INCLINATION"].to_numpy()

# 1. Round to the nearest integer degree
rounded_deg = np.rint(incln).astype(int)

# 2. Keep values within 1° to 180°
valid_deg = rounded_deg[(rounded_deg >= 1) & (rounded_deg <= 180)]

# 3. Count frequency for every integer degree from 1 to 180
counts = np.bincount(valid_deg, minlength=181)[1:181]
max_count = counts.max() if counts.max() > 0 else 1

# 4. Print the full 1°–180° spread
print(f"{'Deg':>4} | {'Count':>6} | Spread")
print("-" * 50)
for deg in range(1, 181):
  c = counts[deg - 1]
  bar = "█" * int((c / max_count) * 40)  # Scales longest bar to 40 chars
  print(f"{deg:3d}° | {c:6d} | {bar}")
