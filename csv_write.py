import csv
import sqlite3
import pandas as pd

with sqlite3.connect("dataset.db") as conn:
    df = pd.read_sql_query(
        "SELECT * FROM tle_2025_catalog ORDER BY norad_id ASC",
        conn,
        parse_dates=["epoch"]  # Ensures epoch is a pandas Timestamp, not a string
    )
satellites = df.set_index("norad_id").to_dict(orient="index")

headers = [
    "norad_id", "perigee_km", "eccentricity", "apogee_km", 
    "inclination", "raan", "mean_anomaly", "mean_motion", 
    "bstar", "epoch"
]
with open("dataset.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(headers)  # Write the column names once at the top

    for i in range(0, 31711):
        sat = satellites[i]
        
        id = i
        perigee = sat["arg_perigee"] + 6371
        eccentricity = sat["eccentricity"]
        apogee = perigee * ((1 + eccentricity) / (1 - eccentricity))
        inclination = sat["inclination"]
        raan = sat["raan"]
        mean_anomaly = sat["mean_anomaly"]
        mean_motion = sat["mean_motion"]
        bstar = sat["bstar"]
        epoch = sat["epoch"]

        # Write the row to the CSV
        writer.writerow([
            id, perigee, eccentricity, apogee, 
            inclination, raan, mean_anomaly, mean_motion, 
            bstar, epoch
        ])

print("Done! Saved to dataset.csv")



