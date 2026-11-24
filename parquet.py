import pyarrow.parquet as pq

file_path = "/home/taquee/Desktop/Science_Fair_10th/leo-ml-satellite-simulation/tle_2025.parquet"
pf = pq.ParquetFile(file_path)

samples = []
seen_ids = set()

# Stream small batches from disk and stop the instant we find 5 LEO satellites
for batch in pf.iter_batches(batch_size=5000):
    df = batch.to_pandas()
    leo = df[(df["altitude_km"] >= 400.0) & (df["altitude_km"] <= 800.0)]
    
    for _, row in leo.iterrows():
        nid = row["norad_id"]
        if nid not in seen_ids:
            seen_ids.add(nid)
            samples.append(row)
            if len(samples) == 5:
                break
    if len(samples) == 5:
        break

import pandas as pd
print(pd.DataFrame(samples).to_string(index=False))
