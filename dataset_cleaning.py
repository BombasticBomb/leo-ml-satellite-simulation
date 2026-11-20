import pandas as pd
import pyarrow.parquet as pq

target_cols = [
    "norad_id",
    "epoch",
    "inclination",
    "raan",
    "eccentricity",
    "arg_perigee",
    "mean_anomaly",
    "mean_motion",
    "bstar",
]

parquet_file = pq.ParquetFile("tle_2025.parquet")
batch_summaries = []

# 250k batch size keeps RAM usage tiny (~30 MB per batch) on a 4GB Pi 5
for batch in parquet_file.iter_batches(batch_size=250_000, columns=target_cols):
  df_chunk = batch.to_pandas(types_mapper=pd.ArrowDtype)

  # Keep 1 row per norad_id in this chunk (latest epoch; set ascending=True for earliest)
  df_chunk = df_chunk.sort_values("epoch", ascending=False).drop_duplicates(
      subset=["norad_id"]
  )
  batch_summaries.append(df_chunk)

# Combine and deduplicate globally across all 20.78M rows
sat_catalog = (
    pd.concat(batch_summaries, ignore_index=True)
    .sort_values("epoch", ascending=False)
    .drop_duplicates(subset=["norad_id"])
    .sort_values("norad_id")
    .reset_index(drop=True)
)

# Format the UTC timestamp directly to a MySQL string (fixes the KeyError)
sat_catalog["epoch"] = (
    pd.to_datetime(sat_catalog["epoch"])
    .dt.tz_localize(None)
    .dt.strftime("%Y-%m-%d %H:%M:%S.%f")
)

print(f"Unique satellites extracted: {len(sat_catalog):,}")

# Save as a standalone MySQL .sql file
table_name = "tle_2025_catalog"
output_file = "tle_2025_clean.sql"

with open(output_file, "w", encoding="utf-8") as f:
  f.write(f"DROP TABLE IF EXISTS `{table_name}`;\n")
  f.write(f"""CREATE TABLE `{table_name}` (
    `norad_id` INT PRIMARY KEY,
    `epoch` DATETIME(6),
    `inclination` FLOAT,
    `raan` FLOAT,
    `eccentricity` FLOAT,
    `arg_perigee` FLOAT,
    `mean_anomaly` FLOAT,
    `mean_motion` DOUBLE,
    `bstar` DOUBLE
);\n\n""")

  for start in range(0, len(sat_catalog), 1000):
    chunk = sat_catalog.iloc[start : start + 1000]
    rows_sql = []
    for row in chunk.itertuples(index=False):
      vals = [
          "NULL"
          if pd.isna(v)
          else (
              f"'{v.replace(chr(39), chr(39)*2)}'"
              if isinstance(v, str)
              else str(v)
          )
          for v in row
      ]
      rows_sql.append(f"({', '.join(vals)})")

    f.write(f"INSERT INTO `{table_name}` VALUES\n" + ",\n".join(rows_sql) + ";\n\n")

print(f"Saved standalone file: {output_file}")


