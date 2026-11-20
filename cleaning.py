import re
import pandas as pd

input_file = "tle_2025_clean.sql"
output_file = "tle_2025_clean.sql"  # Overwrites in place (or change to "tle_2025_reindexed.sql")
table_name = "tle_2025_catalog"

# 1. Read and parse all value tuples from tle_2025_clean.sql
with open(input_file, "r", encoding="utf-8") as f:
  content = f.read()

# Match each row tuple: (norad_id, 'epoch', inc, raan, ecc, argp, ma, mm, bstar)
pattern = re.compile(
    r"\((\d+),\s*'([^']+)',\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*([^)]+)\)"
)
matches = pattern.findall(content)

cols = [
    "orig_norad_id",
    "epoch",
    "inclination",
    "raan",
    "eccentricity",
    "arg_perigee",
    "mean_anomaly",
    "mean_motion",
    "bstar",
]
df = pd.DataFrame(matches, columns=cols)

# Convert numeric columns from string
num_cols = [
    "orig_norad_id",
    "inclination",
    "raan",
    "eccentricity",
    "arg_perigee",
    "mean_anomaly",
    "mean_motion",
    "bstar",
]
for c in num_cols:
  df[c] = pd.to_numeric(df[c], errors="coerce")

# 2. Sort by original NORAD ID and assign new sequential norad_id from 0 to N-1
df = df.sort_values("orig_norad_id").reset_index(drop=True)
df["norad_id"] = df.index  # 0, 1, 2, ..., len(df) - 1

# Reorder columns to match the SQL table schema
final_cols = [
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
df_reindexed = df[final_cols]

print(f"Total rows re-indexed: {len(df_reindexed):,}")
print(
    f"New norad_id range:    0 to {df_reindexed['norad_id'].max():,}"
)

# 3. Write the updated standalone .sql file
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

  for start in range(0, len(df_reindexed), 1000):
    chunk = df_reindexed.iloc[start : start + 1000]
    rows_sql = []
    for row in chunk.itertuples(index=False):
      vals = [
          "NULL"
          if pd.isna(v)
          else (f"'{v}'" if isinstance(v, str) else str(v))
          for v in row
      ]
      rows_sql.append(f"({', '.join(vals)})")

    f.write(f"INSERT INTO `{table_name}` VALUES\n" + ",\n".join(rows_sql) + ";\n\n")

print(f"Updated {output_file} with sequential norad_ids (0 to {len(df_reindexed) - 1})!")
