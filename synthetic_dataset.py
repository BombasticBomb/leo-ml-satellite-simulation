import numpy as np

satellites = 15000
gat_incln = 53
id = np.arange(0,satellites+1)
incln = np.round(np.random.uniform(gat_incln - 15, gat_incln + 15, satellites), 14)

print(incln)