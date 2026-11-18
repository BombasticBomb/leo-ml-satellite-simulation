import numpy as np

#ECI coordinates
r_sat = np.array([0,6000,0])
v_sat = np.array([-6,0,0])
r_deb = np.array([-10,6004,3])
v_deb = np.array([-6.003,0.01,0])

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

#Info
print(f"Radial: ", radial_hat)
print(f"Cross-Track: ", w_hat)
print(f"Along-Track: ", s_hat)
print(f"Rotation Matrix: ", rotation)
print(f"Angular Velocity: ", ang_vel)
print(f"Corrected Velocity: ", v_corr)
print(f"Final Position: ", r_final)
print(f"Final Velocity: ", v_final)
