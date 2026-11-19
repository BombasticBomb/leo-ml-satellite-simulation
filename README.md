# Autonomous LEO Satellite Collision Avoidance on Edge Hardware

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Simulation](https://img.shields.io/badge/Simulation-Basilisk%20%7C%20bsk__rl-orange.svg)](https://hanspeterschaub.info/basilisk/)
[![RL](https://img.shields.io/badge/RL-PPO%20(Stable--Baselines3)-green.svg)](https://stable-baselines3.readthedocs.io/)
[![Target](https://img.shields.io/badge/Target-ESP32--S3%20(INT8)-red.svg)](https://www.espressif.com/en/products/socs/esp32-s3)

An onboard autonomous space debris avoidance system for Low Earth Orbit (LEO) CubeSats[cite: 1, 2]. This project trains a lightweight **Graph Attention Network (GAT)** via **Proximal Policy Optimization (PPO)** inside a high-fidelity `Basilisk` (`bsk_rl`) orbital simulation environment, then quantizes the policy to **INT8 (~9.8 KB)** to run natively on a 240 MHz **ESP32-S3** microcontroller[cite: 1, 2, 5, 6, 8].

---

## Key Features

- **Physics-Accurate LEO Environment:** Built on `Basilisk` and `bsk_rl` (Gymnasium), modeling 2-body gravity, $J_2$ oblateness perturbations, atmospheric drag ($C_D = 2.2$), and Solar Radiation Pressure (SRP) via RK4 integration[cite: 2, 3].
- **5-Stage Real-Time Debris Pipeline:** Filters ~30,000 cataloged 2025 debris objects every training timestep without bottlenecking CPU compute[cite: 2].
- **Ego-Centric Graph Attention Policy:** Uses a 1-layer, 2-head GAT to prioritize the 10 nearest threats in the spacecraft's Local Vertical, Local Horizontal (LVLH) frame[cite: 4, 5, 6].
- **Realistic 6U CubeSat & Propulsion Dynamics:** Models a Dawn Aerospace B1 thruster with dynamic mass depletion as fuel burns[cite: 8, 9, 10].
- **Ultra-Low Edge Footprint:** Compiles to **9,757 parameters (~9.8 KB in INT8 / ~39.0 KB in FP32)** via `ESP-DL`, leaving >510 KB of SRAM free on the ESP32-S3[cite: 2, 8].

---

## 5-Stage Debris Data Pipeline

At every simulation timestep, the environment processes the full 2025 orbital catalog through a 5-stage funnel before feeding observations to the agent:

1. **Load Catalog:** Ingests all **~30,000 TLEs** from the 2025 debris dataset[cite: 2, 3].
2. **Hoots Geometric Filter (Part 1):** Runs a fast perigee/apogee altitude check against the host satellite's orbital band, discarding **~80–90%** of non-intersecting objects immediately.
3. **SGP4 Propagation:** Converts only the surviving ~10–20% candidate TLEs into **ECI** position and velocity vectors for the current epoch[cite: 3, 4].
4. **ECI-to-LVLH Rotation:** Transforms relative state vectors into the satellite's ego-centric **LVLH** frame (Radial, Along-Track, Cross-Track), accounting for frame rotation via the Coriolis term[cite: 4, 5].
5. **100 km Spatial Crop:** Computes Euclidean distance to filter objects within a **100 km radius**, selects the **closest 10 debris objects**, and passes them into the GAT[cite: 6].

---

## Spacecraft & Propulsion Model

- **Satellite Bus:** 6U CubeSat initialized in a 550 km near-circular LEO orbit ($53^\circ$ inclination)[cite: 2, 8].
- **Mass Breakdown:** **11.0 kg wet mass**, **9.7 kg dry mass**, and **1.3 kg propellant** stored in a 2U CubeDrive tank[cite: 8].
- **Thruster System:** **Dawn Aerospace B1** dual-mode propulsion[cite: 8]:
  - *Thruster 0 (High-Efficiency):* $I_{\text{sp}} = 285\text{ s}$ (~0.36 g/N·s) for nominal trajectory adjustments[cite: 8, 10].
  - *Thruster 1 (Emergency Cold-Gas High-Impulse):* ~4.5$\times$ fuel rate for rapid last-second collision avoidance[cite: 8, 9, 10].
- **$\Delta v$ Budget & Mass Updates:** Rated for $2,200\text{ N}\cdot\text{s}$ total impulse, yielding **~213 m/s** of usable operational $\Delta v$ (351.5 m/s theoretical)[cite: 9]. Spacecraft mass is dynamically updated in Basilisk after every burn[cite: 10].

---

## Neural Network & Edge Footprint

### Why a Graph Attention Network (GAT)?
Unlike standard MLPs—which depend on arbitrary input ordering and waste capacity learning permutation invariance—a **GAT** treats surrounding debris as an unordered spatial graph[cite: 5, 6]. It computes attention weights based on relative LVLH position and velocity[cite: 5, 6], letting the agent focus on the most immediate collision threat while still accounting for secondary debris when planning an avoidance burn[cite: 5].

### Model Footprint (PPO + INT8 Quantization)
- **Inputs:** `[Batch, 10, 6]` debris LVLH state tensor + `[Batch, 4]` host satellite telemetry (fuel, $\Delta v$, velocity)[cite: 6].
- **Architecture:** Debris Embedding (`6→32`) $\rightarrow$ 1-Layer, 2-Head GAT (`32`) $\rightarrow$ FFN (`32→64→32`) $\rightarrow$ Policy Head (`36→72`) with hybrid discrete (axis: None, Radial, Cross-Track, Along-Track) and continuous ($\Delta v$ magnitude + thruster mode) outputs[cite: 6, 7, 8].
- **Total Size:** **9,757 parameters** $\rightarrow$ **~9.8 KB (INT8)** via `ESP-DL` (or **~39.0 KB** unquantized FP32), leaving **>510 KB SRAM** available on the ESP32-S3[cite: 2, 8].

---

## Reward Function

The agent is trained with a shaped reward that balances orbital survival, close-approach avoidance, and fuel conservation[cite: 10]:

$$R_t = r_{\text{alive}} - \text{Fuel Penalty (Quadratic)} - \text{Proximity Penalty (Exponential)} - \text{Collision Penalty} + \text{Avoidance Bonus}$$

- **Step Survival (`+1`):** Encourages continuous safe orbit maintenance[cite: 10].
- **Quadratic Fuel Penalty:** Permits small, efficient burns while penalizing excessive $\Delta v$ expenditure[cite: 10].
- **Exponential Proximity Penalty:** Heavily discourages close conjunctions as separation shrinks[cite: 10].
- **Terminal Bonus (`+1000` to `+2000`):** Awarded for safely clearing a conjunction event without collision[cite: 10].
