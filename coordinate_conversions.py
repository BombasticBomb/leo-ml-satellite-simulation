from beyond.frames.local import to_qsw

def hoots_part_1(perigee_A, eccentricity_A, perigee_B, eccentricity_B):
    altitude_band = 30 #Altitude band in km

    #Add Earth's radius as it is currently altitude
    perigee_A += 6371
    perigee_B += 6371

    #Calculate Apogee from Perigee & Eccentricity
    apogee_A = perigee_A((1+eccentricity_A)/(1-eccentricity_A))
    apogee_B = perigee_B((1+eccentricity_B)/(1-eccentricity_B))

    if ((perigee_A - apogee_B) > altitude_band) and ((perigee_B - apogee_A) > altitude_band):
        return True
    return False

def sgp_propagation():
    print("Hello")

def eci_to_lvlh(r_sat, v_sat, r_deb, v_deb):
    rel_eci = r_deb[:3] - r_deb[:3]
    