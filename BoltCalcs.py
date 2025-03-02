"""
Created by Andrew Park 03/02/2025 14:36:24 CST
Bolt tensile and shear strengths as inputs, given in MPa
Input file should give x and r as arrays in inches. The code will convert to SI then back to inches. All other dimensions manually input should be in SI
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sympy import *

def runBoltCalcs(MEOP, contour, axial_bolts, radial_bolts):
    x, r = contour
    MEOP *= 1000000
    load_area = np.pi*(max(r)**2 - min(r)**2) #m2
    
    load = MEOP * load_area #N
    num_axial = axial_bolts["Quantity"][0]
    num_radial = radial_bolts["Quantity"][0]
    if num_axial > 0:
        axial_stress = (load/(axial_bolts["Area"]*num_axial))/1000000 #Pa -> MPa
        SF_axial = axial_bolts["Tensile Strength"]/axial_stress
    if num_radial > 0:
        radial_stress = (load/(radial_bolts["Area"]*num_radial))/1000000 #Pa -> MPa
        SF_radial = radial_bolts["Shear Strength"]/radial_stress
    return np.array(SF_axial)[0], np.array(SF_radial)[0]
        

def create_contour(contour_file):
    contour = pd.read_csv(contour_file, header=0)
    contour.rename(columns={contour.columns[0]: 0, contour.columns[1]: 1}, inplace=True)
    x = np.array(contour[0]) #in
    r = np.array(contour[1]) #in
    x -= x[0]
    x *= 25.4 #mm
    r *= 25.4 #mm
    #Converting all to SI units (m, m, m, m, Pa)
    return np.array([x/1000, [i / 1000 for i in r]])

axial_bolts = pd.DataFrame([{"Tensile Strength": 482.63, "Shear Strength":289.58, "Area": 0.0003888, "Quantity": 8}]) #MPa, MPa, m2, number
radial_bolts = pd.DataFrame([{"Tensile Strength": 482.63, "Shear Strength": 289.58, "Area": 0.0003888, "Quantity": 6}]) #MPa, MPa, m2, number
contour_file = "engine_contour_test9.csv"
contour = create_contour(contour_file)
MEOP = 2.06 #MPa = 300psi
bolt_calcs = runBoltCalcs(MEOP, contour, axial_bolts, radial_bolts)
print(float(bolt_calcs[0]), float(bolt_calcs[1]))