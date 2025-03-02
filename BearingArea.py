#Created by Andrew Park 12/07/2024 15:53:13 CST
#Graphite material properties from: https://www.azom.com/properties.aspx?ArticleID=516
#NOTE: Input file should give x and r as arrays in inches. The code will convert to SI then back to inches. All other dimensions manually input should be in SI
#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sympy import *

def find_min_ring_thickness(x, r, th_Oring, x_Oring, MEOP):
    #Finding radius and x value at the nozzle opening
    r_last = r[len(r)-1]
    x_last = x[len(x)-1]
    #Initializing variables. For increased precision of ring thicknesses, decrease third parameter of np.arange
    thicknesses = np.arange(0, 20.01, 0.01) #mm
    sigma_bearing = []
    sigma_shear = []
    FS_bearing = []
    FS_shear = []
    graphite_tensile_strength = 76*1000000 #Pa
    graphite_compressive_strength = 345*1000000 #Pa
    est_bearing_strength = graphite_compressive_strength
    est_shear_strength = 0.1*graphite_compressive_strength
    #Looping through possible ring thicknesses
    for i in range(len(thicknesses)):
        #Geomtery relations
        th = thicknesses[i]/1000 #mm to m
        r_inner = r_last
        r_outer = r_last + th
        d_inner = r_inner*2
        d_outer = r_outer*2

        #Bearing strength code
        #Finding bearing area (1)
        A_bearing = np.pi*(d_outer-th)*th
        if A_bearing != 0:
            #Finding bearing stress and safety factor (2) & (3)
            s_b = np.pi*(d_inner**2)*MEOP/A_bearing
            FS_b = est_bearing_strength/s_b
        else:
            s_b = 0
            FS_b = 0

        #Shear strength code
        #Finding shear line length
        if th < 1.5*th_Oring:
          index = np.where(r<r_inner-th)[0][0]
          L = x_last - x_Oring
        else:
          L = x_last - x[index]
        #Finding shear area (4)
        A_shear = np.pi*d_inner*L
        #Finding shear stress and safety factor (5) & (6)
        s_s = (np.pi*(d_inner**2)*MEOP)/(4*A_shear)
        FS_s = est_shear_strength/s_s
        #For each ring thickness, adding the shear and bearing stresses and safety factors to a list
        sigma_bearing.append(s_b)
        sigma_shear.append(s_s)
        FS_bearing.append(FS_b)
        FS_shear.append(FS_s)
    #Correcting the data type
    FS_bearing = [float(x) for x in FS_bearing]
    FS_shear = [float(x) for x in FS_shear]
    #Plotting thickness (in inches) against bearing safety factor
    plt.plot(thicknesses/25.4, FS_bearing, 'r', label='Bearing')
    plt.plot(thicknesses/25.4, FS_shear, 'b', label='Shear')
    plt.title('Safety Factors vs. Retaining Ring Thickness')
    plt.xlabel('Ring Thickness (in)')
    plt.ylabel('Safety Factor')
    plt.legend()
    plt.show()

def create_contour(contour):
    contour = pd.read_csv(contour)
    print(contour)
    x = contour['x'] #in
    r = contour['y'] #in
    x -= x[0]
    x *= 25.4 #mm
    r *= 25.4 #mm

    #All manually altered values here should stay in SI or corrections should be made later on
    #These values correspond to: O-ring groove thickness, O-ring groove location, Maximum Expected Operating Pressure
    th_Oring = 1.6 #mm (1/16 in)
    x_Oring = 63.5 #mm (2.5 in)
    MEOP = 2.06 #MPa (300psi)
    plt.plot(x, r)
    plt.title('Contour')
    plt.xlabel('Axial distance (mm)')
    plt.ylabel('Radius (mm)')

    plt.show()
    #Converting all to SI units (m, m, m, m, Pa)
    find_min_ring_thickness(x/1000, [i / 1000 for i in r], th_Oring/1000, x_Oring/1000, MEOP*1000000)

contour_file = "engine_contour_test9.csv"
create_contour(contour_file)