# -*- coding: utf-8 -*-
"""
Created on by Andrew Park on Sun Nov 10 23:59:13 2024

Inputs: x & radius (Contour), initial temp of wall (probably room temp), initial gas temp, pressure, and mach number
(Note: mach number will need to be played around with a bit) at injector face, epselon (surface roughness) of contour
material as an array along the contour, characteristic flow velocity, mass flow rate

Outputs: Anything you really want (gas temp, heat flux, pressure, viscosity, thermal conductivity, gamma, etc.) all in
array form corresponding to the length of the contour. This does not give steady state but rather all at one instant in time

To run the program, input all required values (you'll need to play around with mach number until it returns a plot that goes supersonic at the throat.
Due to the equations being used, we cannot set M=1 exactly at the throat)
Basically we initialize all our initial guesses, then loop through iterating on all of them until some arbitrary percent error is less than our desired error value.
All equations and processes are based off the equations from the "Regen Code Equations" document, which gathered info from cryorocket.com, which got info from NASA papers
Yes, the information was eventually taken directly from the NASA papers and fact-checked. The papers and websites should all be in the "Regen Code Equations" document.
The main NASA document is going to be "THE ONE-DIMENSIONAL THEORY OF STEADY COMPRESSIBLE FLUID FLOW IN DUCTS WITH FRICTION AND HEAT ADDITION" by Hicks, Montgomery, and Wasserman
if you wanted to look at the direct source
The cryo-rocket website outlines the process we take to get our results, which we followed as a guideline to create this code
"""
import numpy as np
from scipy.special import lambertw

#x = np.asarray((input("Input axial sections: ")))
#radius = np.asarray(list(input("Input radii at axial sections: ")))
#These are our inputs that we need to manually change
x = np.linspace(0, 0.5, 101) #Placeholder values (m)
radius = (x + 2)/2
T_wall0 = 298 #Room temp in K
T_free_stream0 = 3300 #Estimated initial chamber temp @ injector face (K)
P_0 = 1 #Estimated initial chamber pressure @ injector face (Pa)
M_0 = 0.25 #Initial guess of mach number at injector is key to a complete analysis. Play around with this until things look right.
epselon = 0.0000015 #Epselon for aluminum (This is epselon in the epselon/d term for darcy friction factor and can be experimentally determined for our graphite)
c_star = 1 #Characteristic velocity (m/s)
m_dot = 2 #Mass flow rate (kg/s)
mass_frac_CO2 = 0.5
mass_frac_H2O = 0.2
mass_frac_O2 = 0.3
mol_frac_CO2 = 0.5
mol_frac_H20 = 0.3
mol_frac_O2 = 0.2

MM_CO2 = 0.044009 #Molar mass (kg/mol)
MM_H2O = 0.01801528 #Molar mass (kg/mol)
MM_O2 = 0.032004 #Molar mass (kg/mol)
MM_total = MM_CO2 + MM_H2O + MM_O2 #(kg/mol)
dx = x[1]-x[0] #(m)
area = 2*np.pi*radius*dx #(m^2)
dAdx = np.gradient(area, dx) #(m^2/m)
D_t = np.min(radius)*2 #Throat diameter (m)
A_t = np.min(area) #Throat area (m)
Ru = 8.31446261815324 #Universal gas constant (J/mol K)
Rs = Ru/MM_total #Specific gas constant of entire mixture (J/kg K)

#We cannot assume N=1 exactly at throat because it breaks the diff. eq because there is (N-1) term in denominator
def Run_Heat_Transfer(T_wall0, T_free_stream0, M_0, P_0, A, dAdx, epselon):
    #initializing variables
    dQdx = np.zeros(len(x)) #Derivative of heat with respect to length (J/m)
    dFdx = np.zeros(len(x)) #Derivative of energy lost to friction with respect to length (J/m)
    T_wall = np.zeros(len(x)) + T_wall0 #Does not change unless there is a time change. Everything else should iteratively converge (K)
    T_free_stream = np.zeros(len(x)) + T_free_stream0 #Temperature of the free stream of gas(K)
    N = np.zeros(len(x)) + (M_0**2) #Initializing static N based on initial guess at injector (Unitless)
    P = np.zeros(len(x)) + P_0 #Initial pressure guess for the rk4 algorithm (Pa)
    T_star = calc_T_star(T_free_stream, N, T_wall) #Reference temperature for calculating transport properties

    #Defining these terms for the error function later on. They will be updated and modified in the error function
    T_free_stream_last = np.array([T_free_stream, T_free_stream, T_free_stream])
    T_wall_last = np.array([T_wall, T_wall, T_wall])
    q_last = np.array([np.zeros(len(x)),np.zeros(len(x)),np.zeros(len(x))])
    hg_last = np.array([np.zeros(len(x)),np.zeros(len(x)),np.zeros(len(x))])
    N_last = np.array([N, N, N])
    P_last = np.array([P, P, P])

    #Running loop. This will end when our percent error is "good enough" or, as defined currently, after 200 iterations
    for i in range(200):
        #Here we use a fourth-order Runge-Kutta algorithm to calculate N, P, and T based on each other and the other variables
        N = rk4(f_N, x, N[0], N, P, T_free_stream, A, dQdx, dFdx, dAdx, T_star) #(Dimensionless)
        P = rk4(f_P, x, P[0], N, P, T_free_stream, A, dQdx, dFdx, dAdx, T_star)
        T_free_stream = rk4(f_T, x, T_free_stream[0], N, P, T_free_stream, A, dQdx, dFdx, dAdx, T_star)

        #Now we use those terms to calculate flow properties
        T_star = calc_T_star(T_free_stream, N, T_wall)
        cp = calc_cp(T_star)
        cv = Rs - cp
        gamma = cp/cv
        viscosity, thermal_conductivity = calc_viscosity_and_lambda(T_star)
        Pr = (cp*viscosity)/thermal_conductivity
        c = np.sqrt(gamma*Rs*T_free_stream)
        f = calc_f(epselon, A, density, N, viscosity, c)
        #etc, etc, whatever we need here (i.e. density, Re, f, and everything needed for hg)
        T_stag, T_aw = calc_Taw(T_free_stream, N, Pr, gamma)
        hg = calc_hg(viscosity, cp, c_star, A, Pr, T_wall, T_stag, P, N, gamma)
        q = hg*A*(T_aw-T_wall)
        #Get ready to calculate dQdx and dFdx here
        dQdx = q*A/m_dot
        dFdx = calc_dFdx(f, N, T_free_stream, A, gamma)
        #End of recursion calculation. Below is error calculation for end condition
        T_free_stream_diff = np.average([np.abs(T_free_stream-T_free_stream_last[0]), np.abs(T_free_stream-T_free_stream_last[1]), np.abs(T_free_stream-T_free_stream_last[2])])
        T_wall_diff = np.average([np.abs(T_wall-T_wall_last[0]), np.abs(T_wall-T_wall_last[1]), np.abs(T_wall-T_wall_last[2])])
        q_diff = np.average([np.abs(q-q_last[0]), np.abs(q-q_last[1]), np.abs(q-q_last[2])])
        hg_diff = np.average([np.abs(hg-hg_last[0]), np.abs(hg-hg_last[1]), np.abs(hg-hg_last[2])])
        N_diff = np.average([np.abs(N-N_last[0]), np.abs(N-N_last[1]), np.abs(N-N_last[2])])
        P_diff = np.average([np.abs(P-P_last[0]), np.abs(P-P_last[1]), np.abs(P-P_last[2])])

        T_free_stream_last = [T_free_stream_last[1], T_free_stream_last[2], T_free_stream]
        T_wall_last = [T_wall_last[1], T_wall_last[2], T_wall]
        q_last = [q_last[1], q_last[2], q]
        hg_last = [hg_last[1], hg_last[2], hg]
        N_last = [N_last[1], N_last[2], N]
        P_last = [P_last[1], P_last[2], P]

    #We could make it return other things here if desired. We could also create a table of values and export to PDF
    return q

def calc_T_star(T_free_stream, N, T_wall):
  #Reference temperature for finding flow properties based on T_free_stream, T_wall, and mach number
  return (T_free_stream*(1+(0.032*N) + 0.58*((T_wall/T_free_stream)-1)))

def rk4(f, x, y0, N, P, T, A, dQdx, dFdx, dAdx, T_star):
    #4th-order Runge-Kutta method
    #Input your function (f_N, f_P, or f_T), x array, initial value (@ x=0), N array (if possible), P, T, Area, dQdx, dFdx, dAdx, and reference temp
    #Outputs either N(x), P(x), or T(x) depending on input function f
    n = len(x) #length of array
    y_list = []
    x0 = x[0]
    for i in range(n-1):
        h = (x[n-1])/n #even step size
        k1 = h * (f(x[i], y0, N[i], P[i], T[i], A[i], dQdx[i], dFdx[i], dAdx[i], T_star[i])) #Technically, for greater accuracy, change N, P, T, dQdx, dFdx, T* in k2, k3, k4
        k2 = h * (f((x[i]+h/2), (y0+k1/2), N[i], P[i], T[i], A[i], dQdx[i], dFdx[i], dAdx[i], T_star[i]))
        k3 = h * (f((x[i]+h/2), (y0+k2/2), N[i], P[i], T[i], A[i], dQdx[i], dFdx[i], dAdx[i], T_star[i]))
        k4 = h * (f((x[i]+h), (y0+k3), N[i], P[i], T[i], A[i], dQdx[i], dFdx[i], dAdx[i], T_star[i]))
        k = (k1+2*k2+2*k3+k4)/6
        yn = y0 + k
        y_list.append(yn) #appending newly calculated y-value to y-list
        y0 = yn  #incrementing y
    return y_list

#Function defining dNdx used for RK4 for calculating N
def f_N(x, N, not_used, P, T, A, dQdx, dFdx, dAdx, T_star):
    gamma = calc_gamma(T_star)
    cp = calc_cp(T_star)
    term1 = (N/(1-N))*((1+gamma*N)/(cp*T))*(dQdx)
    term2 = (N/(1-N))*((2+(gamma-1)*N)/(Rs*T))*(dFdx)
    term3 = -(N/(1-N))*((2+(gamma-1)*N)/A)*(dAdx)
    return term1+term2+term3

#Function defining dPdx used for RK4 for calculating P
def f_P(x, P, N, not_used, T, A, dQdx, dFdx, dAdx, T_star):
    gamma = calc_gamma(T_star)
    cp = calc_cp(T_star)
    term1 = -(P/(1-N))*((gamma*N)/(cp*T))*(dQdx)
    term2 = -(P/(1-N))*((1+(gamma-1)*N)/(Rs*T))*(dFdx)
    term3 = (P/(1-N))*((gamma*N)/A)*(dAdx)
    return term1+term2+term3

#Function defining dTdx used for RK4 for calculating T
def f_T(x, T, N, P, not_used, A, dQdx, dFdx, dAdx, T_star):
    gamma = calc_gamma(T_star)
    cp = calc_cp(T_star)
    term1 = (T/(1-N))*((1-gamma*N)/(cp*T))*(dQdx)
    term2 = -(T/(1-N))*(((gamma-1)*N)/(Rs*T))*(dFdx)
    term3 = (T/(1-N))*(((gamma-1)*N)/A)*(dAdx)
    return term1+term2+term3

#Calculate cp based on reference temperature T*
def calc_cp(T_star):
    CO2_under_1000 = [2.35677352, -.00898459677, -0.00000712356269, 0.00000000245919022, -0.000000000000143699548]
    H2O_under_1000 = [4.19864056, -0.0020364341, 0.00000652040211, -0.00000000548797062, 0.00000000000177197817]
    O2_under_1000 = [3.78245636, -0.00299673415, 0.000009847302, -0.00000000968129508, 0.00000000000324372836]
    CO2 = [4.63659493, 0.00274131991, -0.000000995828531, -0.000000000160373011, -0.00000000000000916103468]
    H2O = [2.67703787, 0.00297318329, -.00000077376969, .0000000000944336689, -.00000000000000426900959]
    O2 = [3.66096083,  .000656365523, -.000000141149485,  .0000000000205797658, -0.00000000000000129913248]
    if T_star <1000:
       CO2_a = CO2_under_1000
       H2O_a = H2O_under_1000
       O2_a = O2_under_1000
       cp_CO2 = CO2_a[0] + CO2_a[1]*T_star + CO2_a[2]*(T_star**2) + CO2_a[3]*(T_star**3) + CO2_a[4]*(T_star**4)
       cp_H2O = H2O_a[0] + H2O_a[1]*T_star + H2O_a[2]*(T_star**2) + H2O_a[3]*(T_star**3) + H2O_a[4]*(T_star**4)
       cp_O2 = O2_a[0] + O2_a[1]*T_star + O2_a[2]*(T_star**2) + O2_a[3]*(T_star**3) + O2_a[4]*(T_star**4)
    else:
       CO2_a = CO2
       H2O_a = H2O
       O2_a = O2
       cp_CO2 = CO2_a[0] + CO2_a[1]*T_star + CO2_a[2]*(T_star**2) + CO2_a[3]*(T_star**3) + CO2_a[4]*(T_star**4)
       cp_H2O = H2O_a[0] + H2O_a[1]*T_star + H2O_a[2]*(T_star**2) + H2O_a[3]*(T_star**3) + H2O_a[4]*(T_star**4)
       cp_O2 = O2_a[0] + O2_a[1]*T_star + O2_a[2]*(T_star**2) + O2_a[3]*(T_star**3) + O2_a[4]*(T_star**4)
    cp = (cp_CO2*mass_frac_CO2 + cp_H2O*mass_frac_H20 + cp_O2*mass_fracO2)
    return cp

#Calculate viscosity and thermal conductivity based on reference temperature T*
def calc_viscosity_and_lambda(T_star):
    #Define each chemical species as an array with values in the following order: [mole fraction, molar mass, thermal conductivity coefficients (A, B, C, D), viscosity coefficients (A, B, C, D)]
    #Coefficients in micropoise converted to poise
    visc_coef_CO2_low = np.array([0.70122551, 5.1717887, -1424.0838, 1.2895991])*(10**-7)
    visc_coef_H2O_low = np.array([0.50019557, -697.12796, 88163.892, 3.0836508])*(10**-7)
    visc_coef_O2_low = np.array([0.60916180, -52.244847, -599.74009, 2.0410801])*(10**-7)
    visc_coef_CO2_high = np.array([0.63978285, -42.637076, -15522.605, 1.6628843])*(10**-7)
    visc_coef_H2O_high = np.array([0.58988538, -537.69814, 54263.513, 2.3386375])*(10**-7)
    visc_coef_O2_high = np.array([0.72216486, 175.50839, -57974.816, 1.0901044])*(10**-7)

    #Coefficients in microwatts per centimeter Kelvin converted to watts per meter Kelvin
    tc_coef_CO2_low = np.array([0.48056568, -507.86720, 35088.811, 3.6747794])/10
    tc_coef_H2O_low = np.array([1.0966389, -555.13429, 106234.08, -0.24664550])/10
    tc_coef_O2_low = np.array([0.77229167, 6.8463210, -5893.3377, 1.2210365])/10
    tc_coef_CO2_high = np.array([0.69857277, -118.30477, -50688.859, 1.8650551])/10
    tc_coef_H2O_high = np.array([0.39367933, -2252.4226, 612174.58, 5.8011317])/10
    tc_coef_O2_high = np.array([0.90917351, 291.24182, 79650.171, 0.064851631])/10

    if T_star <1000:
      CO2 = [mol_frac_CO2, MM_CO2, tc_coef_CO2_low[0], tc_coef_CO2_low[1], tc_coef_CO2_low[2], tc_coef_CO2_low[3], visc_coef_CO2_low[0], visc_coef_CO2_low[1], visc_coef_CO2_low[2], visc_coef_CO2_low[3]]
      H2O = [mol_frac_H2O, MM_H2O, tc_coef_H2O_low[0], tc_coef_H2O_low[1], tc_coef_H2O_low[2], tc_coef_H2O_low[3], visc_coef_H2O_low[0], visc_coef_H2O_low[1], visc_coef_H2O_low[2], visc_coef_H2O_low[3]]
      O2 = [mol_frac_O2, MM_O2, tc_coef_O2_low[0], tc_coef_O2_low[1], tc_coef_O2_low[2], tc_coef_O2_low[3], visc_coef_O2_low[0], visc_coef_O2_low[1], visc_coef_O2_low[2], visc_coef_O2_low[3]]
    elif T_star >=1000:
      CO2 = [mol_frac_CO2, MM_CO2, tc_coef_CO2_high[0], tc_coef_CO2_high[1], tc_coef_CO2_high[2], tc_coef_CO2_high[3], visc_coef_CO2_high[0], visc_coef_CO2_high[1], visc_coef_CO2_high[2], visc_coef_CO2_high[3]]
      O2 = [mol_frac_O2, MM_O2, tc_coef_O2_high[0], tc_coef_O2_high[1], tc_coef_O2_high[2], tc_coef_O2_high[3], visc_coef_O2_high[0], visc_coef_O2_high[1], visc_coef_O2_high[2], visc_coef_O2_high[3]]
      if T_star < 1073.2:
        H2O = [mol_frac_H2O, MM_H2O, tc_coef_H2O_low[0], tc_coef_H2O_low[1], tc_coef_H2O_low[2], tc_coef_H2O_low[3], visc_coef_H2O_low[0], visc_coef_H2O_low[1], visc_coef_H2O_low[2], visc_coef_H2O_low[3]]
      else:
        H2O = [mol_frac_H2O, MM_H20, tc_coef_H2O_high[0], tc_coef_H2O_high[1], tc_coef_H2O_high[2], tc_coef_H2O_high[3], visc_coef_H2O_high[0], visc_coef_H2O_high[1], visc_coef_H2O_high[2], visc_coef_H2O_high[3]]

    all_species = [CO2, H2O, O2]

    #Initializing arrays and mix values
    lambda_species = []
    viscosity_species = []

    lambda_mix = 0
    viscosity_mix = 0
    for i in all_species:
      viscosity_species[i] = i[6]*np.log(T_star) + i[7]/T_star + i[8]/(T_star**2) + i[9]
      lambda_species[i] = i[2]*np.log(T_star) + i[3]/T_star + i[4]/(T_star**2) + i[5]

    for i in range(len(all_species)):
      sum_viscosity = 0
      sum_lambda = 0
      for j in range(len(all_species)):
        if j != i:
          phi_ij = (1/4)*((1 + ((viscosity_species[i]/viscosity_species[j])**(1/2))*((all_species[j][1]/all_species[i][1])**(1/4)))**2)*(((2*all_species[j][1])/(all_species[j][1]+all_species[i][1]))**(1/2))
          psi_ij = phi_ij*(1+((2.41*(all_species[i][1]-all_species[j][1])*(all_species[i][1]-0.142*all_species[j][1]))/((all_species[i][1]+all_species[j][1])**2)))
          sum_viscosity += all_species[j][0]*phi_ij
          sum_lambda += all_species[j][0]*psi_ij
      viscosity_mix += ((all_species[i][0]*viscosity_species[i])/(all_species[i][0] + sum_viscosity))
      lambda_mix += ((all_species[i][0]*lambda_species[i])/(all_species[i][0] + sum_viscosity))

    return viscosity_mix, lambda_mix

def calc_hg(viscosity, cp, c_star, A, Pr, T_wall, T_stag, P, N, gamma):
  C = 0.026 #Generally accepted constant for Bartz
  sigma = (((T_wall/(2*T_stag))*(1+((gamma-1)/2)*N) + 0.5)**(-0.68))*((1+((gamma-1)/2)*N)**(-0.12))
  hg = ((C/(D_t**0.2))*(((viscosity**0.2)*cp)/(Pr**0.6))*((P**0.8)/c_star)*((D_t**0.1)/R))*((A_t/A)**0.9)*sigma
  return hg

def calc_f(epselon, A, density, N, viscosity, c):
  r = A/(2*np.pi*dx)
  Rh = np.sqrt(A/np.pi)/2
  V = np.sqrt(N)*c
  Re = density*V*(r*2)/viscosity
  a = 2.51/Re
  b = (epselon)/(14.8*Rh)
  f = 1/((((2*lambertw((np.log(10)/(2*a))*(10**(b/(2*a)))))/np.log(10)) - (b/a))**2)
  return f

def calc_Taw(T, N, Pr, gamma):
  T_stag = T*(1+((gamma-1)/2)*N)
  T_aw = T_stag*((1+(Pr**0.33)*((gamma-1)/2)*N)/(1+((gamma-1)/2)*N))
  return T_stag, T_aw

def calc_dFdx(f, N, T, A, gamma):
  L = dx
  c = np.sqrt((gamma*Ru*T)/MM_total)
  V = np.sqrt(N)*c
  D = A/(np.pi*dx)
  F = (f*L*(V**2))/(2*D)
  dFdx = np.gradient(F, dx)
  return dFdx



Run_Heat_Transfer(T_wall0, T_free_stream0, M_0, area, dAdx)
