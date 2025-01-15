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
import pandas as pd
from scipy.special import lambertw
import matplotlib.pyplot as plt
import time

Ru = 8.31446261815324 #Universal gas constant (J/mol K)
MM_CO2 = 0.044009 #Molar mass (kg/mol)
MM_H2O = 0.01801528 #Molar mass (kg/mol)
MM_O2 = 0.032004 #Molar mass (kg/mol)
MM_total = MM_CO2 + MM_H2O + MM_O2 #(kg/mol)
Rs = Ru/MM_total #Specific gas constant of entire mixture (J/kg K)

"""
TODO:

- T_star not aligning with T_free_stream. Look into better reference temperature equations
- The big problem is in dFdx
  - Run cryo rocket contour and initial values

"""

class Engine:
  def __init__(self, geometry, conditions_initial, flow_props, chem_props):
    self.x = geometry[0] #m
    self.radius = geometry[1] #m
    self.Rc_t = geometry[2] #m
    self.x -= self.x[0]
    self.T_wall0, self.T_free_stream0, self.P_0, self.M_0 = conditions_initial
    self.epsilon, self.c_star, self.m_dot = flow_props
    self.mass_frac_CO2, self.mass_frac_H2O, self.mass_frac_O2, self.mol_frac_CO2, self.mol_frac_H2O, self.mol_frac_O2 = chem_props
    self.dx = self.x[1]-self.x[0] #(m)
    self.area = 2*np.pi*self.radius*self.dx #(m^2)
    self.dAdx = np.gradient(self.area, self.dx) #(m^2/m)
    self.D_t = np.min(self.radius)*2 #Throat diameter (m)
    self.A_t = np.min(self.area) #Throat area (m)
    self.cp = self.calc_cp(self.calc_T_star(self.T_free_stream0, self.M_0**2, self.T_wall0))
    self.gamma = self.cp/(self.cp-Rs)
    self.c = np.sqrt(self.gamma*Rs*self.T_free_stream0)
    #print(self.calc_T_star(self.T_free_stream0, M0**2, T_wall0), self.cp, self.gamma)

  #We cannot assume N=1 exactly at throat because it breaks the diff. eq because there is (N-1) term in denominator
  def Run_Heat_Transfer(self):
    #initializing variables
    dQdx = np.zeros(len(self.x)) #Derivative of heat with respect to length (J/m)
    dFdx = np.zeros(len(self.x)) #Derivative of energy lost to friction with respect to length (J/m)
    T_wall = np.zeros(len(self.x)) + self.T_wall0 #Does not change unless there is a time change. Everything else should iteratively converge (K)
    T_free_stream = np.zeros(len(self.x)) + self.T_free_stream0 #Temperature of the free stream of gas(K)
    N = np.zeros(len(self.x)) + (self.M_0**2) #Initializing static N based on initial guess at injector (Unitless)
    P = np.zeros(len(self.x)) + self.P_0 #Initial pressure guess for the rk4 algorithm (Pa)
    T_star = self.calc_T_star(T_free_stream, N, T_wall) #Reference temperature for calculating transport properties
    #cp = self.calc_cp(T_star)
    #gamma = cp/(cp-Rs)
    #Defining these terms for the error function later on. They will be updated and modified in the error function
    T_free_stream_last = np.array([T_free_stream, T_free_stream, T_free_stream])
    T_wall_last = np.array([T_wall, T_wall, T_wall])
    q_last = np.array([np.zeros(len(self.x)),np.zeros(len(self.x)),np.zeros(len(self.x))])
    hg_last = np.array([np.zeros(len(self.x)),np.zeros(len(self.x)),np.zeros(len(self.x))])
    N_last = np.array([N, N, N])
    P_last = np.array([P, P, P])

    # Initialize the plot data
    q_diff_plot = []
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'b-', label="Live Data")  # Initialize an empty line
    ax.set_xlim(0, 50)  # Adjust as needed
    ax.set_ylim(0, 150000)  # Adjust as needed
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Heat Flux Residual")
    ax.legend()
    count = 0
    #Running loop. This will end when our percent error is "good enough" or, as defined currently, after 200 iterations
    for i in range(20):
      #Here we use a fourth-order Runge-Kutta algorithm to calculate N, P, and T based on each other and the other variables
      N = np.array(self.rk4(self.f_N, self.x, N[0], N, P, T_free_stream, self.area, dQdx, dFdx, self.dAdx)) #(Dimensionless)
      P = np.array(self.rk4(self.f_P, self.x, P[0], N, P, T_free_stream, self.area, dQdx, dFdx, self.dAdx))
      T_free_stream = np.array(self.rk4(self.f_T, self.x, T_free_stream[0], N, P, T_free_stream, self.area, dQdx, dFdx, self.dAdx))

      #Now we use those terms to calculate flow properties
      T_star = self.calc_T_star(T_free_stream, N, T_wall)

      viscosity, thermal_conductivity = self.calc_viscosity_and_lambda(T_free_stream)#T_star)
      Pr = (self.cp*viscosity)/thermal_conductivity

      
      density = P/(Rs*T_free_stream) #Derived from ideal gas law
      #m_dot = np.average((density/(self.area*self.dx))*(np.sqrt(N)*c))
      f = self.calc_f(self.epsilon, self.area, density, N, viscosity)
      #etc, etc, whatever we need here (i.e. density, Re, f, and everything needed for hg)
      T_stag, T_aw = self.calc_Taw(T_free_stream, N, Pr)
      hg = self.calc_hg(viscosity, self.c_star, self.area, Pr, T_wall, T_stag, P, N)
      q = hg*self.area*(T_aw-T_wall)
      #Get ready to calculate dQdx and dFdx here
      dQdx = q*self.area/self.m_dot
      dFdx = self.calc_dFdx(f, N, T_free_stream, self.area)
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

      q_diff_plot.append(q_diff) #Updating plot
      line.set_xdata(range(len(q_diff_plot)))  # Update x-axis with the indices
      line.set_ydata(q_diff_plot)  # Update y-axis with the data

      # Rescale axes if necessary
      ax.relim()
      ax.autoscale_view()

      # Redraw the plot
      fig.canvas.draw()
      fig.canvas.flush_events()
      count += 1
      print(count)
      #time.sleep(0.2)  # Pause to simulate computation time
    #We could make it return other things here if desired. We could also create a table of values and export to PDF
    
    plt.ioff()  # Turn off interactive mode
    plt.show()  # Keep the plot open after the loop ends
    return np.sqrt(N), viscosity, thermal_conductivity, T_free_stream, P, q, hg, T_star, dFdx, f


  def calc_T_star(self, T_free_stream, N, T_wall):
    #Reference temperature for finding flow properties based on T_free_stream, T_wall, and mach number
    T_star = (T_free_stream*(1+(0.032*N) + 0.58*((T_wall/T_free_stream)-1)))
    return np.array(T_star)

  def rk4(self, f, x, y0, N, P, T, A, dQdx, dFdx, dAdx):
    #4th-order Runge-Kutta method
    #Input your function (f_N, f_P, or f_T), x array, initial value (@ x=0), N array (if possible), P, T, Area, dQdx, dFdx, dAdx, and reference temp
    #Outputs either N(x), P(x), or T(x) depending on input function f
    
    n = len(x) #length of array
    y_list = []
    for i in range(n):
      h = (x[n-1])/n #even step size
      #We need to interpolate all of the values to correspond with our different x values
      x1 = x[i]
      x2 = x[i]+h/2 #x2=x3 so just repeat it
      x4 = x[i]+h

      N1 = N[i]
      N2 = np.interp(x2, x, N)
      N4 = np.interp(x4, x, N)

      P1 = P[i]
      P2 = np.interp(x2, x, P)
      P4 = np.interp(x4, x, P)

      T1 = T[i]
      T2 = np.interp(x2, x, T)
      T4 = np.interp(x4, x, T)

      A1 = A[i]
      A2 = np.interp(x2, x, A)
      A4 = np.interp(x4, x, A)

      dQdx1 = dQdx[i]
      dQdx2 = np.interp(x2, x, dQdx)
      dQdx4 = np.interp(x4, x, dQdx)

      dFdx1 = dFdx[i]
      dFdx2 = np.interp(x2, x, dFdx)
      dFdx4 = np.interp(x4, x, dFdx)
      
      dAdx1 = dAdx[i]
      dAdx2 = np.interp(x2, x, dAdx)
      dAdx4 = np.interp(x4, x, dAdx)

      k1 = h * (f(x1, y0, N1, P1, T1, A1, dQdx1, dFdx1, dAdx1))
      k2 = h * (f(x2, (y0+k1/2), N2, P2, T2, A2, dQdx2, dFdx2, dAdx2))
      k3 = h * (f(x2, (y0+k2/2), N2, P2, T2, A2, dQdx2, dFdx2, dAdx2))
      k4 = h * (f(x4, (y0+k3), N4, P4, T4, A4, dQdx4, dFdx4, dAdx4))
      k = (k1+2*k2+2*k3+k4)/6
      yn = y0 + k
      y_list.append(yn) #appending newly calculated y-value to y-list
      y0 = yn  #incrementing y
    return y_list

  #Function defining dNdx used for RK4 for calculating N
  def f_N(self, x, N, not_used, P, T, A, dQdx, dFdx, dAdx):    
    term1 = 0#(N/(1-N))*((1+self.gamma*N)/(self.cp*T))*(dQdx)
    term2 = 0#(N/(1-N))*((2+(self.gamma-1)*N)/(Rs*T))*(dFdx)
    term3 = -(N/(1-N))*((2+(self.gamma-1)*N)/A)*(dAdx)
    return term1+term2+term3

  #Function defining dPdx used for RK4 for calculating P
  def f_P(self, x, P, N, not_used, T, A, dQdx, dFdx, dAdx):
    term1 = 0#-(P/(1-N))*((self.gamma*N)/(self.cp*T))*(dQdx)
    term2 = 0#-(P/(1-N))*((1+(self.gamma-1)*N)/(Rs*T))*(dFdx)
    term3 = (P/(1-N))*((self.gamma*N)/A)*(dAdx)
    return term1+term2+term3

  #Function defining dTdx used for RK4 for calculating T
  def f_T(self, x, T, N, P, not_used, A, dQdx, dFdx, dAdx):
    term1 = 0#(T/(1-N))*((1-self.gamma*N)/(self.cp*T))*(dQdx)
    term2 = 0#-(T/(1-N))*(((self.gamma-1)*N)/(Rs*T))*(dFdx)
    term3 = (T/(1-N))*(((self.gamma-1)*N)/A)*(dAdx)
    return term1+term2+term3

  #Calculate cp based on reference temperature T*
  def calc_cp(self, T_star):
    CO2_under_1000 = [2.35677352, -.00898459677, -0.00000712356269, 0.00000000245919022, -0.000000000000143699548]
    H2O_under_1000 = [4.19864056, -0.0020364341, 0.00000652040211, -0.00000000548797062, 0.00000000000177197817]
    O2_under_1000 = [3.78245636, -0.00299673415, 0.000009847302, -0.00000000968129508, 0.00000000000324372836]
    CO2 = [4.63659493, 0.00274131991, -0.000000995828531, -0.000000000160373011, -0.00000000000000916103468]
    H2O = [2.67703787, 0.00297318329, -.00000077376969, .0000000000944336689, -.00000000000000426900959]
    O2 = [3.66096083,  .000656365523, -.000000141149485,  .0000000000205797658, -0.00000000000000129913248]
    RsCO2 = Ru/MM_CO2
    RsH2O = Ru/MM_H2O
    RsO2 = Ru/MM_O2

    if T_star <1000:
      CO2_a = CO2_under_1000
      H2O_a = H2O_under_1000
      O2_a = O2_under_1000
    else:
      CO2_a = CO2
      H2O_a = H2O
      O2_a = O2
    cp_CO2 = RsCO2*(CO2_a[0] + CO2_a[1]*T_star + CO2_a[2]*(T_star**2) + CO2_a[3]*(T_star**3) + CO2_a[4]*(T_star**4))
    cp_H2O = RsH2O*(H2O_a[0] + H2O_a[1]*T_star + H2O_a[2]*(T_star**2) + H2O_a[3]*(T_star**3) + H2O_a[4]*(T_star**4))
    cp_O2 = RsO2*(O2_a[0] + O2_a[1]*T_star + O2_a[2]*(T_star**2) + O2_a[3]*(T_star**3) + O2_a[4]*(T_star**4))
    cp = (cp_CO2*self.mass_frac_CO2 + cp_H2O*self.mass_frac_H2O + cp_O2*self.mass_frac_O2)

    return cp

  #Calculate viscosity and thermal conductivity based on reference temperature T*
  def calc_viscosity_and_lambda(self, T_star):
    #Define each chemical species as an array with values in the following order: [mole fraction, molar mass, thermal conductivity coefficients (A, B, C, D), viscosity coefficients (A, B, C, D)]
    #Coefficients to lead to viscosity in micropoise. Units adjusted to poise in the return line
    visc_coef_CO2_low = np.array([0.70122551, 5.1717887, -1424.0838, 1.2895991])
    visc_coef_H2O_low = np.array([0.50019557, -697.12796, 88163.892, 3.0836508])
    visc_coef_O2_low = np.array([0.60916180, -52.244847, -599.74009, 2.0410801])
    visc_coef_CO2_high = np.array([0.63978285, -42.637076, -15522.605, 1.6628843])
    visc_coef_H2O_high = np.array([0.58988538, -537.69814, 54263.513, 2.3386375])
    visc_coef_O2_high = np.array([0.72216486, 175.50839, -57974.816, 1.0901044])

    #Coefficients in microwatts per centimeter Kelvin. Units converted to watts per meter Kelvin in return line
    tc_coef_CO2_low = np.array([0.48056568, -507.86720, 35088.811, 3.6747794])
    tc_coef_H2O_low = np.array([1.0966389, -555.13429, 106234.08, -0.24664550])
    tc_coef_O2_low = np.array([0.77229167, 6.8463210, -5893.3377, 1.2210365])
    tc_coef_CO2_high = np.array([0.69857277, -118.30477, -50688.859, 1.8650551])
    tc_coef_H2O_high = np.array([0.39367933, -2252.4226, 612174.58, 5.8011317])
    tc_coef_O2_high = np.array([0.90917351, 291.24182, 79650.171, 0.064851631])

    thermal_conductivity = []
    viscosity = []

    for temp in T_star:
      if temp <1000:
        CO2 = [self.mol_frac_CO2, MM_CO2, tc_coef_CO2_low[0], tc_coef_CO2_low[1], tc_coef_CO2_low[2], tc_coef_CO2_low[3], visc_coef_CO2_low[0], visc_coef_CO2_low[1], visc_coef_CO2_low[2], visc_coef_CO2_low[3]]
        H2O = [self.mol_frac_H2O, MM_H2O, tc_coef_H2O_low[0], tc_coef_H2O_low[1], tc_coef_H2O_low[2], tc_coef_H2O_low[3], visc_coef_H2O_low[0], visc_coef_H2O_low[1], visc_coef_H2O_low[2], visc_coef_H2O_low[3]]
        O2 = [self.mol_frac_O2, MM_O2, tc_coef_O2_low[0], tc_coef_O2_low[1], tc_coef_O2_low[2], tc_coef_O2_low[3], visc_coef_O2_low[0], visc_coef_O2_low[1], visc_coef_O2_low[2], visc_coef_O2_low[3]]
      else:
        CO2 = [self.mol_frac_CO2, MM_CO2, tc_coef_CO2_high[0], tc_coef_CO2_high[1], tc_coef_CO2_high[2], tc_coef_CO2_high[3], visc_coef_CO2_high[0], visc_coef_CO2_high[1], visc_coef_CO2_high[2], visc_coef_CO2_high[3]]
        O2 = [self.mol_frac_O2, MM_O2, tc_coef_O2_high[0], tc_coef_O2_high[1], tc_coef_O2_high[2], tc_coef_O2_high[3], visc_coef_O2_high[0], visc_coef_O2_high[1], visc_coef_O2_high[2], visc_coef_O2_high[3]]
        if temp < 1073.2:
          H2O = [self.mol_frac_H2O, MM_H2O, tc_coef_H2O_low[0], tc_coef_H2O_low[1], tc_coef_H2O_low[2], tc_coef_H2O_low[3], visc_coef_H2O_low[0], visc_coef_H2O_low[1], visc_coef_H2O_low[2], visc_coef_H2O_low[3]]
        else:
          H2O = [self.mol_frac_H2O, MM_H2O, tc_coef_H2O_high[0], tc_coef_H2O_high[1], tc_coef_H2O_high[2], tc_coef_H2O_high[3], visc_coef_H2O_high[0], visc_coef_H2O_high[1], visc_coef_H2O_high[2], visc_coef_H2O_high[3]]
      all_species = [CO2, H2O, O2]
      #Initializing arrays and mix values
      lambda_species = []
      viscosity_species = []

      lambda_mix = 0
      viscosity_mix = 0
      for i in range(len(all_species)):
        viscosity_species.append(all_species[i][6]*np.log(temp) + all_species[i][7]/temp + all_species[i][8]/(temp**2) + all_species[i][9])
        lambda_species.append(all_species[i][2]*np.log(temp) + all_species[i][3]/temp + all_species[i][4]/(temp**2) + all_species[i][5])

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
        lambda_mix += ((all_species[i][0]*lambda_species[i])/(all_species[i][0] + sum_lambda))
      thermal_conductivity.append(lambda_mix)
      viscosity.append(viscosity_mix)
    return np.array(viscosity)*(10**-7), np.array(thermal_conductivity)/10

  def calc_hg(self, viscosity, c_star, A, Pr, T_wall, T_stag, P, N):
    C = 0.026 #Generally accepted constant for Bartz
    sigma = (((T_wall/(2*T_stag))*(1+((self.gamma-1)/2)*N) + 0.5)**(-0.68))*((1+((self.gamma-1)/2)*N)**(-0.12))
    hg = ((C/(self.D_t**0.2))*(((viscosity**0.2)*self.cp)/(Pr**0.6))*((P**0.8)/c_star)*((self.D_t**0.1)/self.Rc_t))*((self.A_t/A)**0.9)*sigma
    return hg

  def calc_f(self, epsilon, A, density, N, viscosity):
    r = A/(2*np.pi*self.dx)
    D = 2*r
    V = np.sqrt(N)*self.c
    Re = density*V*D/viscosity
    #Using the Haaland Equation
    f = (1/(-1.8*np.log10((6.9/Re)+((epsilon/(3.7*D))**1.11))))**2

    #We cannot use this explicit method of solving for Darcy friction coefficient because the 10**(b/(2*a)) term causes an overflow error.
    #Rh = np.sqrt(A/np.pi)/2
    #a = 2.51/Re
    #b = (epsilon)/(14.8*Rh)
    #f = 1/((((2*lambertw((np.log(10)/(2*a))*(10**(b/(2*a)))))/np.log(10)) - (b/a))**2)

    return f

  def calc_Taw(self, T, N, Pr):
    T_stag = T*(1+((self.gamma-1)/2)*N)
    T_aw = T_stag*((1+(Pr**0.33)*((self.gamma-1)/2)*N)/(1+((self.gamma-1)/2)*N))
    return T_stag, T_aw

  def calc_dFdx(self, f, N, T, A):
    L = self.dx
    V = np.sqrt(N)*self.c
    D = A/(np.pi*self.dx)
    F = (f*L*(V**2))/(2*D)
    dFdx = np.gradient(F, self.dx)
    return dFdx
"""
The Engine class intakes many parameters, split up into four categories: geometry, initial conditions, flow properties, and chemical properties
The contents of each of these lists are as follows
Geometry: x (array), radius (array), radius of curvature at the throat (float)
Initial conditions: Initial temperature of the wall (float), temperature estimate at injector face (float), pressure estimate at injector face (float), mach number estimate at injector face (float)
Flow properties: Absolute roughness of the contour (float), characteristic velocity (float), mdot (float)
Chemical properties: Mass fraction of CO2, H2O, and O2, and mole fraction of CO2, H2O, and O2 (6-array)
"""

contour = pd.read_csv("engine_contour_test7.csv")
x = np.array(contour["x"])*0.0254 #in to m
radius = np.array(contour["y"])*0.0254 #in to m
R_ct = 0.625*0.0254 #in to m
T_wall0 = 800#293 #Room temp in K
T0 = 3284 #K
P0 = 220*6894.75729 #psi to Pa
M0 = 0.09
epsilon = 0.0015
c_star = 1837.8 #m/s
mdot = 0.366 #kg/s

geometries = [x, radius, R_ct]
conditions_initial = [T_wall0, T0, P0, M0]
flow_props = [epsilon, c_star, mdot]
chem_props = [MM_CO2/MM_total, MM_H2O/MM_total, MM_O2/MM_total, 0.25, 0.5, 0.25]

Blip = Engine(geometries, conditions_initial, flow_props, chem_props)

finalMach, final_viscosity, final_thermal_conductivity, final_temp, final_pressure, final_q, final_hg, final_T_star, final_dFdx, f = Blip.Run_Heat_Transfer()

fig, axes = plt.subplots(3, 3, figsize=(12, 12))  # 3 rows, 3 column

# First Row
# First subplot
axes[0, 0].plot(x*1000, finalMach, color='red')
axes[0, 0].set_title("Mach vs Length")
axes[0, 0].set_xlabel("Axial distance (mm)")
axes[0, 0].set_ylabel("Mach Number")

# Second subplot
axes[1, 0].plot(x*1000, final_temp, color='green')
axes[1, 0].set_title("T vs Length")
axes[1, 0].set_xlabel("Axial distance (mm)")
axes[1, 0].set_ylabel("Temperature (K)")


# Third subplot
axes[2, 0].plot(x*1000, final_pressure/1000, color='blue')
axes[2, 0].set_title("Pressure vs Length")
axes[2, 0].set_xlabel("Axial distance (mm)")
axes[2, 0].set_ylabel("Pressure (kPa)")


# Second Row
# First subplot
axes[0, 1].plot(x*1000, final_viscosity*1000, color='red')
axes[0, 1].set_title("Viscosity vs Length")
axes[0, 1].set_xlabel("Axial distance (mm)")
axes[0, 1].set_ylabel("Viscosity (micropoise)")

# Second subplot
axes[1, 1].plot(x*1000, radius*1000, color='green')
axes[1, 1].set_title("Contour")
axes[1, 1].set_xlabel("Axial distance (mm)")
axes[1, 1].set_ylabel("Radius (mm)")

# Third subplot
axes[2, 1].plot(x*1000, final_thermal_conductivity, color='blue')
axes[2, 1].set_title("Lambda vs Length")
axes[2, 1].set_xlabel("Axial distance (mm)")
axes[2, 1].set_ylabel("Thermal conductivity (Watts/K m)")

#Third Row
# First subplot
axes[0, 2].plot(x*1000, final_dFdx, color='red')
axes[0, 2].set_title("dFdx vs Length")
axes[0, 2].set_xlabel("Axial distance (mm)")
axes[0, 2].set_ylabel("dFdx")

# Second subplot
axes[1, 2].plot(x*1000, final_hg, color='green')
axes[1, 2].set_title("Heat Transfer Coefficient vs Length")
axes[1, 2].set_xlabel("Axial distance (mm)")
axes[1, 2].set_ylabel("Heat Transfer Coefficient")

# Third subplot
axes[2, 2].plot(x*1000, f, color='blue')
axes[2, 2].set_title("Friction Factor vs Length")
axes[2, 2].set_xlabel("Axial distance (mm)")
axes[2, 2].set_ylabel("Friction Factor")
# Adjust layout to prevent overlapping
plt.tight_layout()

# Show the plot
plt.show()