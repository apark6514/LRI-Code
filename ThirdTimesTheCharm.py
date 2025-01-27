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
from scipy.optimize import newton
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
- change rk4 algorithm to new influence coefficients
- change definitions of q and dQdx
- eliminate dFdx in favor of new friction accounting method
"""

class Engine:
  def __init__(self, geometry, conditions_initial, flow_props, chem_props):
    self.x = geometry[0] #m
    self.radius = geometry[1] #m
    self.Rc_t = geometry[2] #m
    self.x -= self.x[0]
    self.T_wall0, self.T_e0, self.P_0, self.M_0 = conditions_initial
    self.epsilon, self.c_star, self.m_dot = flow_props
    self.mass_frac_CO2, self.mass_frac_H2O, self.mass_frac_O2, self.mol_frac_CO2, self.mol_frac_H2O, self.mol_frac_O2 = chem_props
    self.dx = self.x[1]-self.x[0] #(m)
    self.area = np.pi*(self.radius**2) #(m^2)
    self.dAdx = np.gradient(self.area, self.dx) #(m^2/m)
    self.D_t = np.min(self.radius)*2 #Throat diameter (m)
    self.A_t = np.min(np.pi*(self.radius**2)) #Throat area (m)
    self.index_t = np.where(self.radius==np.min(self.radius))[0]
    self.cp = self.calc_cp(self.calc_T_star(self.T_e0, self.M_0**2, self.T_wall0))
    self.gamma = self.cp/(self.cp-Rs)
    self.c = np.sqrt(self.gamma*Rs*self.T_e0)

  #We cannot assume N=1 exactly at throat because it breaks the diff. eq because there is (N-1) term in denominator
  def Run_Heat_Transfer(self):
    #initializing variables
    T_wall = np.zeros(len(self.x)) + self.T_wall0 #Does not change unless there is a time change. Everything else should iteratively converge (K)
    q = np.zeros(len(self.x))
    T_e = np.zeros(len(self.x)) + self.T_e0 #Temperature of the free stream of gas(K)
    N = np.zeros(len(self.x)) + (self.M_0**2) #Initializing static N based on initial guess at injector (Unitless)
    P = np.zeros(len(self.x)) + self.P_0 #Initial pressure guess for the rk4 algorithm (Pa)
    T_star = self.calc_T_star(T_e, N, T_wall) #Reference temperature for calculating transport properties
    
    viscosity, thermal_conductivity = self.calc_viscosity_and_lambda(T_star)

    T_s = T_e*(1+((self.gamma-1)/2)*N)
    dTsdx = np.gradient(T_s, self.dx)
    P_s = P*(1+((self.gamma-1)/2)*N)**(self.gamma/(self.gamma-1))

    u = np.zeros(len(x)) + np.sqrt(N*self.gamma*Rs*T_e)
    density = np.zeros(len(x)) + P/(Rs*T_e)
    
    Re = density*u*x/viscosity
    Re[0] = 0.1

    Cf = 0.0576/(Re**0.2)
    #Defining these terms for the error function later on. They will be updated and modified in the error function
    T_e_last = np.array([T_e, T_e, T_e])
    T_wall_last = np.array([T_wall, T_wall, T_wall])
    q_last = np.array([np.ones(len(self.x)),np.ones(len(self.x)),np.ones(len(self.x))])
    hg_last = np.array([np.ones(len(self.x)),np.ones(len(self.x)),np.ones(len(self.x))])
    N_last = np.array([N, N, N])
    P_last = np.array([P, P, P])

    # Initialize the plot data
    q_diff_plot = []
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'b-', label="Live Data")  # Initialize an empty line
    ax.set_xlim(0, 5)  # Adjust as needed
    ax.set_ylim(0, 0.1)  # Adjust as needed
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Heat Flux Residual")
    ax.legend()
    count = 0
    #Running loop. This will end when our percent error is "good enough" or, as defined currently, after 200 iterations
    q_diff = 1
    while count < 20:
      
      #Thermodynamic Properties
      #We're using a 4th order Runge-Kutta algorithm to evaluate the partial differential equations for velocity (u), density, N, stagnation pressure, and free stream temperature
      u = np.sqrt(N*self.gamma*Rs*T_e)
      density = np.array(self.rk4(self.f_density, density[0], density, N, T_s, Cf, dTsdx))
      N = np.array(self.rkN(self.f_N, N[0], T_s, Cf, dTsdx))
      Re = density*u*self.x/viscosity
      Re[0] = 0.0000001
      Cf = 0.0576/(Re**0.2)
      P_s = np.array(self.rk4(self.f_Ps, P[0], density, N, T_s, Cf, dTsdx))
      T_e = np.array(self.rk4(self.f_T, T_e[0], density, N, T_s, Cf, dTsdx))

      #Flow properties
      T_star = self.calc_T_star(T_e, N, T_wall)
      viscosity, thermal_conductivity = self.calc_viscosity_and_lambda(T_star)
      Pr = (self.cp*viscosity)/thermal_conductivity
      T_aw = T_s*((1+(Pr**0.33)*((self.gamma-1)/2)*N)/(1+((self.gamma-1)/2)*N))

      #Heat transfer
      hg = self.calc_hg(viscosity, Pr, T_wall, T_s, P_s, N)
      q = hg*(2*np.pi*self.radius*self.dx)*(T_aw-T_wall)
      #Update stagnation temperature
      T_s = np.array(self.rk4(self.f_Ts, T_s[0], density, N, T_s, Cf, q)) #(Dimensionless)
      dTsdx = np.gradient(T_s, self.dx)

      #Residuals
      T_e_diff = np.average([np.abs(T_e-T_e_last[0]), np.abs(T_e-T_e_last[1]), np.abs(T_e-T_e_last[2])])/np.average(T_e_last)
      T_wall_diff = np.average([np.abs(T_wall-T_wall_last[0]), np.abs(T_wall-T_wall_last[1]), np.abs(T_wall-T_wall_last[2])])/np.average(T_wall_last)
      q_diff = np.average([np.abs(q-q_last[0]), np.abs(q-q_last[1]), np.abs(q-q_last[2])])/np.average(q_last)
      hg_diff = np.average([np.abs(hg-hg_last[0]), np.abs(hg-hg_last[1]), np.abs(hg-hg_last[2])])/np.average(hg_last)
      N_diff = np.average([np.abs(N-N_last[0]), np.abs(N-N_last[1]), np.abs(N-N_last[2])])/np.average(N_last)
      P_diff = np.average([np.abs(P-P_last[0]), np.abs(P-P_last[1]), np.abs(P-P_last[2])])/np.average(P_last)

      T_e_last = [T_e_last[1], T_e_last[2], T_e]
      T_wall_last = [T_wall_last[1], T_wall_last[2], T_wall]
      q_last = [q_last[1], q_last[2], q]
      hg_last = [hg_last[1], hg_last[2], hg]
      N_last = [N_last[1], N_last[2], N]
      P_last = [P_last[1], P_last[2], P]

      #Plotting
      q_diff_plot.append(q_diff) #Updating plot
      line.set_xdata(range(len(q_diff_plot)))  # Update x-axis with the indices
      line.set_ydata(q_diff_plot)  # Update y-axis with the data

      # Rescale axes if necessary
      ax.relim()
      ax.autoscale_view()

      # Redraw the plot
      fig.canvas.draw()
      fig.canvas.flush_events()
      count+=1
    plt.ioff()  # Turn off interactive mode
    plt.show()  # Keep the plot open after the loop ends
    
    return np.sqrt(N), T_s, P_s, T_e, hg, q, self.dAdx, dTsdx, viscosity, thermal_conductivity, Cf


  def calc_T_star(self, T_e, N, T_wall):
    #Reference temperature for finding flow properties based on T_e, T_wall, and mach number
    T_star = (T_e*(1+(0.032*N) + 0.58*((T_wall/T_e)-1)))
    return np.array(T_star)

  def rk4(self, f, y0, density, N, T_s, Cf, dTsdx):
    x = self.x
    n = len(x)
    y_list = []
    last = 0
    for i in range(n):
      h = (x[n-1])/n #even step size
      #We need to interpolate all of the values to correspond with our different x values
      x1 = x[i]
      x2 = x[i]+h/2 #x2=x3 so just repeat it
      x4 = x[i]+h

      d1 = density[i]
      d2 = np.interp(x2, x, density)
      d4 = np.interp(x4, x, density)

      N1 = N[i]
      N2 = np.interp(x2, x, N)
      N4 = np.interp(x4, x, N)

      T_s1 = T_s[i]
      T_s2 = np.interp(x2, x, T_s)
      T_s4 = np.interp(x4, x, T_s)

      Cf1 = Cf[i]
      Cf2 = np.interp(x2, x, Cf)
      Cf4 = np.interp(x4, x, Cf)

      A1 = self.area[i]
      A2 = np.interp(x2, x, self.area)
      A4 = np.interp(x4, x, self.area)
      
      dAdx1 = self.dAdx[i]
      dAdx2 = np.interp(x2, x, self.dAdx)
      dAdx4 = np.interp(x4, x, self.dAdx)

      dTsdx1 = dTsdx[i]
      dTsdx2 = np.interp(x2, x, dTsdx)
      dTsdx4 = np.interp(x4, x, dTsdx)

      k1 = h * (f(x1, y0, d1, N1, T_s1, Cf1, A1, dAdx1, dTsdx1, last))
      k2 = h * (f(x2, (y0+k1/2), d2, N2, T_s2, Cf2, A2, dAdx2, dTsdx2, last))
      k3 = h * (f(x2, (y0+k2/2), d2, N2, T_s2, Cf2, A2, dAdx2, dTsdx2, last))
      k4 = h * (f(x4, (y0+k3), d4, N4, T_s4, Cf4, A4, dAdx4, dTsdx4, last))
      k = (k1+2*k2+2*k3+k4)/6
      last = k/h
      yn = y0 + k
      y_list.append(yn) #appending newly calculated y-value to y-list
      y0 = yn  #incrementing y
    return y_list
  
  def f_density(self, x, d, not_used, N, T_s, Cf, A, dAdx, dTsdx, last):    
    dH = 2*(np.sqrt(A/np.pi))
    term1 = (N/A)*(dAdx)
    term2 = -((1+((self.gamma-1)/2)*N)/T_s)*(dTsdx)
    term3 = -((2*self.gamma*N)/dH)*Cf

    return (d/(1-N))*(term1+term2+term3)
  #Function defining dNdx used for RK4 for calculating N

  def rkN(self, f, y0, T_s, Cf, dTsdx):
    x = self.x
    n = len(x)
    y_list = []
    last = 0
    for i in range(n):
      h = (x[n-1])/n #even step size
      #We need to interpolate all of the values to correspond with our different x values
      x1 = x[i]
      x2 = x[i]+h/2 #x2=x3 so just repeat it
      x4 = x[i]+h

      T_s1 = T_s[i]
      T_s2 = np.interp(x2, x, T_s)
      T_s4 = np.interp(x4, x, T_s)

      Cf1 = Cf[i]
      Cf2 = np.interp(x2, x, Cf)
      Cf4 = np.interp(x4, x, Cf)

      A1 = self.area[i]
      A2 = np.interp(x2, x, self.area)
      A4 = np.interp(x4, x, self.area)
      
      dAdx1 = self.dAdx[i]
      dAdx2 = np.interp(x2, x, self.dAdx)
      dAdx4 = np.interp(x4, x, self.dAdx)

      dTsdx1 = dTsdx[i]
      dTsdx2 = np.interp(x2, x, dTsdx)
      dTsdx4 = np.interp(x4, x, dTsdx)

      k1 = h * (f(x1, y0, T_s1, Cf1, A1, dAdx1, dTsdx1, last))
      k2 = h * (f(x2, (y0+k1/2), T_s2, Cf2, A2, dAdx2, dTsdx2, last))
      k3 = h * (f(x2, (y0+k2/2), T_s2, Cf2, A2, dAdx2, dTsdx2, last))
      k4 = h * (f(x4, (y0+k3), T_s4, Cf4, A4, dAdx4, dTsdx4, last))
      k = (k1+2*k2+2*k3+k4)/6
      yn = y0 + k
      y_list.append(yn) #appending newly calculated y-value to y-list
      y0 = yn  #incrementing y
      last = np.min([k1, k2, k3, k4])/h
    return y_list

  def f_N(self, x, N, T_s, Cf, A, dAdx, dTsdx, last):    
    if x > self.x[self.index_t - 10] and x <= self.x[self.index_t]:
      E = (self.gamma+1)/(self.gamma-1)
      P = (E-1)/E
      Q = 1/E
      if N < 1:
        term1 = 2*((1/self.A_t)**2)*A*dAdx*N
        term2 = ((P+Q*N)**E-1) - (A/self.A_t)**2
        dNdx = term1/term2
      else:
        X = 1/N
        R = (A/self.A_t)**(2*Q/P)
        Rp = (2*Q/P)*((1/self.A_t)**(2*Q/P))*(A**((2*Q/P)-1))*dAdx
        dNdx = (-N*Rp)/(((P*X+Q)**((1-P)/P)) - R)
      if dNdx < last:
        dNdx = last
    else:
      dH = 2*(np.sqrt(A/np.pi))
      term1 = -(2*(1+((self.gamma-1)/2)*N)/A)*(dAdx)
      term2 = ((1+self.gamma*N)*(1+((self.gamma-1)/2)*N)/T_s)*(dTsdx)
      term3 = ((4*self.gamma*N*(1+((self.gamma-1)/2)*N))/dH)*Cf
      #print(dAdx, dTsdx, Cf, N, (N/(1-N))*(term1+term2+term3), x*1000)
      dNdx = (N/(1-N))*(term1+term2+term3)
      #print(dNdx, N, x)
    return dNdx

  #Function defining dPdx used for RK4 for calculating Ps
  def f_Ps(self, x, Ps, d, N, T_s, Cf, A, dAdx, dTsdx, last):
    dH = 2*(np.sqrt(A/np.pi))
    term1 = 0
    term2 = -((self.gamma*N)/(2*T_s))*(dTsdx)
    term3 = -(2*self.gamma*N/dH)*Cf
    return Ps*(term1+term2+term3)

  #Function defining dTdx used for RK4 for calculating T
  def f_T(self, x, T, d, N, T_s, Cf, A, dAdx, dTsdx, last):
    dH = 2*(np.sqrt(A/np.pi))
    term1 = (((self.gamma-1)*N)/A)*(dAdx)
    term2 = ((1-self.gamma*N)*(1+((self.gamma-1)/2)*N)/T_s)*(dTsdx)
    term3 = -((2*self.gamma*(self.gamma-1)*(N**2))/dH)*Cf
    dTdx = (T/(1-N))*(term1+term2+term3)
    
    if dTdx > 0:
      dTdx = last
    return dTdx
  
  def f_Ts(self, x, T_s, d, N, not_used, Cf, A, dAdx, q, last):
    return -(1/self.cp)*q

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
        viscosity_species.append(np.exp(all_species[i][6]*np.log(temp) + all_species[i][7]/temp + all_species[i][8]/(temp**2) + all_species[i][9]))
        lambda_species.append(np.exp(all_species[i][2]*np.log(temp) + all_species[i][3]/temp + all_species[i][4]/(temp**2) + all_species[i][5]))

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

    return np.array(viscosity)*(10**-7), np.array(thermal_conductivity)/10000

  def calc_hg(self, viscosity, Pr, T_wall, T_s, P, N):
    C = 0.026 #Generally accepted constant for Bartz
    sigma = (((T_wall/(2*T_s))*(1+((self.gamma-1)/2)*N) + 0.5)**(-0.68))*((1+((self.gamma-1)/2)*N)**(-0.12))
    hg = ((C/(self.D_t**0.2))*(((viscosity**0.2)*self.cp)/(Pr**0.6))*((P**0.8)/self.c_star)*((self.D_t**0.1)/self.Rc_t))*((self.A_t/self.area)**0.9)*sigma
    return hg

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
index_t = np.where(radius==np.min(radius))[0]
R_ct = 0.625*0.0254 #in to m
T_wall0 = 800#293 #Room temp in K
T0 = 3284 #K
P0 = 220*6894.75729 #psi to Pa
M0 = 0.018
epsilon = 0.0015
c_star = 1837.8 #m/s
mdot = 0.366 #kg/s

geometries = [x, radius, R_ct]
conditions_initial = [T_wall0, T0, P0, M0]
flow_props = [epsilon, c_star, mdot]
chem_props = [MM_CO2/MM_total, MM_H2O/MM_total, MM_O2/MM_total, 0.25, 0.5, 0.25]

Blip = Engine(geometries, conditions_initial, flow_props, chem_props)

finalMach, final_Ts, final_Ps, final_temp, final_hg, final_q, final_dAdx, final_dTsdx, final_viscosity, final_thermal_conductivity, final_Cf = Blip.Run_Heat_Transfer()

fig, axes = plt.subplots(3, 3, figsize=(12, 12), constrained_layout=True)  # 3 rows, 3 column

# First Row
# First subplot
axes[0, 0].plot(x*1000, finalMach, color='red')
axes[0, 0].set_title("Mach vs Length")
axes[0, 0].set_xlabel("Axial distance (mm)")
axes[0, 0].set_ylabel("Mach Number")

# Second subplot
axes[1, 0].plot(x*1000, final_Ts, color='green')
axes[1, 0].set_title("Ts vs Length")
axes[1, 0].set_xlabel("Axial distance (mm)")
axes[1, 0].set_ylabel("Stagnation Temperature (K)")


# Third subplot
axes[2, 0].plot(x*1000, final_Ps, color='blue')
axes[2, 0].set_title("Ps vs Length")
axes[2, 0].set_xlabel("Axial distance (mm)")
axes[2, 0].set_ylabel("Stagnation Pressure (kPa)")


# Second Row
# First subplot
axes[0, 1].plot(x*1000, final_q, color='red')
axes[0, 1].set_title("Heat Flux vs Length")
axes[0, 1].set_xlabel("Axial distance (mm)")
axes[0, 1].set_ylabel("Heat Flux (J/m2s)")

# Second subplot
axes[1, 1].plot(x*1000, radius*1000, color='green')
axes[1, 1].set_title("Contour")
axes[1, 1].set_xlabel("Axial distance (mm)")
axes[1, 1].set_ylabel("Radius (mm)")

# Third subplot
axes[2, 1].plot(x*1000, final_temp, color='blue')
axes[2, 1].set_title("T vs Length")
axes[2, 1].set_xlabel("Axial distance (mm)")
axes[2, 1].set_ylabel("Temperature (K)")

#Third Row
# First subplot
axes[0, 2].plot(x*1000, final_dAdx, color='red')
axes[0, 2].set_title("dAdx vs Length")
axes[0, 2].set_xlabel("Axial distance (mm)")
axes[0, 2].set_ylabel("dAdx")

# Second subplot
axes[1, 2].plot(x*1000, final_hg, color='green')
axes[1, 2].set_title("Heat Transfer Coefficient vs Length")
axes[1, 2].set_xlabel("Axial distance (mm)")
axes[1, 2].set_ylabel("Heat Transfer Coefficient")

# Third subplot
axes[2, 2].plot(x*1000, final_dTsdx, color='blue')
axes[2, 2].set_title("dTsdx vs Length")
axes[2, 2].set_xlabel("Axial distance (mm)")
axes[2, 2].set_ylabel("dTsdx")
# Adjust layout to prevent overlapping
plt.tight_layout()
fig.subplots_adjust(wspace=0.5, hspace=0.5)
# Show the plot
plt.show()