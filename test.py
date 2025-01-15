import numpy as np

Ru = 8.31446261815324 #Universal gas constant (J/mol K)
MM_CO2 = 0.044009 #Molar mass (kg/mol)
MM_H2O = 0.01801528 #Molar mass (kg/mol)
MM_O2 = 0.032004 #Molar mass (kg/mol)
MM_total = MM_CO2 + MM_H2O + MM_O2 #(kg/mol)
Rs = Ru/MM_total #Specific gas constant of entire mixture (J/kg K)

chem_props = [MM_CO2/MM_total, MM_H2O/MM_total, MM_O2/MM_total, 0.25, 0.5, 0.25]
mass_frac_CO2, mass_frac_H2O, mass_frac_O2, mol_frac_CO2, mol_frac_H2O, mol_frac_O2 = chem_props
def calc_viscosity_and_lambda(T_star):
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
        CO2 = [mol_frac_CO2, MM_CO2, tc_coef_CO2_low[0], tc_coef_CO2_low[1], tc_coef_CO2_low[2], tc_coef_CO2_low[3], visc_coef_CO2_low[0], visc_coef_CO2_low[1], visc_coef_CO2_low[2], visc_coef_CO2_low[3]]
        H2O = [mol_frac_H2O, MM_H2O, tc_coef_H2O_low[0], tc_coef_H2O_low[1], tc_coef_H2O_low[2], tc_coef_H2O_low[3], visc_coef_H2O_low[0], visc_coef_H2O_low[1], visc_coef_H2O_low[2], visc_coef_H2O_low[3]]
        O2 = [mol_frac_O2, MM_O2, tc_coef_O2_low[0], tc_coef_O2_low[1], tc_coef_O2_low[2], tc_coef_O2_low[3], visc_coef_O2_low[0], visc_coef_O2_low[1], visc_coef_O2_low[2], visc_coef_O2_low[3]]
      else:
        CO2 = [mol_frac_CO2, MM_CO2, tc_coef_CO2_high[0], tc_coef_CO2_high[1], tc_coef_CO2_high[2], tc_coef_CO2_high[3], visc_coef_CO2_high[0], visc_coef_CO2_high[1], visc_coef_CO2_high[2], visc_coef_CO2_high[3]]
        O2 = [mol_frac_O2, MM_O2, tc_coef_O2_high[0], tc_coef_O2_high[1], tc_coef_O2_high[2], tc_coef_O2_high[3], visc_coef_O2_high[0], visc_coef_O2_high[1], visc_coef_O2_high[2], visc_coef_O2_high[3]]
        if temp < 1073.2:
          H2O = [mol_frac_H2O, MM_H2O, tc_coef_H2O_low[0], tc_coef_H2O_low[1], tc_coef_H2O_low[2], tc_coef_H2O_low[3], visc_coef_H2O_low[0], visc_coef_H2O_low[1], visc_coef_H2O_low[2], visc_coef_H2O_low[3]]
        else:
          H2O = [mol_frac_H2O, MM_H2O, tc_coef_H2O_high[0], tc_coef_H2O_high[1], tc_coef_H2O_high[2], tc_coef_H2O_high[3], visc_coef_H2O_high[0], visc_coef_H2O_high[1], visc_coef_H2O_high[2], visc_coef_H2O_high[3]]
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