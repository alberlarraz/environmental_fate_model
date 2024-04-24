# -*- coding: utf-8 -*-
"""

@author: Alberto Larraz
@author: Blanca Pozuelo

"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from time import sleep
from scipy.integrate import quad

def soil_DVLO(pr, den, a, T, surf_pot, v_min):

    Hamaker = 1e-20
    epsilon_0 = 8.8541878176e-12
    epsilon_r = 80
    inv_k_debye = 0.43e-9
    fitting_lambda = 0.00000000045
    fitting_V0 = 0.002
    k_b = 1.3806488e-23
    pi = np.pi
    h_soil=0.1                      #depth of soil [m]
    Finf_ws=0.25					# Volume fraction of rain infiltrating into soil
    erosion=9.51e-13				# Soil erosion [m/s]
    den_water = 998.				# Water density [kg/m3]
    F_run=0.25						# Volume fraction of rain running off to waters
    mu_water = 8.9e-4				# Dynamic water viscosity [Pa*s]
    G_shear = 10					# Surface water shear rate [s^-1]
    r_nc = 1.43e-7					# Radius of natural colloids in water [m]
    N_nc = 10**(15.8-3.2*np.log10(2*r_nc*1e9))	# Concentration of natural colloids in water [N/m^3]
    den_nc = 2e3					# Density of natural colloid [kg/m^3]
    g = 9.81						# Gravitational acceleration [m/s^2]
    por = .35						# Porous medium porosity
    d_grain = 2.56e-4				# Grain collector diameter [m]
    U_darcy = 3.53e-6

	###########################
	### AUXILIARY FUNCTIONS ###
	###########################

	# To compute the gravitational settling velocity
    def v_sett(radius, density):
        return (2*(density - den_water)*g*(radius**2)) / (9*mu_water)

	# To compute collision rate
    def f_col(radius, density):
		# Brownian motion (random diffusive movements of particles)
        f_brown_nc = ((2*k_b*T)/(3*mu_water)) * (((a + radius)**2)/(a*radius))
		# Interception (motion of the surrounding fluids and collide)
        f_int_nc = (4/3)*G_shear*((a + radius)**3)
		# Differential settling (gravitational settling velocities causes the particles to deposit on top of each other)
        f_grav_nc = pi*((a + radius)**2)*np.abs(v_sett(a, den) - v_sett(radius, density))
		# Return the value of the collision rate
        return f_brown_nc + f_int_nc + f_grav_nc

	# To compute the collection efficiency
    def n0(diam):
        gam = (1-por)**(1/3)
        Di = k_b * T / (6*pi*mu_water*a)
        radius = diam/2

        A_s = (2*(1-gam**5)) / (2-(3*gam)+(3*gam**5)-(2*gam**6))		# Aspect ratio
        N_r = a / radius												
        N_pe = U_darcy * diam / Di										# Peclet number
        N_vdw = Hamaker / (k_b*T)										# Van der waals number
        N_g = (2*(a**2)*(den-den_water)*g) / (9*mu_water*U_darcy)		# Gravitational number

        n_brown = 2.4*(A_s**(1/3))*(N_r**(-0.081))*(N_pe**(-0.715))*(N_vdw**(0.052))
        n_int = 0.55*A_s*(N_r**(1.55))*(N_pe**(-0.125))*(N_vdw**(0.125))
        n_grav = 0.22*(N_r**(-0.24))*(N_g**(1.11))*(N_vdw**(0.053))
        return n_brown + n_int + n_grav

    #To compute collision efficiency 
    def alpha_DVLO(den, a, T, surf_pot, v_min):

        def beta(dist):
            return ((6*(dist**2))+(13*a*dist)+(2*(a**2)))/((6*(dist**2))+(4*a*(dist)))

        def Va(dist):
            return -(Hamaker/6)*((2*a**2)/(dist**2+4*dist*a) + (2*a**2)/((dist+2*a)**2) + np.log((dist**2+4*dist*a)/((dist+2*a)**2)))

        def Vr(dist):
            return 2*np.pi*epsilon_0*epsilon_r*a*((surf_pot)**2)*np.log(1+np.exp(-(dist)/(inv_k_debye)))

        def integral(h):
            return 3*mu_water*a*(beta(h))/((2*a+h)**2) * np.exp((Va(h) + Vr(h))/(k_b*T))

        def maxwell(v):
                return 4*pi*((((4/3*den*pi*a**3)/(2*pi*k_b*T))**(3/2))*(v**2)*np.exp(-(4/3*den*pi*a**3)*(v**2)/(2*k_b*T)))


        k_list = []
        x_values = np.arange(34e-9, 1e-2, 1e-8)
        y_values = []

        for i in tqdm(x_values):
            k_int = integral(i)
            k_list.append((i,k_int))
            y_values.append(k_int)

        area_list = []

        for count, (x,y) in enumerate( tqdm(k_list) ):
            if count > 0: 
                area = (x -x_0)*(y_0 + y)/2
                area_list.append(area)
            x_0 = x
            y_0 = y

        k = sum(area_list)
        print("k:", k)

        dvlo_mod = 2*k_b*T/k
        max_mod = quad(maxwell, v_min, np.inf)[0]

        return dvlo_mod*(1-max_mod)


	###########################
	###    REMOVAL RATES    ###
	###########################

	##### LEACHING #####
    C_leach=np.exp(-5)*10*h_soil/(1.-np.exp(-10*h_soil)) #depth dependend correction factor
    k_leach=C_leach*pr*Finf_ws/h_soil #leaching to groundwater rate [1/s]

	##### AGGREGATION #####
    print("Calculating soil model...")
    k_agg = f_col(r_nc, den_nc) * alpha_DVLO(den, a, T, surf_pot, v_min) * N_nc


	##### ATTACHMENT #####
    lambda_filter = (3/2)*((1-por)/(d_grain*por))	# Filtration
    alpha_att = 1									# Approach for attachment efficiency on NP and grains

    k_att = lambda_filter * alpha_att * n0(d_grain) * U_darcy
	

	###########################
	###   TRANSPORT RATES   ###
	###########################

    C_runoff = (h_soil/0.1) / (1 - np.exp(-h_soil/0.1))			# Correction factor

	##### RUNOFF #####
	
    k_runoff=F_run*pr*C_runoff/h_soil #runoff rate [1/s]
	
	##### EROSION #####

    k_erosion=erosion*C_runoff/h_soil #erosion rate [1/s]

	###########################
	###       RESULTS       ###
	###########################


    k_list = {"ksoil_leach": k_leach, "krunoff": k_runoff, "kerosion": k_erosion, "kagg": k_agg, "katt": k_att}
    print("soil", k_list)

    return k_list

#soilmod(pr, den, r, t, surf_pot, thickness, v_min):
#soil_klist = soil_DVLO(2.22e-8, 10500, 50e-9, 293, -32.3e-3, 0.1)
