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

#init_printing(use_latex="mathjax")

def water_DVLO(den, a, T, h_water, surf_pot, v_min):
	
	###########################
	###      CONSTANTS      ###
	###########################	

	den_water = 998.						# Water density [kg/m3]
	g = 9.81								# Gravitational acceleration [m/s^2]
	pi = np.pi								# pi
	mu_water = 0.001						# Dynamic water viscosity [kg/m.s]
	r_nc = 5e-8								# Radius of natural colloids in water [m]
	N_nc = 10**(15.8-3.2*np.log10(2*r_nc*1e9))	# Concentration of natural colloids in water [N/m^3]
	den_nc = 1.25e3							# Density of natural colloid [kg/m^3]
	psi_nc = -8.80e-3						# Surface potential of natural colloid [V] #Z potential NC
	r_nlp = 5e-7							# Radius of natural larger particles in water [m]
	N_nlp = 10**(15.8-3.2*np.log10(2*r_nlp*1e9))	# Concentration of larger particles in water [N/m^3]
	den_nlp = 2e3							# Density of natural larger particles [kg/m^3]
	psi_nlp = -12e-3						# Surface potential of larger particles [V] #Z potential NLP
	G_shear = 10							# Surface water shear rate [s^-1]
	k_b = 1.3806488e-23						# Boltzman constant [J/K]
	epsilon_0 = 8.8541878176e-12			# Dielectric constant of free space
	epsilon_r = 80							# Relative permitivity of the solution
	R_gas = 8.31446261815324				# J⋅K−1⋅mol−1
	I = 0.72								# mol /kg ionic strength of the electrolyte for seawater
	e = 1.60218e-19 						# elementary charge [C]
	NA = 6.02214076e23						#Avogrado’s number
	Hamaker = 1e-20
	surf_pot = -23e-3
	inv_k_debye = 0.43e-9
	fitting_lambda = 0.00000000045
	fitting_V0 = 0.002

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

	##### DEGRADATION #####
	k_deg = 0

##### AGGREGATION #####
	print("Calculating water model...")
	k_agg = f_col(r_nc, den_nc) * alpha_DVLO(den, a, T, surf_pot, v_min) * N_nc

	##### ATTACHMENT ##### 
	k_att = f_col(r_nlp, den_nlp) * alpha_DVLO(den, a, T, surf_pot, v_min) * N_nlp

	###########################
	###   TRANSPORT RATES   ###
	###########################

	##### SEDIMENTATION #####
	k_sed = v_sett(a, den)/h_water		# Sedimentation rate


	###########################
	###       OUTPUTS       ###
	###########################

	k_list = {"ksedim": k_sed, "kagg": k_agg, "katt": k_att}
	print("water: ", k_list)

	return k_list

#watermod(den, a, T, h_water, surf_pot, v_min):
#water_klist = water_DVLO(10500, 50e-9, 293, 10, -32.3e-3, 0.1)
