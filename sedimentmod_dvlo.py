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
def sediment_DVLO(den, a, T, h_sed, surf_pot, v_min):

	###########################
	###      CONSTANTS      ###
	###########################

	den_water = 998					# Water density
	sed_net = 8.69e-11				# Net sediment accumulation rate in water [m/s]
	sett_vel = 2.94e-5				# Settling velocity of suspended particles [m/s]
	susp_water = 5e-3				# Concentration of suspended matter in water [kg/m3]
	Fssed = 0.2						# Volume fraction of solids in sediment
	rho_solid = 2500				# Mineral density of soil/sediment [kg/m3]
	pi = np.pi						# pi
	mu_water = 8.9e-4				# Dynamic water viscosity [Pa*s]
	g = 9.81						# Gravitational acceleration [m/s^2]
	r_nc = 1e-7						# Radius of natural colloids in water [m]
	N_nc = 10**(15.8-3.2*np.log10(2*r_nc*1e9))		# Concentration of natural colloids in water [N/m^3]
	den_nc = 2e3					# Density of natural colloid [kg/m^3]
	psi_nc = -3e-2					# Surface potential of natural colloid [V] #Z potential NC
	G_shear = 10						# Surface water shear rate [s^-1]
	k_b = 1.3806488e-23				# Boltzman constant [J/K]
	por = 0.35						# Porous medium porosity [m^3/m^3]
	d_grain = 0.256e-3				# Grain collector diameter [m]
	U_darcy = 1e-6					# Darcy approach velocity [m/s]
	epsilon_0 = 8.8541878176e-12	# Dielectric constant of free space
	epsilon_r = 80					# Relative permitivity of the solution
	R_gas = 8.31446261815324		# J⋅K−1⋅mol−1
	I = 0.72						# mol /kg ionic strength of the electrolyte for seawater
	e = 1.60218e-19 				# elementary charge [C]
	NA = 6.02214076e23				# Avogrado’s number
	Hamaker = 1e-20
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
	

	###########################
	###    REMOVAL RATES    ###
	###########################

	##### DEGRADATION #####
	k_deg = 0
	
	##### BURIAL #####
	k_bur = sed_net/h_sed		#burial rate [1/s]

	##### AGGREGATION #####
	print("Calculating sediment model...")
	k_agg = f_col(r_nc, den_nc) * alpha_DVLO(den, a, T, surf_pot, v_min) * N_nc

	##### ATTACHMENT #####
	lambda_filter = (3/2)*((1-por)/(d_grain*por))
	alpha_att = 1		
	k_att = lambda_filter * alpha_att * n0(d_grain) * U_darcy
	#print(lambda_filter, alpha_att, n0(d_grain), U_darcy)
	#print(k_att)


	###########################
	###   TRANSPORT RATES   ###
	###########################

	### RESUSPENSION ********************
	sed_gross = sett_vel*susp_water/(Fssed*rho_solid) #gross sedimentation rate in water [m/s]
	if (sed_gross > sed_net):
		resusp = sed_gross-sed_net
	else:
		resusp = 0. #resuspension flux [m/s]
	k_resusp = resusp/h_sed #resuspension rate [1/s]

	### VOLATILLIZATION ********************
	k_volat_wa = 0


	###########################
	###       OUTPUTS       ###
	###########################

	k_list = {"ksed_bur": k_bur, "kresus": k_resusp, "kagg": k_agg, "katt": k_att}
	print("sediment: ", k_list)

	return k_list

#sediment_DVLO(den, a, T, h_sed, surf_pot, v_min):
#sediment_klist = sediment_DVLO(10500, 50e-9, 293, 1, -32.3e-3, 0.1)