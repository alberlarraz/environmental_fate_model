# -*- coding: utf-8 -*-
"""

@author: Alberto Larraz
@author: Blanca Pozuelo

"""

import numpy as np

def airmod(r, den, pr, t):

	###########################
	###      CONSTANTS      ###
	###########################

	Asys=2.3e11		#surface of the system [m2]
	Fsoil=0.87		#fraction of soil surface (60% Natural soil + 27% Agricultural Soil)
	Furban=0.1		#fraction of urban soil surface
	Fwater=0.03		#fraction of lakes surface
	hair=1000.		#air mixing height [m]
	Asoil=Asys*Fsoil				#soil surface [m2]
	Awater=(Fwater)*Asys		#water surface [m2]
	V_air=hair*Asys					#air volume [m3]
	lambda_air=6.7e-8 # Particle mean free path in air [m]
	den_air=1.225 #air density [kg/m^3]
	den_water=998. #water density [kg/m3]
	g=9.81 #gravitational acceleration [m/s^2]
	mu_air=1.81e-5 #dynamic air viscosity [J/m^3/s]
	k_b=1.38e-23 #boltzmann constant [J/K]
	pi=np.pi #pi
	cd_w = 1.2e-3	# Drag coefficient for water (oceans)
	cd_s = 1e-2	# Drag coefficient for soil
	u = 3	# Wind velocity [m/s]
	drag_ratio=0.3 #viscous to total ratio --> Cv/Cd
	a_veg=1e-5 #vegetation hair width [m]
	A_veg=5e-4 #vegetation large collection radius [m]
	alpha_veg=0.8
	beta_veg=2.
	f_iv=0.01 #fraction of interception by vegetation
	vfr=0.19 #friction velocity [m/s]
	alpha_cc=1.142
	beta_cc=0.558
	gamma_cc=0.999
	r_nuc=17e-9	#Radius nucleation aerosols [m]
	N_nuc=3.2e9	#N nucleation aerosols [N/m^3]
	den_nuc=1300	#Density of nucleation aerosols [kg/m^3]
	r_acc=70e-9	#Radius accumulation aerosols [m]
	N_acc=2.9e9	#N accumulation aerosols [N/m^3]
	den_acc=970	#Density accumulation aerosols [kg/m^3]
	r_coarse=450e-9	#Radius coarse aerosols [m]
	N_coarse=3e5	#N coarse aerosols [N/m^3]
	den_coarse=1500	#Density coarse aerosols [kg/m^3]

	###########################
	### AUXILIARY FUNCTIONS ###
	###########################

	def fCC(radius):					# To compute cunningham coefficient
		return (1 + lambda_air/radius*(alpha_cc + beta_cc * np.exp(-gamma_cc*lambda_air/(4*radius))))

	def fDiffusivity(radius):			# To compute particle diffusivity
		cc = fCC(radius)
		return ((k_b*t*cc) / (6*pi*mu_air*radius))

	def c_thermal (radius, density):	# To compute particle thermal velocity
		vol = (4/3)*pi*(radius**3)
		m = density * vol
		return ((8*k_b*t) / ((pi*m)**(1/2)))

	def k_coag(radius, density, N):
		# Coagulation coefficient
		f_coag = 4*pi*(r + radius)*(fDiffusivity(r) + fDiffusivity(radius))
		# Fuchs Transitional correction coefficient
		alpha = (1 + (4*(fDiffusivity(r) + fDiffusivity(radius)) / ((r + radius) * ((c_thermal(r, den)**2 + c_thermal(radius, density)**2)**(1/2)))))**-1
		# N --> Number concentration of natural particles
		return (f_coag*alpha*N)


	###########################
	###    REMOVAL RATES    ###
	###########################

	print("Calculating air model...")

	##### AGGREGATION #####
	k_coag_nuc = k_coag(r_nuc, den_nuc, N_nuc)	
	k_coag_acc = k_coag(r_acc, den_acc, N_acc)

	k_agg = k_coag_nuc + k_coag_acc

	##### ATTACHMENT #####
	k_att = k_coag(r_coarse, den_coarse, N_coarse)

	###########################
	###   TRANSPORT RATES   ###
	###########################

	##### DEPOSITION #####
	#***** WET DEPOSITION 
	k_wdep_w=pr*Awater/V_air #wet deposition rate constant for water [1/s]
	k_wdep_s=pr*Asoil/V_air #wet deposition rate constant for soil [1/s]

	#***** DRY DEPOSITION
	Sc = mu_air/(den_air*fDiffusivity(r))

	cc = fCC(r)
	tau_air = (den-den_air)*((2*r)**2)*cc/18*mu_air			# Particle relaxion time
	V_term_R = (den_water-den_air)*g*((2*r)**2)*cc/18*mu_air	# Raindrop terminal velocity
	V_term = (den-den_air)*g*((2*r)**2)*cc/18*mu_air			# Terminal velocity
	d_R = (7.e-4)*((6.e5*pr)**(0.25)) 					#raindrop diameter [m]
	St = 2*tau_air*(V_term_R - V_term)/d_R

	R_aer_w = 1/(cd_w * u)
	E_brown_w = Sc**(-1/2)
	E_int_w = 0
	E_grav_w = 10**(-3*St)
	R_sur_w = 1/(vfr * (E_brown_w + E_int_w + E_grav_w))
	v_dep_w = V_term + (1/(R_aer_w + R_sur_w))

	R_aer_s = 1/(cd_s * u)
	E_brown_s = Sc**(-2/3)
	E_int_s = drag_ratio * (f_iv*(r/(r+a_veg)) + (1-f_iv)*(r/(r+A_veg)))
	E_grav_s = (St/(St + alpha_veg))**(beta_veg)
	R_sur_s = 1/(vfr * (E_brown_s + E_int_s + E_grav_s))
	v_dep_s = V_term + (1/(R_aer_s + R_sur_s))

	k_ddep_w=v_dep_w*Awater/V_air #dry deposition rate constant for water [1/s]
	k_ddep_s=v_dep_s*Asoil/V_air #dry deposition rate constant for soil [1/s]

	#***** TOTAL DEPOSITION
	k_dep_w = k_wdep_w + k_ddep_w		# Deposition rate for water
	k_dep_s = k_wdep_s + k_ddep_s		# Deposition rate for soil

	###########################
	###       OUTPUTS       ###
	###########################

	k_list = {"kdry_w": k_ddep_w, "kdry_s": k_ddep_s, "kwet_w": k_wdep_w, "kwet_s": k_wdep_s, "kagg": k_agg, "katt": k_att}
	print("air", k_list)

	return k_list