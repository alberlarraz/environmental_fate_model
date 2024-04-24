# -*- coding: utf-8 -*-
"""

@author: Alberto Larraz
@author: Blanca Pozuelo

"""

from scipy.integrate import odeint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate
import sys
from airmod import airmod
from soil_dvlo import soil_DVLO
from watermod_dvlo import water_DVLO
from sedimentmod_dvlo import sediment_DVLO

	
### INPUTS
#
"""
if len(sys.argv) < 13:
    print("Error!: python emod_DVLO.py em_water, em_air, em_soil, em_sediment, time, t_units, nm_name, d, psi_enp, den, v_min, L")
    sys.exit(1)

inputs = sys.argv[1:]
"""

# From input -> inputs=[em_water, em_air, em_soil, em_sediment, time, t_units, nm_name, d, psi_enp (Z Potential), den, v_min, L)
# From input -> inputs=(0 1000 0 0 1 4 "Ag" 100 -32.3e-3 10500 0.1 0)

em_air = float(input("initial emissions for air (kg): "))
em_water = float(input("initial emissions for water (kg): "))
em_soil = float(input("initial emissions for soil (kg): "))
em_sediment = float(input("initial emissions for sediment (kg): "))										
t_units = int(input("""Units of time (Choose a number between 0 and 4)
0 -> Hours 
1 -> Days 
2 -> Weeks 
3 -> Months 
4 -> Years : 
"""))												#Time units [0: Hours, 1: Days, 2: Weeks, 3: Months, 4: Years]
while t_units < 0 or t_units > 4:
	t_units = int(input("""Units of time (CHOOSE ONE OF THE FOLLOWING NUMBERS [0-1-2-3-4])
0 -> Hours 
1 -> Days 
2 -> Weeks 
3 -> Months 
4 -> Years : 
"""))												#Time units [0: Hours, 1: Days, 2: Weeks, 3: Months, 4: Years]
time = float(input("Period of time: "))			
time_0 = 0 																#initial of time interval [months]. User chooses time scale
units = ["Hours", "Days", "Weeks", "Months", "Years"]
scale = [3600, 3600*24, 3600*24*7, 3600*24*30, 3600*24*365] 			#secs to ts
t_int=np.linspace(time_0*scale[t_units],time*scale[t_units],10000)  	#time interval [months to seconds]

nm_name = input("Name of your MCNM/HARN: ")
shape = int(input("""Dimensionality of your MCNM/HARN (Choose a number between 0 and 1): 
0 -> 0D = Nanospheres, clusters, ...
1 -> 1D = Nanotubes, wires, rods, ...
2 -> 2D (layered material) = Graphene, TMDs, MXenes, ...
"""))
while shape < 0 or shape > 2:
	shape = int(input("""Dimensionality of your MCNM/HARN (CHOOSE ONE OF THE FOLLOWING NUMBERS [0-1]): 
0 -> 0D = Nanospheres, clusters, ...
1 -> 1D = Nanotubes, wires, rods, ...
2 -> 2D (layered material) = Graphene, TMDs, MXenes, ...
"""))
d = float(input("Particle diameter of the nanoparticle (m). Please, express scientific Notation as the following example '50e-9': "))
r = d/2								# Particle radius [m]
psi_enp = float(input("Z potential (V) of the MCNM/HARN: "))
den = float(input("Particle density (kg/m3): ")) 				# Particle density [kg/m^3]

#thickness = float(inputs[10])*1e-9	# Thickness of structural layer

### HIGH ASPECT RATIO NANOPARTICLES

#Aerodynamic ratio - For those particles with a non-spherical shape
if shape == 0:
	r_aer = r
if shape == 1:
	L = float(input("Length of the 1D High Aspect Ratio Nanoparticles (m): "))			#Length of the High Aspect Ratio Nanoparticles
	d_aer_para = d*(9/4*(np.log(2*L/d)-0.807))**(1/2)
	d_aer_perp = d*(9/8*(np.log(2*L/d)+0.193))**(1/2)
	r_aer = (d_aer_para + 2*d_aer_perp)/6
	print("aerodynamic radius: ", r_aer)
elif shape == 2:
	L = float(input("Geometrical diameter of 2D High Aspect Ratio Nanoparticles (m): "))			#Geometrical diametre of the High Aspect Ratio Nanoparticles
	d_aer_para = (9*np.pi*den*L*d/16)**(1/2)
	d_aer_perp = (27*np.pi*den*L*d/32)**(1/2)
	r_aer = (d_aer_para*2/3 + d_aer_perp*1/3)/2
	print("aerodynamic radius: ", r_aer)



### CONSTANTS 
Asys = float(input("Surface of the system (m2): ")) 	#surface of the system [m2]
Fsoil = float(input("Fraction of soil of your system (0-1): ")) #fraction of soil surface
while Fsoil > 1:
	print("Fsoil must be a number between 0 and 1")
	Fsoil = float(input("Fraction of soil of your system (0-1): ")) #fraction of soil surface
Flake = 0
Ffresh = round(1.0 - Fsoil, 2)
print("Fraction of freshwater: ", Ffresh)
precipitation_rate = float(input("Precipitation rate in that place (mm/year): "))




# TRANSPORT AND FATE MODEL ***************************************************************
#*****************************************************************************************

# Environmental media characteristics
#Asys=1.018e13  	#surface of the system [m2]
hair=1000.     	#air mixing height [m]
hsoil=0.1      	#depth of soil [m]
hlake=10.	   	#lake's water depth [m]
hfresh=1.      	#freshwater depths [m]
hsed=1.        	#sediment depth [m]
#Flake=0.003    	#fraction of lakes surface 
#Ffresh=0.027   	#fraction of freshwater surface
#Fsoil=0.97     	#fraction of soil surface
t_w = 293	   	#water temperature [K]
t_air = 287    	#air temperature [K]
pr = (precipitation_rate*1e-3)/(365*24*60*60)  	#precipitation rate [m/s]
v_min = 0.1 	# Min velocity to stay aggregated/attached (m/s)

Vair=hair*Asys                 		#air volume [m3]
Vw=(hlake*Flake+hfresh*Ffresh)*Asys #water volume [m3]
Vs=hsoil*Fsoil*Asys            		#soil volume [m3]
Vsed=hsed*(Flake+Ffresh)*Asys  		#sediment volume [m3]

# FATE AND TRANSPORT EQUATIONS *************************************************************
#*******************************************************************************************

# IN AIR
air_klist = airmod(r_aer, den, pr, t_air)

#IN SOIL
soil_klist = soil_DVLO(pr, den, r_aer, t_w, psi_enp, v_min)

# IN WATER
water_klist = water_DVLO(den, r_aer, t_w, hlake, psi_enp, v_min)

# IN SEDIMENT
sediment_klist = sediment_DVLO(den, r_aer, t_w, hsed, psi_enp, v_min)

# DEFINITION OF THE COEFFICIENTS OF THE 4 DIFFERENTIAL EQUATIONS
A=air_klist["kwet_w"] + air_klist["kdry_w"] + air_klist["kwet_s"] + air_klist["kdry_s"] + air_klist["kagg"] + air_klist["katt"]
B=air_klist["kwet_s"] + air_klist["kdry_s"] 
C=soil_klist["ksoil_leach"] + soil_klist["krunoff"] + soil_klist["kerosion"] + soil_klist["kagg"] + soil_klist["katt"]
D=air_klist["kwet_w"] + air_klist["kdry_w"]
E=soil_klist["krunoff"] + soil_klist["kerosion"]
F=water_klist["ksedim"]
I=water_klist["kagg"] + water_klist["katt"]
G=sediment_klist["kresus"]
H=sediment_klist["ksed_bur"] + sediment_klist["kagg"] + sediment_klist["katt"] + sediment_klist["kresus"]

# M[0], M[1], M[2], M[3] are the masses in air, soil, water and sediment
def dM_dt(M,time):
	return -A*M[0], B*M[0]-C*M[1], D*M[0]+E*M[1]-F*M[2]-I*M[2]+G*M[3], F*M[2]-H*M[3]

# This must be the output of the material flow
#----shall be calculated with the corrected emissions emt_xxx but lacking coeffs!!
M0=[em_air,em_soil,em_water,em_sediment]
	
Ms=odeint(dM_dt, M0, t_int)
#Trick to avoid negative mass values ---- problem with tolerance of odeint 
Ms[Ms<0] = 0
np.savetxt('Ms.txt', Ms)
	
mair=Ms[:,0]
msoil=Ms[:,1]
mwater=Ms[:,2]
msed=Ms[:,3]

print(f"LIST OF INPUTS:\nInitial Air emmisions: {em_air} kg \nInitial Water emissions: {em_water} kg \nInitial Soil emissions: {em_soil} kg \nInitial Sediment Emissions: {em_sediment} kg")
print(f"Time of evaluation: {time} {units[t_units]}")
if shape == 0:
	print(f"Name of MCNM/HARN: {nm_name}\nNumber of dimensions: {shape}\nRadious: {r} nm\nZeta Potential: {psi_enp} V\nDensity: {den} kg/m3")
elif shape > 0:
	print(f"Name of MCNM/HARN: {nm_name}\nNumber of dimensions: {shape}\nRadious: {r_aer} nm\nLenght of non-nano dimension: {L} m\nZeta Potential: {psi_enp} V\nDensity: {den} kg/m3\nAerodynamic Diameter: {r} m")
print(f"Surface of the system: {Asys} m\nFraction of soil oy the system: {Fsoil} \nFraction of water of your system: {Ffresh}\nPrecipitation rate in that place:{precipitation_rate} mm/year")

print ("\nMair(0)=%.3f \t\tMsoil(0)=%.3f \t\tMwater(0)=%.3f \tMsed(0)=%.3f" %(Ms[0,0], Ms[0,1], Ms[0,2], Ms[0,3]))
print ("Mair(end)=%e \tMsoil(end)=%.3f \tMwater(end)=%.3f \tMsed(end)=%.3f" %(Ms[-1,0], Ms[-1,1], Ms[-1,2], Ms[-1,3]))
print("\nMtot(0)=%f kg\nMtot(end)=%f kg" %(Ms[0,0]+Ms[0,1]+Ms[0,2]+Ms[0,3], Ms[-1,0]+Ms[-1,1]+Ms[-1,2]+Ms[-1,3]))
mloss= 100*((Ms[0,0]+Ms[0,1]+Ms[0,2]+Ms[0,3]) - (Ms[-1,0]+Ms[-1,1]+Ms[-1,2]+Ms[-1,3]))/(Ms[0,0]+Ms[0,1]+Ms[0,2]+Ms[0,3])
print("mass lost %% = %f" % (mloss))

plt.figure(1)
plt.plot(t_int/scale[t_units], Ms)
plt.xlabel(units[t_units])
plt.ylabel("Mass [kg]")
plt.title(nm_name)
plt.legend(("Air","Soil","Water","Sediment"))
plt.savefig('mass.png')
plt.show()

# EVALUATION OF PECs *****************************************************************
#*************************************************************************************

Cair = (mair/Vair)*1e6	#mg/m3
Cwater = (mwater/Vw)*1e6	#mg/m3
Csoil = (msoil/Vs)*1e6	#mg/m3
Csed = (msed/Vsed)*1e6	#mg/m3

"""plt.figure(2)
plt.plot(t_int/scale[t_units], Cair, label="Air")
plt.plot(t_int/scale[t_units], Csoil, label="Soil")
plt.plot(t_int/scale[t_units], Cwater, label="Water")
plt.plot(t_int/scale[t_units], Csed, label="Sediment")
plt.legend(("Air","Soil","Water","Sediment"))
plt.xlabel(units[t_units])
plt.ylabel("[NM] $[mg/m^3]$")
plt.title(nm_name)
plt.savefig('concs.png')
plt.show()"""

PECair=integrate.simps(Cair, t_int)/t_int[-1]
PECwater=integrate.simps(Cwater, t_int)/t_int[-1]
PECsoil=integrate.simps(Csoil, t_int)/t_int[-1]
PECsed=integrate.simps(Csed, t_int)/t_int[-1]

print("\nPEC air= %.2e mg/m^3\t PEC sediment= %.2e mg/m^3" % (PECair,PECsed))
print("PEC water= %.2e mg/m^3\t PEC soil= %.2e mg/m^3" % (PECwater,PECsoil))

# Conditional return of PEC
pec_ret_list = {}

pec_ret_list['PEC_Water'] = PECwater

pec_ret_list['PEC_Soil'] = PECsoil

pec_ret_list['PEC_Air'] = PECair

pec_ret_list['PEC_Sediment'] = PECsed

print("\nReturned PECs: ", pec_ret_list)



