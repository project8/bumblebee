import matplotlib.pyplot as plt
from phase3_cavity_methods import CRES_RightCylinder_Cavity
import numpy as np
import sys

pi = np.pi
length = 0.25 #m
radius = 0.12 #m

my_cavity = CRES_RightCylinder_Cavity(radius, length)
#freqs = my_cavity.get_mode_frequencies(1110, 60)
freqs = my_cavity.get_mode_frequencies(20,10)

mode = (1,2,2,1)
rel_modes = my_cavity.find_mode_density(my_cavity.mode_freqs[mode], 1e9)[1]

# antenna positions
antenna_loc = [
               (radius/10, 0, length/10),
               (radius/2, 0, length/10),
               (radius*9/10, 0, length/10),
               (radius/10, pi, length/10),
               (radius/2, pi, length/10),
               (radius*9/10, pi, length/10)
               ]

antenna_length = [
                 (0.01, 0, 0.0),
                 (0.01, 0, 0.0),
                 (0.01, 0, 0.0),
                 (-0.01, 0, 0.0),
                 (-0.01, 0, 0.0),
                 (-0.01, 0, 0.0)
                 ]

#add these antennas to the cavity
for loc, l in zip(antenna_loc, antenna_length):
    my_cavity.add_antenna(loc, l)

# get all the Qs
my_cavity.calculate_inv_loaded_qs()

# calculate the antenna couplings
test = my_cavity.precalculate_antenna_mode_couplings(freq_mode_list=rel_modes)

# move the electron along the radius of the cavity and see how the antenna
# voltages change

r = np.linspace(0.001, radius - 0.001, 100)
voltage = np.zeros((len(r), len(antenna_loc)), dtype=np.complex_)
voltage_mag = np.zeros((len(r), len(antenna_loc)))
voltage_phase = np.zeros((len(r), len(antenna_loc)))

#define a frequency to drive the cavity with
mode = (1,2,2,1)
freq = my_cavity.mode_freqs[mode]
#move antenna across the middle of the cavity, z=Length/2
for i in range(len(r)):
    electron_position = (r[i], 0, length/2)
    electron_dipole = (0.01, 0, 0)
    ex = my_cavity.calculate_antenna_voltages(electron_position,
                                              electron_dipole, freq, freq_mode_list = rel_modes)
    exabs = ((ex*np.conjugate(ex))**(1/2)).real
    exangle = np.angle(ex, deg=1)

    voltage[i] = ex
    voltage_mag[i] = exabs
    voltage_phase[i] = exangle

f, axarr = plt.subplots(2, sharex=True)
for i in range(len(antenna_loc)):
    axarr[0].plot(r, voltage.real[:,i], label = 'antenna {}'.format(i))

plt.xlabel('Electron Position [m]')
axarr[0].set_title(r'f = {:.3f} GHz, $z_e$ = length/2, $\phi_e = 0$'.format(freq/1e9))
axarr[0].set_ylabel('Re[V] [arbitrary]')
axarr[0].legend(loc='upper right')
plt.tight_layout()

for i in range(len(antenna_loc)):
    axarr[1].plot(r, voltage[:,i].imag, label = 'antenna {}'.format(i))

axarr[1].set_xlabel(r'$r_e$ [m]')
axarr[1].set_ylabel('Im[V] [degress]')
plt.tight_layout()
plt.savefig('voltage_lowf.pdf', bbox_inches='tight')
plt.close()


