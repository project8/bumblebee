import matplotlib.pyplot as plt
from phase3_cavity_methods import CRES_RightCylinder_Cavity
import numpy as np
import sys
import os

pi = np.pi

length = 0.25 #m
radius = 0.12 #m

my_cavity = CRES_RightCylinder_Cavity(radius, length)
freqs = my_cavity.get_mode_frequencies(100, 100)

mode = (1, 2, 2, 1)
f = my_cavity.mode_freqs[mode]
print(f/1e9)
resolution = 300
bw=2e9
save_text=False

my_cavity.field_map_2d_zslice_single_mode(mode, length/2, plotz=False)
plt.savefig('mode_fields/f{}{}{}{}_mode_zhalf.png'.format(mode[0], mode[1], mode[2], mode[3]), bbox_inches = 'tight', dpi=resolution)
plt.close()
my_cavity.field_map_2d_phislice_single_mode(mode, 0, plotz=False)
plt.savefig('mode_fields/f{}{}{}{}_mode_phi0.png'.format(mode[0], mode[1], mode[2], mode[3]), bbox_inches = 'tight', dpi=resolution)
plt.close()

rel_modes = my_cavity.find_mode_density(my_cavity.mode_freqs[mode], bw)[1]

#get all the Qs
my_cavity.calculate_inv_loaded_qs()

electron_position = [(0.01, 0 , length/2),
                     (0.01, 0 , length/4),
                     (my_cavity.radius/2, 0 , length/2),
                     (my_cavity.radius/2, 0 , length/4),
                     ]
electron_dipole = (1.0e5, 0, 0)

zprobe = length/2
for ex in electron_position:
    directory='unperturbed/f{0[0]}{0[1]}{0[2]}{0[3]}bw2/'.format(mode)
    if not os.path.exists(directory):
        os.makedirs(directory)
    zslice = my_cavity.field_map_2d_zslice(rel_modes, zprobe, ex, electron_dipole, f, plotz=False )
#    if save_text:
#        np.savetxt('{3}/f{0[0]}{0[1]}{0[2]}{0[3]}_zslice_er{1}_ez{2}_zhalflength_Er.txt'.format(mode, ex[0],ex[2], directory),zslice[0])
#        np.savetxt('{3}/f{0[0]}{0[1]}{0[2]}{0[3]}_zslice_er{1}_ez{2}_zhalflength_Ephi.txt'.format(mode, ex[0],ex[2],directory), zslice[1])
    plt.savefig('{3}/f{0[0]}{0[1]}{0[2]}{0[3]}_zslice_er{1}_ez{2}zhalflength_.png'.format(mode, ex[0], ex[2], directory), bbox_inches = 'tight', dpi=resolution)
    plt.close()

    phislice = my_cavity.field_map_2d_phislice(rel_modes, 0, ex, electron_dipole, f, plotz=False)
    plt.savefig('{3}/f{0[0]}{0[1]}{0[2]}{0[3]}_phislice_er{1}_ez{2}_phi0.png'.format(mode, ex[0], ex[2], directory), bbox_inches = 'tight', dpi=resolution)
    plt.close()


