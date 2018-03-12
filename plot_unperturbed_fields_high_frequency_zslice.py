import matplotlib.pyplot as plt
from phase3_cavity_methods import CRES_RightCylinder_Cavity
import numpy as np
import sys
import os
from multiprocessing import Pool

pi = np.pi

length = 0.25 #m
radius = 0.12 #m

my_cavity = CRES_RightCylinder_Cavity(radius, length)
freqs = my_cavity.get_mode_frequencies(1100, 60)

f = 26e9
print(f/1e9)
resolution = 300
bw=2e9

rel_modes = my_cavity.find_mode_density(f, bw)[1]
print(len(rel_modes))

#get all the Qs
my_cavity.calculate_inv_loaded_qs()

electron_position = [(0.01, 0 , length/4),
                     (my_cavity.radius/2, 0 , length/4)
                     ]
electron_dipole = (1.0e5, 0, 0)
zprobe = length/2
phiprobe = 0
directory='unperturbed/f26bw2/'

def zslice_for_eposition(ex):
    if not os.path.exists(directory):
        os.makedirs(directory)
    zslice = my_cavity.field_map_2d_zslice(rel_modes, zprobe, ex, electron_dipole, f, plotz=False )
    np.savetxt('{2}/f25_zslice_er{0}_ez{1}_zhalf_Er_real.txt'.format(ex[0],ex[2], directory),zslice[0])
    np.savetxt('{2}/f25_zslice_er{0}_ez{1}_zhalflength_Ephi_imag.txt'.format(ex[0],ex[2], directory),zslice[1])
    plt.savefig('{2}/f26_zslice_er{0}_ez{1}zhalflength_.png'.format(ex[0], ex[2], directory), bbox_inches = 'tight', dpi=resolution)
    plt.close()

if __name__ == '__main__':
    pool = Pool(processes=2)
    pool.map(zslice_for_eposition, electron_position)



#for ex in electron_position:
#    directory='unperturbed/f26bw2/'
#    if not os.path.exists(directory):
#        os.makedirs(directory)
#    zslice = my_cavity.field_map_2d_zslice(rel_modes, zprobe, ex, electron_dipole, f, plotz=False )
##    if save_text:
##        np.savetxt('{3}/f{0[0]}{0[1]}{0[2]}{0[3]}_zslice_er{1}_ez{2}_zhalflength_Er.txt'.format(mode, ex[0],ex[2], directory),zslice[0])
##        np.savetxt('{3}/f{0[0]}{0[1]}{0[2]}{0[3]}_zslice_er{1}_ez{2}_zhalflength_Ephi.txt'.format(mode, ex[0],ex[2],directory), zslice[1])
#    plt.savefig('{2}/f26_zslice_er{0}_ez{1}zhalflength_.png'.format(ex[0], ex[2], directory), bbox_inches = 'tight', dpi=resolution)
#    plt.close()
#
#    phislice = my_cavity.field_map_2d_phislice(rel_modes, 0, ex, electron_dipole, f, plotz=False)
#    plt.savefig('{2}/f26_phislice_er{0}_ez{1}_phi0.png'.format(ex[0], ex[2], directory), bbox_inches = 'tight', dpi=resolution)
#    plt.close()
#
#
