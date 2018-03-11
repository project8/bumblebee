from scipy.special import jv, jn, jvp, jnjnp_zeros
import scipy.integrate as integrate
import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
import sys

c = constants.speed_of_light
pi = constants.pi
mu = constants.mu_0
I = 1j

max_zeros = 1200 #maximum number of bessel function zeros to calculate

# calculate bessel function zeros.
bess_zo, bess_n, bess_m, bess_t = jnjnp_zeros(max_zeros)
# n is the order of the bessel function, m is the serial number of the zero
# within the bessel order. Note, this convention is the reverse of what Jackson
# uses, but it's consistent with Pozar and the Scipy documentation.
# also note that bess_t = 0 corresponds to the zeros of Jn, and bess_t = 1
# corresponds to the zeros of the Jn'.

# now we make a matrix of bessel function zeros. The rows are labeled by n (the
# bessel function order) and the columns are labeled by m (the serial number of
# the zero within the bessel function order). We two separate matrices, one for
# Jn zeros and the other corresponding to Jn'.
bess_zeros = np.zeros((np.max(bess_n)+1, np.max(bess_m)+1))
bessprime_zeros = np.copy(bess_zeros)

for i in range(len(bess_zo)):
    if bess_t[i] == 0: #Jn
        bess_zeros[bess_n[i],bess_m[i]] = bess_zo[i]
    else: #besst=1 corresponding to Jn'
        bessprime_zeros[bess_n[i],bess_m[i]] = bess_zo[i]

class CRES_RightCylinder_Cavity:
    """A model of a right circular cylinder used for CRES"""
    default_mode_q = 1000
    linear_resolution = 0.001
    angular_resolution = pi/180
    antenna_positions = []
    antenna_dipoles = []
    antenna_impedances = []
    mode_freqs = {}
    sorted_freqs_modes = []
    inv_q_by_mode = {} # this is the inverse of the Q-Factor. This makes the math
                       # a bit cleaner.

    def __init__(self, radius, length):
        self.radius = radius
        self.length = length

    def add_antenna(self, location, dipole, impedance = 73):
        """location and dipole are in (r, theta, z). Impedance is typically a
        complex value. A default value of 73 ohms, corresponding to the
        impedence of a half-wave dipole antenna, is put here as a
        placeholder."""
        self.antenna_positions.append(location)
        self.antenna_dipoles.append(dipole)
        self.antenna_impedances.append(impedance)

    def get_mode_frequencies(self, max_bess, max_l):
        """Generate a list of mode names and frequencies and return an array
        of: (frequency in Hz, 0/1 for TM/TE, (n, m, l)"""
        modes = []
        for i in range(min(len(bess_zo), max_bess) + 1):
            for j in range(max_l + 1):
                if bess_t[i] == 1 and (j == 0 or bess_m[i]==0):
                    continue
                if bess_t[i] == 0 and(bess_m[i] == 0):
                    continue
                modeid = (bess_t[i], bess_n[i], bess_m[i], j)
                freq = c/(2*pi)*((bess_zo[i]/self.radius)**2 +
                                 (j*pi/self.length)**2)**(1/2)
                self.mode_freqs[modeid] = freq
                modes.append((freq, modeid))
        modes.sort()
        self.sorted_freqs_modes = modes
        return modes

    def find_mode_density(self, central_frequency, bandwidth,
                          density_file = None):
        """Generate a list of modes around a central frequency within a
        bandwidth. Also tells you the number of modes within that bandwidth."""
        low_f = central_frequency - bandwidth/2
        high_f = central_frequency + bandwidth/2
        relevant_freqs_modes = []
        for f in self.sorted_freqs_modes:
            if (f[0] > low_f) & (f[0] < high_f):
                relevant_freqs_modes.append(f)
        n_modes = len(relevant_freqs_modes)
        if density_file is not None:
            with open(density_file, 'w') as ofile:
                ofile.write('# modes = {}, radius = {} m, length = {} m, cf = {} GHz, bw = {} MHz \n\n'.format(n_modes, self.radius, self.length, central_frequency/1e9, bandwidth/1e6))
                ofile.write('# GHz (TM or TE, n, m, l), n refers to radial mode, m refers to azimuthal mode, l refers to longitudinal mode\n')
                for f in relevant_freqs_modes:
                    ofile.write("{} {}\n".format(f[0]/1e9, f[1]))
        return n_modes, relevant_freqs_modes


    def get_mode_normalization(self, modeid, normalize_to=1):
        #TODO choose proper value to normalize to.

        tm_or_te, n, m, l = modeid
        w = 2*pi*self.mode_freqs[modeid]
        #TODO do TM mode normalization

        if tm_or_te ==1:
            #TODO make the integration look prettier
            te_zero = bessprime_zeros[n][m]
            kr = te_zero/self.radius
            norm = 2*kr/(mu*w)*1/np.sqrt(integrate.quad(lambda r: r*n/(kr*r)**2*jv(n, kr*r)**2 + r*jvp(n,kr*r)**2, 0, self.length))[0]
            return norm

    def get_mode_e_field(self, modeid, x):
        """given the mode info and position x, calculate the electric field"""
        r = x[0]
        phi = x[1]
        z = x[2]
        tm_or_te, n, m, l = modeid
        kz = np.pi*l/self.length
        w = 2*pi*self.mode_freqs[modeid]
#        norm = 1
        norm = self.get_mode_normalization(modeid)

        if tm_or_te == 0: #TM
            #TODO fix TM mode
            tm_zero = bess_zeros[n][m]
            kr = tm_zero/self.radius
            Ez = norm * np.sin(n*phi) * jv(n, kr*r) * np.cos(kz*z)
            Er = (-I*kz/kr*norm * np.sin(n*phi) * jvp(n, kr*r)
                  * (-I*np.sin(kz*z)))
            Ephi = (-I*kz*n/(kr**2*r)*norm * np.cos(n*phi) * jv(n, kr*r)
                    * (-I*np.sin(kz*z)))
        else: #TE
            te_zero = bessprime_zeros[n][m]
            kr = te_zero/self.radius
            Er = -w*mu*n/(kr**2*r) * np.cos(n*phi) * jv(n, kr*r) *np.sin(kz*z)
            Ephi = -w*mu/kr*norm * np.sin(n*phi) * jvp(n, kr*r) * np.sin(kz*z)
            Ez = np.zeros_like(Er)
        return (Er, Ephi, Ez)

#    def get_antenna_mode_inv_q(self, modeid, antenna_num):
#        """given mode number and antenna number, figure out what the Q of the
#        antenna is. Remember antennas are labeled indexed by their number. So
#        the antenna number will give you the position and dipole."""
#        #TODO deal with non-linearity of solving for external Qs. This is all wrong
#        xi = self.antenna_positions[antenna_num]
#        di = self.antenna_dipoles[antenna_num]
#        Zi = self.antenna_impedances[antenna_num]
#        E_xi = self.get_mode_e_field(modeid, xi)
#        w0 = 2*pi*self.mode_freqs[modeid]
#        di_mag = np.dot(np.conjugate(di), di)**(1/2)
#        inv_q = np.dot(E_xi, di)*np.dot(np.conjugate(E_xi), di)/(2.0*w0*Zi*di_mag)
#        return inv_q

    def get_mode_inv_loaded_q(self, modeid):
        """Calculate the loaded Q of a mode given all the antennas"""
        # TODO fix external Q
        # for now, just use unloaded Q
        inv_qsum = 1/self.default_mode_q
        #inv_qsum = 1/self.default_mode_q
        #for i in range(len(self.antenna_positions)):
        #    inv_qsum += self.get_antenna_mode_inv_q(modeid, i)
        self.inv_q_by_mode[modeid] = inv_qsum
        return inv_qsum

    def calculate_inv_loaded_qs(self):
        """Calculates the loaded Qs for all modes simulated after antennas are
        added"""
        for mode in self.mode_freqs:
            self.get_mode_inv_loaded_q(mode)

    def calculate_source_mode_excitation(self, modeid, location, dipole, freq):
        """Calculate the field amplitude A for a certain mode given a given
        dipole source"""
        pe = dipole
        E_xe = self.get_mode_e_field(modeid, location)
        w = 2*pi*freq
        w0 = 2*pi*self.mode_freqs[modeid]
        inv_q = self.inv_q_by_mode[modeid]
        # TODO change inv_q to inv_loaded_q throughout entire script
        A = w**2 * np.dot(pe,np.conjugate(E_xe))/(2*(w**2-w0**2+I*w*w0*inv_q))
        return A

    def precalculate_antenna_mode_couplings(self):
        """Basically, just calculate Psi dot d for all antennas and modes, and
        turn that into a matrix. The first index (row number) of the matrix will
        refer to the mode label. The second index (column number) will refer to
        the antenna number."""
        n_modes = len(self.sorted_freqs_modes)
        n_antennas = len(self.antenna_positions)
        self.Psi_dot_d = np.zeros((n_modes, n_antennas), dtype=np.complex_)
        for i in range(n_modes):
            for j in range(n_antennas):
                xj = self.antenna_positions[j]
                dj = self.antenna_dipoles[j]
                E_xj = self.get_mode_e_field(self.sorted_freqs_modes[i][1], xj)
                #print("Field at antenna {}".format(E_xj))
                self.Psi_dot_d[i,j] = np.dot(dj, E_xj)
        return self.Psi_dot_d

    def calculate_antenna_voltages(self, electron_x, electron_dipole, frequency):
        n_modes = len(self.sorted_freqs_modes)
        A = np.zeros(n_modes, dtype=np.complex_)
        for i in range(n_modes):
            A[i] = self.calculate_source_mode_excitation(self.sorted_freqs_modes[i][1],
                                                         electron_x,
                                                         electron_dipole,
                                                         frequency)
        # for i in range(n_modes):
            # print('{0}\t{1.real:.2E} + {1.imag:.2E}i \t {2:.3f} GHz'.format(self.sorted_freqs_modes[i][1], A[i], self.sorted_freqs_modes[i][0]/1e9))
        self.antenna_voltage = np.dot(A.T,self.Psi_dot_d)
        return self.antenna_voltage


    def field_map_2d_zslice_single_mode(self, modeid, z, plotz=True):
        """For a fixed z, plot the field as a function of r and phi"""
        length = self.length
        phi = np.arange(0, 2*pi, self.angular_resolution)
        r = np.arange(0.001, self.radius, self.linear_resolution)
        phiphi, rr = np.meshgrid(phi, r)

        zslice = self.get_mode_e_field(modeid, (rr, phiphi, z))
        zslice_real = np.real(zslice)

        cm = 'bwr'
        scale = abs(max(zslice_real.min(), zslice_real.max(), key=abs))
        f0 = self.mode_freqs[modeid]

        if plotz:
            fig, ax = plt.subplots(1,3, figsize=(8,3.5),
                                   subplot_kw= dict(projection= 'polar'))
            ax[0].pcolormesh(phi, r, zslice_real[0], cmap = cm,
                           vmin = -scale, vmax=scale)
            ax[0].set_yticklabels([])
            ax[0].grid()
            ax[0].set_title('Er', y = 1.2)

            ax[1].pcolormesh(phi, r, zslice_real[1], cmap = cm,
                           vmin = -scale, vmax=scale)
            ax[1].set_yticklabels([])
            ax[1].grid()
            ax[1].set_title('Ephi', y = 1.2)

            im = ax[2].pcolormesh(phi, r, zslice_real[2], cmap = cm,
                           vmin = -scale, vmax=scale)
            ax[2].set_yticklabels([])
            ax[2].grid()
            ax[2].set_title('Ez', va = 'bottom', y = 1.2)
            fig.subplots_adjust(wspace=0.8)

            fig.suptitle('Mode = {}, f = {:.3f} GHz, z = {} m'.format(modeid, f0/1e9, z), ha = 'center',
                         va='top', x=0.43)
            fig.tight_layout()
            fig.colorbar(im, ax=ax.ravel().tolist(), shrink = 0.5)

        if not plotz:
            fig, ax = plt.subplots(1,2, figsize=(8,3.5),
                                   subplot_kw= dict(projection= 'polar'))
            ax[0].pcolormesh(phi, r, zslice_real[0], cmap = cm,
                           vmin = -scale, vmax=scale)
            ax[0].set_yticklabels([])
            ax[0].grid()
            ax[0].set_title('Er', y = 1.15)

            im = ax[1].pcolormesh(phi, r, zslice_real[1], cmap = cm,
                                  vmin = -scale, vmax=scale)
            ax[1].set_yticklabels([])
            ax[1].grid()
            ax[1].set_title('Ephi', y = 1.15)

            fig.subplots_adjust(wspace=0.8)

            fig.suptitle('Mode = {}, f = {:.3f} GHz, z = {} m'.format(modeid, f0/1e9, z), ha = 'center',
                         va='top', x=0.43, y =1.05)
            fig.tight_layout()
            fig.colorbar(im, ax=ax.ravel().tolist(), shrink = 0.5)



    def field_map_2d_phislice_single_mode(self, modeid, phi, plotz=True):
        """For a fixed phi, plot the field as a function of r and z"""
        z = np.arange(0, self.length, self.linear_resolution)
        r = np.arange(0.01, self.radius, self.linear_resolution)
        zz, rr = np.meshgrid(z, r)

        phi_slice = self.get_mode_e_field(modeid, (rr, phi, zz))
        phi_slice_real = np.real(phi_slice)
        cm = 'bwr'
        f0 = self.mode_freqs[modeid]
        scale = abs(max(phi_slice_real.min(), phi_slice_real.max(), key=abs))

        if plotz:
            f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (16,3.1), sharey = True)
            ax1.pcolormesh(z, r, phi_slice_real[0], cmap = cm,
                           vmin = -scale, vmax=scale)
            ax1.set_aspect('equal')
            ax1.set_title('Er')
            ax1.set_xlabel('z [m]')
            ax1.set_ylabel('r [m]')

            ax2.set_aspect('equal')
            ax2.pcolormesh(z, r, phi_slice_real[1], cmap = cm,
                           vmin = -scale, vmax=scale)
            ax2.set_title('Ephi')
            ax2.set_xlabel('z [m]')


            ax3.set_aspect('equal')
            im3 = ax3.pcolormesh(z, r, phi_slice_real[2], cmap = cm,
                           vmin = -scale, vmax=scale)
            ax3.set_title('Ez')
            ax3.set_xlabel('z [m]')
            plt.suptitle('Mode = {}, f = {:.3f} GHz, phi = {:.2f} '.format(modeid, f0/1e9, phi),
                         y=1)
            plt.tight_layout()
            f.colorbar(im3)

        if not plotz:
#            f, (ax1, ax2) = plt.subplots(1, 2, figsize = (16,3.1), sharey = True)
            f, (ax1, ax2) = plt.subplots(2, 1, sharex = True)
            ax1.pcolormesh(z, r, phi_slice_real[0], cmap = cm,
                           vmin = -scale, vmax=scale)
            ax1.set_aspect('equal')
            ax1.set_title('Er')
#            ax1.set_xlabel('z [m]')
#            ax1.xaxis.set_ticks_position('none') 
            ax1.set_ylabel('r [m]')

            ax2.set_aspect('equal')
            im2 = ax2.pcolormesh(z, r, phi_slice_real[1], cmap = cm,
                                 vmin = -scale, vmax=scale)
            ax2.set_title('Ephi')
            ax2.set_ylabel('r [m]')
            ax2.set_xlabel('z [m]')

            plt.suptitle('Mode = {}, f = {:.3f} GHz, phi = {:.2f} '.format(modeid, f0/1e9, phi),
                         y=1)
            plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
            cax = plt.axes([0.85, 0.1, 0.075, 0.8])
            plt.colorbar(im2, cax=cax)
#            plt.tight_layout()
            #f.colorbar(im2, cax=cax, orientation='horizontal')


    def field_map_2d_zslice(self, rel_freqs_modes, z, electron_x,
                            electron_dipole, freq):
        """For a fixed z, plot the field as a function of r and phi"""
        length = self.length
        phi = np.arange(0,2*pi, self.angular_resolution)
        r = np.arange(0.001, self.radius, self.linear_resolution)
        phiphi, rr = np.meshgrid(phi, r)
        zslice = [np.zeros_like(phiphi, dtype=np.complex_),
                  np.zeros_like(phiphi, dtype=np.complex_),
                  np.zeros_like(phiphi, dtype=np.complex_)]
        j = 0
        print(len(rel_freqs_modes))
        for f in rel_freqs_modes:
            j += 1
            if j%10 == 0:
                print(j)
            modeid = f[1]
            A = self.calculate_source_mode_excitation(modeid, electron_x,
                                                      electron_dipole, freq)
            zslice_mode = np.dot(A.T, self.get_mode_e_field(modeid, (rr, phiphi, z)))
            for i in range(len(zslice_mode)):
                zslice[i] += zslice_mode[i]
        zslice_real = np.real(zslice)

        cm = 'bwr'
        scale = abs(max(zslice_real.min(), zslice_real.max(), key=abs))
        fig, ax = plt.subplots(1,3, figsize=(8,3.5),
                               subplot_kw= dict(projection= 'polar'))
        ax[0].pcolormesh(phi, r, zslice_real[0], cmap = cm,
                       vmin = -scale, vmax=scale)
        ax[0].set_yticklabels([])
        ax[0].grid()
        ax[0].set_title('Er', y = 1.2)

        ax[1].pcolormesh(phi, r, zslice_real[1], cmap = cm,
                       vmin = -scale, vmax=scale)
        ax[1].set_yticklabels([])
        ax[1].grid()
        ax[1].set_title('Ephi', y = 1.2)

        im = ax[2].pcolormesh(phi, r, zslice_real[2], cmap = cm,
                       vmin = -scale, vmax=scale)
        ax[2].set_yticklabels([])
        ax[2].grid()
        ax[2].set_title('Ez', va = 'bottom', y = 1.2)
        fig.subplots_adjust(wspace=0.8)

        fig.suptitle(r'f = {0:.3f} GHz, z = {1:.3f} m, $r_e$ = {2[0]} $\phi_e$ = {2[1]} $z_e$ = {2[2]}'.format(freq/1e9, z, electron_x),
                     ha = 'center',
                     va='top', x=0.43)
        fig.tight_layout()
        fig.colorbar(im, ax=ax.ravel().tolist(), shrink = 0.5)
        return zslice


    def field_map_2d_phislice(self, rel_freqs_modes, phi, electron_x,
                              electron_dipole, freq):
        """For a fixed phi, plot the field as a function of r and z"""
        z = np.arange(0, self.length, self.linear_resolution)
        r = np.arange(0.001, self.radius, self.linear_resolution)
        zz, rr = np.meshgrid(z, r)
        phi_slice = [np.zeros_like(zz, dtype=np.complex_),
                     np.zeros_like(zz, dtype=np.complex_),
                     np.zeros_like(zz, dtype=np.complex_)]

        j = 0
        for f in rel_freqs_modes:
            j += 1
            if j%10==0:
                print(j)
            modeid = f[1]
            A = self.calculate_source_mode_excitation(modeid, electron_x,
                                                      electron_dipole, freq)
            phi_slice_mode = np.dot(A.T, self.get_mode_e_field(modeid, (rr, phi, zz)))
            for i in range(len(phi_slice_mode)):
                phi_slice[i] += phi_slice_mode[i]

        phi_slice_real = np.real(phi_slice)

        cm = 'bwr'
        scale = abs(max(phi_slice_real.min(), phi_slice_real.max(), key=abs))

        f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey = True, figsize=(16,3.1))
        ax1.set_aspect('equal')
        ax1.pcolormesh(z, r, phi_slice_real[0], cmap = cm,
                       vmin = -scale, vmax=scale)
        ax1.set_title('Er')
        ax1.set_xlabel('z [m]')
        ax1.set_ylabel('r [m]')

        ax2.pcolormesh(z, r, phi_slice_real[1], cmap = cm,
                       vmin = -scale, vmax=scale)
        ax2.set_aspect('equal')
        ax2.set_title('Ephi')
        ax2.set_xlabel('z [m]')


        im3 = ax3.pcolormesh(z, r, phi_slice_real[2], cmap = cm,
                       vmin = -scale, vmax=scale)
        ax3.set_aspect('equal')
        ax3.set_title('Ez')
        ax3.set_xlabel('z [m]')
        plt.suptitle('f = {0:.3f} GHz, phi = {1:.2f}, $r_e$ = {2[0]} $\phi_e$ = {2[1]} $z_e$ = {2[2]}'.format(freq/1e9, phi, electron_x),
                     y=1)
        plt.tight_layout()
        f.colorbar(im3)
        return phi_slice

