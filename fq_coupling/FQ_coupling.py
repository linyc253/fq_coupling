import numpy as np
import pandas as pd
import qutip as qt
import math
from io import StringIO
from scipy.linalg import block_diag
from scipy.optimize import root_scalar
import pkg_resources
import re

# Calculate coupling strength g_ij between floating transmons qubits from capacitance matrix
# Reference: http://dx.doi.org/10.1103/PhysRevApplied.15.064063 (APPENDIX B)

# Input: Capacitance matrix C (in fF) from Ansys Q3D
# SignalNet: GND, Q0_L (pad1), Q0_R (pad2), Q0_xy (xy line), Q0_read (readout line), C0_L, ......
# where Q* represents qubit, and C* represents coupler

def capacitance_reader(filename, internal_data=False):
    '''Read capacitance matrix to initialize (csv file), note that the unit of capacitance must be fF'''
    if internal_data:
        filename = pkg_resources.resource_filename('fq_coupling', 'data/'+filename)

    with open(filename, "r") as f:
        lines = f.readlines()

    capture = False
    table_lines = []

    # Locate capacitance matrix
    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("Capacitance Matrix Coupling Coefficient"):
            break  # stop before the next block

        if line.startswith("Capacitance Matrix"):
            capture = True
            continue

        if capture:
            table_lines.append(line)

    # Parse into DataFrame
    if not table_lines:
        raise ValueError("Capacitance Matrix block not found in file.")

    # Get capacitance matrix
    C = pd.read_csv(StringIO("\n".join(table_lines)), index_col=0)
    if len(C.columns) == len(C.index) + 1:
        C.drop(C.columns[-1], axis=1, inplace=True) # Drop last column
    assert np.array_equal(C.index, C.columns), "BUG: columns and rows index does not match"
    return C


class Couple():
    '''Read capacitance matrix to initialize (csv file) and pre-process, note that the unit of capacitance must be fF'''
    def __init__(self, filename, internal_data=False, fr=6.0, quarter=True, C_prefactor=1):
        self.c_prefactor = C_prefactor # You can scale the capacitance matrix by this prefactor if needed, similar to modifying the dielectric constant in Q3D
        self.C = capacitance_reader(filename, internal_data) *C_prefactor
        # Pre-process to get rid of the stray capacitance to infinity
        for i in range(self.C.shape[0]):
            self.C.iloc[i, i] -= np.sum(self.C.iloc[i, :])
        # Replace C_{read, gnd} by Cr (capacitance of resonator)
        if quarter:
            Cr = 0.25 * np.pi / (2*np.pi * fr) / 50 * 1e6 # For lambda/4 resonator (fF)
        else:
            Cr = 0.5 * np.pi / (2*np.pi * fr) / 50 * 1e6 # For lambda/2 resonator (fF)
        for i in range(self.C.shape[0]):
            if self.C.index[i].endswith('_read'):
                self.C.iloc[i, i] += self.C.loc[self.C.index[i], "GND"] + Cr # Add resonator capacitance to ground
        # Replace C_{xy, gnd} by C_B (very large capacitor)
        C_B = 1E6
        for i in range(self.C.shape[0]):
            if self.C.index[i].endswith('_xy'):
                self.C.iloc[i, i] += self.C.loc[self.C.index[i], "GND"] + C_B # Add resonator capacitance to ground
        
        # Define class variables (should be read only)
        self.fr, self.Cr = fr, Cr
        self.qubit_list = [name.split('_')[0] for name in self.C.columns if name.endswith('_L') or name.endswith('_I')]
        self.Nq = len(self.qubit_list)
        self.EC_matrix, self.EC_readout, self.C_xy = self._get_Ec_matrix()
        self.EC = self.EC_matrix.diagonal()
    
    # Calculate Ec matrix from capacitance matrix, eliminating the redundant degree of freedom
    def _get_Ec_matrix(self):
        e = 1.60217657e-19  # electron charge
        h = 6.62606957e-34  # Plank's constant
        # Check whether all columns and rows end with '_L' (left pad of floating qubit) or '_R' (right pad of floating qubit)
        # or '_read' (readout resonator coupling pad) or '_I' (floating island of single-ended qubit) or '_B(number)' (other floating bus)
        for name in self.C.columns:
            assert name.endswith(('_L', '_R', '_read', '_xy', '_I')) or re.search(r'_B\d+$', name) or name=="GND",\
                   "Labels should be either 'GND' or end with '_L', '_R', '_read', '_xy', '_I', or '_B(number)'"
        # Remove GND column/row (if present)
        selected = [name for name in self.C.columns if name!="GND"]
        C_matrix = self.C.loc[selected, selected].to_numpy()

        # Transfrom capacitance matrix to remove the redundant DOF
        blocks = []
        for i in range(len(selected)):
            if selected[i].endswith('_L'):
                blocks.append(np.array([[1, -1], [1, 1]])) # transform: (L, R) = (L-R, L+R). Then, we'll drop R (L+R) to remove the redundant DOF
            elif selected[i].endswith('_R'):
                pass # -> do nothing
            else:
                blocks.append(np.array([[1]]))
        U = block_diag(*(blocks))

        C_matrix = np.linalg.inv(U.T) @ C_matrix @ np.linalg.inv(U)

        # Inverse the capacitance matrix, then remove non-qubit DOF
        reduced = np.array([i for i in range(len(selected)) if selected[i].endswith('_L') or selected[i].endswith('_I')])
        assert len(reduced) == self.Nq, "BUG: Size of EC_matrix wrong!!"
        C_inv = np.linalg.inv(C_matrix)
        EC_matrix = e**2 / (2 * h) * C_inv[reduced, :][:, reduced] * 1e6 # Ec in GHz, C in fF

        # For readout coupling strength
        EC_readout = []
        for i in reduced:
            read_tag = selected[i].split('_')[0] + '_read'
            if read_tag in selected:
                EC_readout.append(C_inv[selected.index(read_tag), i])
            else:
                EC_readout.append(0)
        EC_readout = e**2 / (2 * h) * np.array(EC_readout) * 1e6 # Ec in GHz, C in fF

        # For xyline coupling strength
        C_xy = []
        for i in reduced:
            xy_tag = selected[i].split('_')[0] + '_xy'
            if xy_tag in selected:
                C_xy.append(C_inv[selected.index(xy_tag), i] / C_inv[selected.index(xy_tag), selected.index(xy_tag)] / C_inv[i, i])
            else:
                C_xy.append(0)

        return EC_matrix, EC_readout, C_xy
    
    
    def _get_zeta_omega(self, EJ):
        '''Formula (B20) in PhysRevApplied.15.064063'''
        zeta = (2*self.EC / EJ)**0.5
        omega = np.sqrt(8 * EJ * self.EC) - self.EC * (1 + zeta / 4)
        return zeta, omega
    
    # Should I add dispersive shift (due to other qubit or resonator) as well?
    def get_freq(self, EJ):
        '''Calculate qubit frequency using formula (B19) in PhysRevApplied.15.064063, `EJ` in GHz'''
        zeta, omega = self._get_zeta_omega(EJ)
        freq = omega - 5 * self.EC * zeta / 32
        return freq
    
    def solve_EJ(self, freq):
        '''Reversely solve for EJ for given qubit frequency, `freq` in GHz'''
        EJ_sol = []
        for i in range(self.Nq):
            def func(Ej):
                EJ = np.array([Ej if k == i else 15 for k in range(self.Nq)])
                return self.get_freq(EJ)[i] - freq[i]
            EJ_sol.append(root_scalar(func, bracket=[0.1, 10000])['root'])
        return np.array(EJ_sol)
    
    def get_anharmonicity(self, EJ):
        '''Calculate anharmonicity using formula (B19) in PhysRevApplied.15.064063, `EJ` in GHz'''
        zeta, _ = self._get_zeta_omega(EJ)
        anharmonicity = -self.EC * (1 + 9 * zeta / 16)
        return anharmonicity
    
    def get_gij(self, EJ):
        '''Calculate coupling strength g_ij using formula (B21) in PhysRevApplied.15.064063, `EJ` in GHz'''    
        zeta, _ = self._get_zeta_omega(EJ)

        g_ij = self.EC_matrix / 2**0.5 * ((EJ / self.EC)**0.25)[:, None] * ((EJ / self.EC)**0.25)[None, :] * (1 - zeta[:, None] / 8 - zeta[None, :] / 8)

        # Set lower-left of the matrix as zero, and scale upper-right by two
        for i in range(self.Nq):
            for j in range(i+1):
                g_ij[i, j] = 0
        g_ij *= 2 # because (n1 n2) * g_ij * (n1 n2)^T = g_11 * n1^2 + g_22 * n2^2 + 2*g_12*n1*n2
                  #                                                                  ^

        return g_ij
    
    def get_grq(self, EJ):
        '''Calculate qubit-resonator coupling strength by generalizing get_gij(), `EJ` in GHz'''
        e = 1.60217657e-19  # electron charge
        h = 6.62606957e-34  # Plank's constant
        zeta, _ = self._get_zeta_omega(EJ)
        g_rq = self.EC_readout / 2**0.5 * (EJ / self.EC)**0.25 * (self.fr**2 / 2 / (e**2 / self.Cr / h * 1e6)**2)**0.25 * (1 - zeta / 8)
        g_rq *= 2 # see get_gij() for details
        return g_rq

    def _Hamiltonian_fast(self, EJ, dim=3):
        '''
        Construct 3-level Hamiltonian using formula (B19) in PhysRevApplied.15.064063, `EJ` in GHz\n
        Faster but slightly less accurate than Hamiltonian(), especially when it comes to zz-interaction
        '''
        g_ij = self.get_gij(EJ)
        zeta, omega = self._get_zeta_omega(EJ)
    
        H = 0
        # \sum_i \omega_i (a^dagger a) + Ec_i/2 (1 + zeta_i/4 - (1 + 9*zeta_i/16) a^dagger a) a^dagger a
        for i in range(self.Nq):
            H_sub = (omega[i] + self.EC[i] / 2 * ((1 + zeta[i] / 4) - (1 + 9 * zeta[i] / 16) * qt.num(dim))) * qt.num(dim)
            H += qt.tensor([H_sub if j == i else qt.qeye(dim) for j in range(self.Nq)])
        
        # \sum_{i<j} -g_ij (a^dagger - a)(b^dagger - b)
        for i in range(self.Nq): 
            for j in range(i+1, self.Nq):
                H -= g_ij[i, j] * qt.tensor([ (qt.create(dim) - qt.destroy(dim)) if k == i else (qt.create(dim) - qt.destroy(dim)) if k == j else qt.qeye(dim) for k in range(self.Nq)])
        return H, dim
    
    def _Hamiltonian(self, EJ, dim=10):
        '''Construct Hamiltonian from cQED textbook, user can increase `dim` for higher accuracy, `EJ` in GHz'''
        n_ZPF = 0.5**0.5 * (EJ / self.EC / 8)**0.25
        n_hat = []
        for i in range(self.Nq):
            n_hat.append(qt.tensor([1j * n_ZPF[i] * (qt.create(dim) - qt.destroy(dim)) if j == i else qt.qeye(dim) for j in range(self.Nq)]))

        phi_ZPF = 0.5**0.5 * (8 * self.EC / EJ)**0.25
        phi_hat = []
        for i in range(self.Nq):
            phi_hat.append(qt.tensor([phi_ZPF[i] * (qt.create(dim) + qt.destroy(dim)) if j == i else qt.qeye(dim) for j in range(self.Nq)]))

        H = 0
        # Kinetic terms
        for i in range(self.Nq): 
            for j in range(self.Nq):
                H += 4 * self.EC_matrix[i, j] * n_hat[i] * n_hat[j]
        # Expand cosine by Taylor series
        for i in range(self.Nq):
            for n in range(dim):
                H -= EJ[i] * (-1)**n * phi_hat[i]**(2*n) / math.factorial(2*n)
        return H, dim
    
    def _n_hat(self, EJ, dim):
        '''Return a list of n_hat operator for each qubit'''
        n_ZPF = 0.5**0.5 * (EJ / self.EC / 8)**0.25
        n_hat = []
        for i in range(self.Nq):
            n_hat.append(qt.tensor([1j * n_ZPF[i] * (qt.create(dim) - qt.destroy(dim)) if j == i else qt.qeye(dim) for j in range(self.Nq)]))
        return n_hat

    def get_eig(self, EJ, fast=True):
        '''Calculate eigenvalues of Hamiltonian, `EJ` in GHz'''
        if fast:
            H, _ = self._Hamiltonian_fast(EJ)
        else:
            H, _ = self._Hamiltonian(EJ)
        eigenvalues = H.eigenenergies()
        return eigenvalues
    
    def _max_overlap_index(self, states, s):
        '''Calculate |<`s`|`si`>| for each `si` in `states`, and return the index of `si` with maximal overlap'''
        overlap = []
        for i in range(np.size(states)):
            overlap.append(abs(s.dag()*states[i]))
        return np.argmax(np.array(overlap))


    def get_zz(self, EJ, q0: int, q1: int, fast=False):
        '''
        Calculate the zz-interaction between `q0` and `q1`, with `EJ` in GHz\n
        You can check the index by printing the qubit list: `self.qubit_list`
        '''
        if fast:
            H, dim = self._Hamiltonian_fast(EJ)
        else:
            H, dim = self._Hamiltonian(EJ)
        eigenvalues, eigenstates = H.eigenstates()

        # Identify the states by projection
        g = qt.basis(dim, 0)
        e = qt.basis(dim, 1)

        s000 = qt.tensor([g for i in range(self.Nq)])
        s001 = qt.tensor([e if i == q0 else g for i in range(self.Nq)])
        s100 = qt.tensor([e if i == q1 else g for i in range(self.Nq)])
        s101 = qt.tensor([e if i == q0 or i == q1 else g for i in range(self.Nq)])

        
        E_000 = eigenvalues[self._max_overlap_index(eigenstates, s000)]
        E_001 = eigenvalues[self._max_overlap_index(eigenstates, s001)]
        E_100 = eigenvalues[self._max_overlap_index(eigenstates, s100)]
        E_101 = eigenvalues[self._max_overlap_index(eigenstates, s101)]
        

        zz = (E_101 - E_001) - (E_100 - E_000)
        return zz

    def get_epr(self, EJ, q: int, dress=False, fast=False):
        '''
        Calculate the epr of `q`, with `EJ` in GHz\n
        `dress=False` use bare state to calculate epr\n
        `dress=True` use dress state to calculate epr\n
        You can check the index by printing the qubit list: `self.qubit_list`
        '''
        if fast:
            H, dim = self._Hamiltonian_fast(EJ)
        else:
            H, dim = self._Hamiltonian(EJ)
        n_hat = self._n_hat(EJ, dim)

        g = qt.basis(dim, 0)
        e = qt.basis(dim, 1)
        vac = qt.tensor([g for i in range(self.Nq)]) # vacuum state
        s = qt.tensor([e if i == q else g for i in range(self.Nq)]) # bare s

        if dress:
            eigenvalues, eigenstates = H.eigenstates()
            vac = eigenstates[self._max_overlap_index(eigenstates, vac)] # Replace bare vac with dress vac
            s = eigenstates[self._max_overlap_index(eigenstates, s)] # Replace bare s with dress s
        
        # Calculate epr based on n_q
        C_matrix = self.C.to_numpy()
        C_inv = np.linalg.inv(C_matrix)

        q = 0
        n_hat_full = [] # in terms of original capacitance matrix
        for name in self.C.columns:
            if name.endswith('_L'):
                n_hat_full.append(n_hat[q] / 2**0.5)
            elif name.endswith('_R'):
                n_hat_full.append(-n_hat[q] / 2**0.5)
                q += 1
            elif name.endswith('_I'):
                n_hat_full.append(n_hat[q])
                q += 1
            else:
                n_hat_full.append(0)
        
        epr = self.C.copy()
        epr.iloc[:, :] = 0
        Nc = self.C.shape[0]
        for i in range(Nc):
            for j in range(i, Nc):
                for l in range(Nc):
                    for m in range(Nc):
                        epr.iloc[i, j] += -C_matrix[i, j] * (C_inv[i, l] - C_inv[j, l]) * (C_inv[i, m] - C_inv[j, m]) * np.real(s.dag()*n_hat_full[l]*n_hat_full[m]*s)
                        epr.iloc[i, j] -= -C_matrix[i, j] * (C_inv[i, l] - C_inv[j, l]) * (C_inv[i, m] - C_inv[j, m]) * np.real(vac.dag()*n_hat_full[l]*n_hat_full[m]*vac)
        
        e = 1.60217657e-19  # electron charge
        h = 6.62606957e-34  # Plank's constant
        epr *= (2*e)**2 / h * 1e6 # Ec in GHz, C in fF

        # Check total energy by summing epr and U
        phi_ZPF = 0.5**0.5 * (8 * self.EC / EJ)**0.25
        phi_hat = []
        for i in range(self.Nq):
            phi_hat.append(qt.tensor([phi_ZPF[i] * (qt.create(dim) + qt.destroy(dim)) if j == i else qt.qeye(dim) for j in range(self.Nq)]))
        U = 0
        for i in range(self.Nq):
            for n in range(dim):
                U -= EJ[i] * (-1)**n * phi_hat[i]**(2*n) / math.factorial(2*n)
        print("fq = sum(epr) + <U> = ", np.sum(epr.to_numpy())+qt.expect(U, s)-qt.expect(U, vac), "GHz")

        return epr

    def calculate_all(self, EJ):
        '''
        Return two data frames:\n
            df_1q:  single qubit properties\n
            df_gij: coupling strength between qubits (in MHz)
        '''
        freq = self.get_freq(EJ)
        anharmonicity = self.get_anharmonicity(EJ)
        g_rq = self.get_grq(EJ)
        g_ij = self.get_gij(EJ)

        df_1q = pd.DataFrame(np.transpose([self.EC * 1e3, EJ, freq, anharmonicity * 1e3, g_rq * 1e3, self.C_xy]), index=self.qubit_list, 
                             columns=['EC (MHz)', 'EJ (GHz)', 'Frequency (GHz)', 'Anharmonicity (MHz)', 'g_rq (MHz)', 'C_xy (fF)'])
        df_gij = pd.DataFrame(g_ij * 1e3, index=self.qubit_list, columns=self.qubit_list)
        return df_1q, df_gij
