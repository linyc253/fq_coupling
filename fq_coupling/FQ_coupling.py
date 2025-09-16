import numpy as np
import pandas as pd
import qutip as qt
import math
from io import StringIO
from scipy.linalg import block_diag
from scipy.optimize import root_scalar

# Calculate coupling strength g_ij between floating transmons qubits from capacitance matrix
# Reference: http://dx.doi.org/10.1103/PhysRevApplied.15.064063 (APPENDIX B)

# Input: Capacitance matrix C (in fF) from Ansys Q3D
# SignalNet: GND, Q0_L (pad1), Q0_R (pad2), Q0_xy (xy line), Q0_read (readout line), C0_L, ......
# where Q* represents qubit, and C* represents coupler

class Couple():
    '''Read capacitance matrix to initialize (csv file), note that the unit of capacitance must be fF'''
    def __init__(self, filename, fr=6.0, quarter=True):
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
        self.C = pd.read_csv(StringIO("\n".join(table_lines)), index_col=0)
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
                self.C.loc[i, i] += self.C.loc[self.C.index[i], "GND"] + Cr # Add resonator capacitance to ground
        
        # Define class variables (should be read only)
        self.fr, self.Cr = fr, Cr
        self.qubit_list = [name.split('_')[0] for name in self.C.columns if name.endswith('_L')]
        self.Nq = len(self.qubit_list)
        self.EC_matrix, self.EC_readout = self._get_Ec_matrix()
        self.EC = self.EC_matrix.diagonal()
    
    # Calculate Ec matrix from capacitance matrix, eliminating the redundant degree of freedom
    def _get_Ec_matrix(self):
        e = 1.60217657e-19  # electron charge
        h = 6.62606957e-34  # Plank's constant
        # Only keep columns and rows with names ending in '_L' (left pad of floating qubit) or '_R' (right pad of floating qubit)
        # or '_read' (readout resonator coupling pad) or '_I' (floating island of single-ended qubit)
        selected = [name for name in self.C.columns if name.endswith('_L') or name.endswith('_R') or name.endswith('_read') or name.endswith('_I')]
        C_matrix = self.C.loc[selected, selected].to_numpy()

        # Transfrom capacitance matrix to remove the redundant DOF
        blocks = []
        for i in range(len(selected)):
            if selected[i].endswith('_L'):
                blocks.append(np.array([[1, -1], [1, 1]])) # transform: (L, R) = (L-R, L+R). Then, we'll drop R (L+R) to remove the redundant DOF
            elif selected[i].endswith('_read') or selected[i].endswith('_I'):
                blocks.append(np.array([[1]]))
            #elif selected[i].endswith('_R'): -> do nothing
        U = block_diag(*(blocks))

        C_matrix = np.linalg.inv(U.T) @ C_matrix @ np.linalg.inv(U)

        # Inverse of reduced capacitance matrix
        reduced = np.array([i for i in range(len(selected)) if selected[i].endswith('_L') or selected[i].endswith('_I')])
        assert len(reduced) == self.Nq, "Size of EC_matrix wrong!! Please check the naming convention of capacitance matrix."
        C_inv = np.linalg.inv(C_matrix)
        EC_matrix = e**2 / (2 * h) * C_inv[reduced, :][:, reduced] * 1e6 # Ec in GHz, C in fF

        # For readout coupling strength
        EC_readout = []
        for i in reduced:
            read_tag = selected[i].split('_')[0] + 'read'
            if read_tag in selected:
                EC_readout.append(C_inv[selected.index(read_tag), i])
            else:
                EC_readout.append(0)
        EC_readout = e**2 / (2 * h) * np.array(EC_readout)

        return EC_matrix, EC_readout
    
    
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
            EJ_sol.append(root_scalar(func, bracket=[5, 30])['root'])
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

    def get_eig(self, EJ, fast=True):
        '''Calculate eigenvalues of Hamiltonian, `EJ` in GHz'''
        if fast:
            H, _ = self._Hamiltonian_fast(EJ)
        else:
            H, _ = self._Hamiltonian(EJ)
        eigenvalues = H.eigenenergies()
        return eigenvalues

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

        overlap_000 = []
        overlap_001 = []
        overlap_100 = []
        overlap_101 = []

        for ii in range(dim*self.Nq):
            overlap_000.append(abs(s000.dag()*eigenstates[ii])) 
            overlap_001.append(abs(s001.dag()*eigenstates[ii]))
            overlap_100.append(abs(s100.dag()*eigenstates[ii]))
            overlap_101.append(abs(s101.dag()*eigenstates[ii]))
        
        E_000 = eigenvalues[np.argmax(np.array(overlap_000))]
        E_001 = eigenvalues[np.argmax(np.array(overlap_001))]
        E_100 = eigenvalues[np.argmax(np.array(overlap_100))]
        E_101 = eigenvalues[np.argmax(np.array(overlap_101))]

        zz = (E_101 - E_001) - (E_100 - E_000)
        return zz

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

        df_1q = pd.DataFrame(np.transpose([self.EC * 1e3, EJ, freq, anharmonicity * 1e3, g_rq * 1e3]), index=self.qubit_list, columns=['EC (MHz)', 'EJ (GHz)', 'Frequency (GHz)', 'Anharmonicity (MHz)', 'g_rq (MHz)'])
        df_gij = pd.DataFrame(g_ij * 1e3, index=self.qubit_list, columns=self.qubit_list)
        return df_1q, df_gij

    ### Legacy code ###
    def calculate_freq_and_gij(self, EJ, print_result=False): # EJ in GHz      
        # Calculate qubit frequencies
        zeta = (2*self.EC / EJ)**0.5
        freq = np.sqrt(8 * EJ * self.EC) - self.EC * (1 + zeta / 4)


        # Calculate coupling strengths
        g_ij = self.EC_matrix / 2**0.5 * ((EJ / self.EC)**0.25)[:, None] * ((EJ / self.EC)**0.25)[None, :] * (1 - zeta[:, None] / 8 - zeta[None, :] / 8)
        for i in range(self.Nq):
            g_ij[i, i] = 0
        g_ij *= 2 # (Q1 Q2) * g_ij * (Q1 Q2)^T = g_11 * Q1^2 + g_22 * Q2^2 + 2*g_12*Q1*Q2

        if print_result:
            print(pd.DataFrame(np.transpose([self.EC * 1e3, freq]), index=self.qubit_list, columns=['Ec (MHz)', 'Frequency (GHz)']))
            print("\nCoupling strengths g_ij (MHz):")
            print(pd.DataFrame(g_ij * 1e3, index=self.qubit_list, columns=self.qubit_list))

        return freq, g_ij, self.EC, zeta
    
    def get_readout_g(self, EJ, fr, quarter=True):
        e = 1.60217657e-19  # electron charge
        h = 6.62606957e-34  # Planck's constant
        G = []
        for q in self.qubit_list:
            if f"{q}_read" not in self.C.columns:
                continue
            selected = [f"{q}_read", f"{q}_L", f"{q}_R"]
            if quarter:
                Cr = 0.25 * np.pi / (2*np.pi * fr) / 50 * 1e6 # For lambda/4 resonator (fF)
            else:
                Cr = 0.5 * np.pi / (2*np.pi * fr) / 50 * 1e6 # For lambda/2 resonator (fF)

            C_matrix = self.C.loc[selected, selected].to_numpy()
            C_matrix[0, 0] += self.C.loc[f"{q}_read", "GND"] + Cr # Add resonator capacitance to ground

            U = block_diag(*([1] + [np.array([[1, -1], [1, 1]])]))
            C_transform = np.linalg.inv(U.T) @ C_matrix @ np.linalg.inv(U)
            C_inv = np.linalg.inv(C_transform)[:2, :2]
            Ec_matrix = e**2 / (2 * h) * C_inv * 1e6 # Ec in GHz, C in fF

            # Calculate qubit frequencies
            Ec = Ec_matrix[1, 1]
            zeta = (2*Ec / EJ)**0.5
            freq = np.sqrt(8 * EJ * Ec) - Ec * (1 + zeta / 4)

            # Calculate coupling strengths
            g = Ec_matrix[0, 1] / 2**0.5 * (EJ / Ec)**0.25 * (fr**2 / 2 / (e**2 / Cr / h * 1e6)**2)**0.25 * (1 - zeta / 8)
            g *= 2 # (Q1 Q2) * g_ij * (Q1 Q2)^T = g_11 * Q1^2 + g_22 * Q2^2 + 2*g_12*Q1*Q2
            print(f"Qubit {q}: freq = {freq:.3f} GHz, Ec = {Ec*1e3:.1f} MHz, g = {g*1e3:.1f} MHz")
            G.append(g)
        return np.array(G)
        
