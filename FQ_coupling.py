import numpy as np
import pandas as pd
import qutip as qt
import matplotlib.pyplot as plt
import math
from io import StringIO
from scipy.linalg import block_diag
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams['font.family'] = "STIXGeneral"
plt.rcParams['font.size'] = 12


# Calculate coupling strength g_ij between floating transmons qubits from capacitance matrix
# Reference: http://dx.doi.org/10.1103/PhysRevApplied.15.064063 (APPENDIX B)

# Input: Capacitance matrix C (in fF) from Ansys Q3D
# SignalNet: GND, Q0_L (pad1), Q0_R (pad2), Q0_xy (xy line), Q0_read (readout line), C0_L, ......
# where Q* represents qubit, and C* represents coupler

class Couple():
    def __init__(self, filename):
        with open(filename, "r") as f:
            lines = f.readlines()

        capture = False
        table_lines = []

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

        self.C = pd.read_csv(StringIO("\n".join(table_lines)), index_col=0)
        self.qubit_list = [name.split('_')[0] for name in self.C.columns if name.endswith('_L')]
        self.Nq = len(self.qubit_list)
    
    def get_Ec_matrix(self):
        e = 1.60217657e-19  # electron charge
        h = 6.62606957e-34  # Plank's constant
        # Only keep columns and rows with names ending in '_L' or '_R'
        selected = [name for name in self.C.columns if name.endswith('_L') or name.endswith('_R')]
        C_matrix = self.C.loc[selected, selected].to_numpy()

        # Transfrom capacitance matrix to remove the redundant DOF
        num_blocks = C_matrix.shape[0] // 2
        U = block_diag(*([np.array([[1, -1], [1, 1]])]*num_blocks))

        C_reduced = np.linalg.inv(U.T) @ C_matrix @ np.linalg.inv(U)
        qubits = [name.split('_')[0] for name in selected[::2]]

        # Inverse of reduced capacitance matrix
        C_inv = np.linalg.inv(C_reduced)[::2, ::2]
        Ec_matrix = e**2 / (2 * h) * C_inv * 1e6 # Ec in GHz, C in fF
        return Ec_matrix

    def calculate_freq_and_gij(self, EJ, print_result=False): # EJ in GHz
        Ec_matrix = self.get_Ec_matrix()
        
        # Calculate qubit frequencies
        Ec = Ec_matrix.diagonal()
        zeta = (2*Ec / EJ)**0.5
        freq = np.sqrt(8 * EJ * Ec) - Ec * (1 + zeta / 4)


        # Calculate coupling strengths
        g_ij = Ec_matrix / 2**0.5 * ((EJ / Ec)**0.25)[:, None] * ((EJ / Ec)**0.25)[None, :] * (1 - zeta[:, None] / 8 - zeta[None, :] / 8)
        for i in range(self.Nq):
            g_ij[i, i] = 0
        g_ij *= 2 # (Q1 Q2) * g_ij * (Q1 Q2)^T = g_11 * Q1^2 + g_22 * Q2^2 + 2*g_12*Q1*Q2

        if print_result:
            print(pd.DataFrame(np.transpose([Ec * 1e3, freq]), index=self.qubit, columns=['Ec (MHz)', 'Frequency (GHz)']))
            print("\nCoupling strengths g_ij (MHz):")
            print(pd.DataFrame(g_ij * 1e3, index=self.qubit, columns=self.qubit))

        return freq, g_ij, Ec, zeta

    def Hamiltonian_fast(self, EJ, dim=3):
        freq, g_ij, Ec, zeta = self.calculate_freq_and_gij(EJ)

        H = 0
        # \sum_i \omega_i (a^dagger a) + Ec_i/2 (1 + zeta_i/4 - (1 + 9*zeta_i/16) a^dagger a) a^dagger a
        for i in range(self.Nq):
            H_sub = (freq[i] + Ec[i] / 2 * ((1 + zeta[i] / 4) - (1 + 9 * zeta[i] / 16) * qt.num(dim))) * qt.num(dim)
            H += qt.tensor([H_sub if j == i else qt.qeye(dim) for j in range(self.Nq)])
        
        # \sum_{i<j} -g_ij (a^dagger - a)(b^dagger - b)
        for i in range(self.Nq): 
            for j in range(i+1, self.Nq):
                H -= g_ij[i, j] * qt.tensor([ (qt.create(dim) - qt.destroy(dim)) if k == i else (qt.create(dim) - qt.destroy(dim)) if k == j else qt.qeye(dim) for k in range(self.Nq)])
        return H, dim

    def Hamiltonian(self, EJ, dim=10):
        Ec_matrix = self.get_Ec_matrix()

        Ec = Ec_matrix.diagonal()

        # Construct Hamiltonian
        n_ZPF = 0.5**0.5 * (EJ / Ec / 8)**0.25
        n_hat = []
        for i in range(self.Nq):
            n_hat.append(qt.tensor([1j * n_ZPF[i] * (qt.create(dim) - qt.destroy(dim)) if j == i else qt.qeye(dim) for j in range(self.Nq)]))

        phi_ZPF = 0.5**0.5 * (8 * Ec / EJ)**0.25
        phi_hat = []
        for i in range(self.Nq):
            phi_hat.append(qt.tensor([phi_ZPF[i] * (qt.create(dim) + qt.destroy(dim)) if j == i else qt.qeye(dim) for j in range(self.Nq)]))

        H = 0
        for i in range(self.Nq): 
            for j in range(self.Nq):
                H += 4 * Ec_matrix[i, j] * n_hat[i] * n_hat[j]
        for i in range(self.Nq):
            for n in range(dim):
                H -= EJ[i] * (-1)**n * phi_hat[i]**(2*n) / math.factorial(2*n)
        return H, dim

    def get_eig(self, EJ, fast=True):
        if fast:
            H, dim = self.Hamiltonian_fast(EJ)
        else:
            H, dim = self.Hamiltonian(EJ)
        eigenvalues = H.eigenenergies()
        return eigenvalues

    def get_zz(self, EJ, q0, q1, fast=False):
        if fast:
            H, dim = self.Hamiltonian_fast(EJ)
        else:
            H, dim = self.Hamiltonian(EJ)
        eigenvalues, eigenstates = H.eigenstates()
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

    def get_readout_g(self, EJ, fr):
        e = 1.60217657e-19  # electron charge
        h = 6.62606957e-34  # Planck's constant

        names = [name for name in self.C.columns if name.endswith('_L')]
        qubits = [name.split('_')[0] for name in names]
        G = []
        for q in qubits:
            if f"{q}_read" not in self.C.columns:
                continue
            selected = [f"{q}_read", f"{q}_L", f"{q}_R"]
            Cr = 0.25 * np.pi / (2*np.pi * fr) / 50 * 1e6 #For lambda/4 resonator (fF)
            # Cr = 0.5 * np.pi / (2*np.pi * fr) / 50 * 1e6 #For lambda/2 resonator (fF)

            C_matrix = self.C.loc[selected, selected].to_numpy()
            C_matrix[0, 0] += C.loc[f"{q}_read", "GND"] + Cr # Add resonator capacitance to ground

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
        

if __name__ == '__main__':
    system = Couple("2FQ1FC-H_q3d.csv")
    EJc_list = np.linspace(5, 15, 20)
    fc_list = []
    zz_list = []
    Eig = []
    
    for EJc in EJc_list:
        freq, g_ij, Ec, zeta = system.calculate_freq_and_gij(np.array([EJc, 17.39, 15.40]))
        fc_list.append(freq[0])
        zz_list.append(system.get_zz(np.array([EJc, 17.39, 15.40]), 1, 2) * 1e3)
    plt.figure()
    plt.plot(fc_list, zz_list)
    plt.xlabel("$\omega_c / 2\pi$ (GHz)")
    plt.ylabel("zz (MHz)")
    plt.show()
