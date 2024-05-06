import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from math import isclose
from scipy.integrate import odeint
from sympy.physics.quantum import Ket
import seaborn as sns
import heapq
sns.set()
sns.set_style('white')
sns.despine(top=True, left=True, right=True, bottom=True)

# LaTeX fonts
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
sorted([f.name for f in matplotlib.font_manager.fontManager.ttflist]);

# Pauli Matrices:
sigma_1 = np.matrix([[0, 1], [1, 0]])
sigma_2 = np.matrix([[0, -1j], [1j, 0]])
sigma_3 = np.matrix([[1, 0], [0, -1]])
sigma_p = (1/2)*(sigma_1 + 1j*sigma_2)
sigma_m = (1/2)*(sigma_1 - 1j*sigma_2)

def sigma_s(n, num, index):
    sigma_list = [sigma_1, sigma_2, sigma_3]
    for i in range(2):
        if index[i] > n-1:
            index[i] = 1
        else:
            index[i] = 0
    return sigma_list[num][index[0],index[1]]

# Folded Kronecker product of sigma_z
def kronk(q):
    sigma_3 = np.array([[1, 0], [0, -1]])
    if (q == 0):
        return(np.identity(1))
    else:
        ans = sigma_3
        for i in range(q-1):
            ans = np.kron(ans, sigma_3)
        return(ans)

# Analogue of the Mathematica Chop function
def Chop(num, tol = 1e-6):
    return(np.where(np.abs(num) < tol, 0., num))

# Class that defines quantum state:
class QuantumState:
    def __init__(self, index, energy, vector, nelectrons, state_string):
        self.i = index
        self.energy = energy
        self.vector = vector
        self.ne = nelectrons
        self.string = state_string

# Class that defines the transition:
class QuantumTransition:
    def __init__(self, qs1, qs2):
        self.i1 = qs1.i
        self.i2 = qs2.i
        self.vec1 = qs1.vector
        self.vec2 = qs2.vector
        self.str1 = qs1.string
        self.str2 = qs2.string
        self.ne1 = qs1.ne
        self.ne2 = qs2.ne
        self.E1 = qs1.energy
        self.E2 = qs2.energy
        self.dE = qs1.energy - qs2.energy
        self.change_ne = False
        if (qs1.ne > qs2.ne):
            self.change_ne = True


# Quantum system class:
class QuantumSystem:
    def __init__(self, ndots, hopping_c, ons_c, coulomb_c, coulomb_intra, mag_c, leadcoup1, leadcoup2):
        # number of quantum dots
        self.l = ndots
        # Hoping Matrix:
        hop = np.zeros((self.l, self.l))
        # Coulomb Interaction:
        Coulombint = np.zeros((self.l, self.l))
        # Spin Interaction:
        Jspin = np.zeros((self.l, self.l))
        # Magnetic Field:
        mag = mag_c*np.ones((self.l, 3))
        # Filling Matrices:
        for i in range(self.l):
            for j in range(self.l):
                if i!=j:
                    try:
                        hop[i][j] = - hopping_c[i][j]
                    except(TypeError):
                        hop[i][j] = - hopping_c
                if(i == j):
                    try:
                        hop[i][j] = ons_c[i]
                    except(TypeError):
                        hop[i][j] = ons_c
                    Coulombint[i][j] = coulomb_intra
                else:
                    Coulombint[i][j] = coulomb_c/(np.abs(i-j) + 1)
                    #Jspin[i][j] =Coulombint[i][j] * np.exp(-np.abs(i-j))
                    Jspin[i][j] = Coulombint[i][j]
        # The dimension of the Hamiltonian matrices:
        self.N = 2**(2*self.l)
        # Define system Hamiltonians for single e- dynamics H_single
        Hsingle = np.zeros((self.N, self.N), dtype="complex128")
        for k in range(self.l):
            for j in range(self.l):
                Hsingle += (hop[k][j]*(
                                np.matmul(self.ac(k), self.ad(j))
                                + np.matmul(self.ac(k + self.l),
                                            self.ad(j + self.l)))
                                  + np.conjugate(np.transpose(
                                          hop[k][j]*(
                                              np.matmul(
                                                  self.ac(k), self.ad(j))
                                              + np.matmul(self.ac(k + self.l),
                                                          self.ad(j + self.l))))))/2.

        # Define system Hamiltonians for single e- dynamics H_single
        HCoulomb = np.zeros((self.N, self.N), dtype="complex128")
        for j in range(self.l):
            for k in range(self.l):
                HCoulomb += Coulombint[j][k]*np.matmul(np.matmul(self.ac(k), self.ad(k)), np.matmul(self.ac(j + self.l), self.ad(j + self.l)))
        for j in range(self.l):
            for k in range(j):
                HCoulomb += Coulombint[j][k]*np.matmul(np.matmul(self.ac(k + self.l), self.ad(
                    k + self.l)), np.matmul(self.ac(j + self.l), self.ad(j + self.l))) + Coulombint[j][k]*np.matmul(np.matmul(self.ac(k), self.ad(k)), np.matmul(self.ac(j), self.ad(j)))
        # Define the Spin matrix:
        Hspin = np.zeros((self.N, self.N), dtype="complex128")
        for k in range(self.l):
            for j in range(self.l):
                for p in range(3):
                        Hspin += Jspin[k][j] * (self.Xrot(k) @ self.Xrot(j) + self.Yrot(k) @ self.Yrot(j) + self.Zrot(k) @ self.Zrot(j))
                        
        # Define the Magnetic field matrix:
        Hmag = np.zeros((self.N, self.N), dtype="complex128")
        for k in range(self.l):
                Hmag += float(mag[k][0]) * self.Xrot(k) + float(mag[k][1]) * self.Yrot(k) + float(mag[k][2]) * self.Zrot(k)
                
        # Total Hamiltonian
        self.Htot = Hsingle + HCoulomb + Hspin + Hmag

        # Eigenvectors and Eigenvalues
        e_val, evec = np.linalg.eig(self.Htot)
        idx = e_val.argsort()
        e_val = np.real(e_val[idx])
        evec = np.transpose(evec)[idx]

        # Normalizing eigenvectors:
        for vec in evec:
            norm = np.sqrt(np.sum(np.abs(vec)**2))
            vec = vec/norm

        # Array of strings with states
        States_string = []
        for i in range(self.N):
            temp = " Eigenstate |E"+str(i+1)+"> = "
            vector = evec[i]
            norm = 0
            for j in range(self.N):
                norm += np.abs(vector[j])**2
            vector = vector/np.sqrt(norm)
            tol = 1e-13
            vector.real[abs(vector.real) < tol] = 0.0
            vector.imag[abs(vector.imag) < tol] = 0.0
            for j in range(self.N):
                if(abs(vector[j]) != 0.0):
                    if temp == " Eigenstate |E"+str(i+1)+"> = ":
                        temp += str(round(vector[j],3)) + self.symbolicstate(j)
                    else:
                        temp+= "  +  " + str(round(vector[j],3)) + self.symbolicstate(j)
                    
            States_string.append(temp)

        # Determine the number of electrons per state
        ecount = np.zeros(self.N, dtype = "int")
        for i in range(self.N):
            for j in range(self.N):
                if(Chop(evec[i][j])!=0):
                    ecount[i] = self.ecount(j)

        # Determine the states:
        self.states = []
        for i in range(self.N):
            self.states.append(QuantumState(i, e_val[i], evec[i], ecount[i], States_string[i]))

        # Determine the transitions:
        self.all_transitions = []
        self.ne_transitions  = []
        for j in range(self.N):
            for i in range(self.N):
                trans = QuantumTransition(self.states[i],self.states[j])
                self.all_transitions.append(trans)
                if(trans.change_ne):
                    self.ne_transitions.append(trans)

        # Lead tunneling:
        self.tlead = np.array([leadcoup1,leadcoup2])
        # Lenergy levels:
        self.Elead = np.zeros([len(self.tlead),len(self.all_transitions)])
        for i in range(len(self.all_transitions)):
            for j in range(len(self.Elead)):
                self.Elead[j][i] = self.all_transitions[i].dE

        # Define population number initial state:
        self.P0 = np.zeros(self.N)
        self.P0[0] = 1.0

        # Define the equation matrix:
        self.eqn2_matrix = np.zeros((self.N, self.N))

        # Define the maximal time:
        self.tmax = 10.0
        self.nintegrate = 1001

    # Electron count function of a specific state
    # (in computation basis)

    def ecount(self, x):
        num = 0
        ind = 2
        if ((len(bin(x))-2) > (2*self.l)):
            ind+=(len(bin(x))-2)-2*self.l
        for x in bin(x)[ind:]:
            num += int(x)
        return num

    # Define creation operator
    # via Jordan-Wigner decomposition

    def ac(self, i):
        i = i + 1
        return(np.kron(np.kron(kronk(i - 1), sigma_m),
                       np.identity(2**(2*self.l - i))))

    # Define creation operator
    # via Jordan-Wigner decomposition
    def ad(self, i):
        i = i + 1
        return(np.kron(np.kron(kronk(i - 1), sigma_p),
                       np.identity(2**(2*self.l - i))))

    # Rotation operators
    def Xrot(self, i):
        q=np.zeros((self.N , self.N), dtype = "complex128")
        for m in range(2):
            for n in range(2):
                q += (1/2) * (np.matmul(self.ac(i + m * self.l)*sigma_s(self.l, 0 ,[i + m * self.l,i + n * self.l]),self.ad(i + n * self.l)))
        return(q)

    def Yrot(self, i):
        q=np.zeros((self.N , self.N), dtype = "complex128")
        for m in range(2):
            for n in range(2):
                q += (1/2) * (np.matmul(self.ac(i + m * self.l)*sigma_s(self.l, 1 ,[i + m * self.l,i + n * self.l]),self.ad(i + n * self.l)))
        return(q)

    def Zrot(self, i):
        q=np.zeros((self.N , self.N), dtype = "complex128")
        for m in range(2):
            for n in range(2):
                q += (1/2) * (np.matmul(self.ac(i + m * self.l) * sigma_s(self.l, 2 ,[i + m * self.l,i + n * self.l]),self.ad(i + n * self.l)))
        return(q)


    # Unitary operator:
    def U(self, t):
        return(expm(-self.Htot*t*1j))

    # Transform mathematical state to symbolic basis
    # (bra- ket- functions)
    def symbolicstate(self, x):
        g = [int(x) for x in bin(x)[2:]]
        gg = np.zeros(2*self.l)
        for i in range(len(g)):
            gg[-i-1] = g[-i-1]
        state = "|"
        for i in range(self.l):
            if((gg[i+self.l]+gg[i]) > 0):
                if(gg[i+self.l] == 1):
                    state += "↑"
                if(gg[i] == 1):
                    state += "↓"
            else:
                state += "0"
            if(i != (self.l-1)):
                state += ";"
        state += ">"
        return(state)

    # Gamma function:
    def Gamma(self, lr, n1, n2):
        G = 0.0
        itrans = 0
        for i in range(len(self.all_transitions)):
            if((self.all_transitions[i].i1 == n1)&(self.all_transitions[i].i2==n2)):
                itrans = i

        for p in range(len(self.Elead[lr])):
            for j in range(self.l):
                if(self.all_transitions[itrans].dE == self.Elead[lr][p]):
                    v1 = self.states[n1].vector
                    v2 = self.states[n2].vector
                    H = self.ad(j) + self.ad(j+self.l)
                    #print(2.0*np.pi*self.tlead[lr][0]*np.abs(np.matmul(v1,np.transpose(np.matmul(H,v2))))**2, j, p, )
                    G += 2.0*np.pi*self.tlead[lr][0]*np.abs(np.matmul(v1,np.transpose(np.matmul(H,v2))))**2
        return(float(G))

    # Fermi function:
    def fermif(self, mu, T):  #CHANGED MU TO A LIST TO ACCOUNT FOR MULTIPLE LEADS. IF CODE DOESN"T RUN, CHECK AGAIN!!!
        fermi = np.zeros((len(self.all_transitions), len(mu)))
        for j in range(len(mu)):
            for i in range(len(self.all_transitions)):
                fermi[i][j] = 1/(np.exp((self.all_transitions[i].dE - mu[j])*11.6/T) + 1)
        return(fermi)

    # Population equations array:
    def eqn2(self, i, mu, T):
        if(self.states[i].ne == 2*self.l):
            for i1 in range(len(self.all_transitions)):
                for kappa in range(len(self.tlead)):
                    if((self.all_transitions[i1].i1==i)&(self.all_transitions[i1].ne2==self.states[i].ne-1)):
                        self.eqn2_matrix[i][self.all_transitions[i1].i2] += self.Gamma(int(kappa),self.all_transitions[i1].i2,i)*self.fermif(mu[int(kappa)],T)[i1]
                        self.eqn2_matrix[i][i] += self.Gamma(int(kappa),self.all_transitions[i1].i2,i)*(self.fermif(mu[int(kappa)],T)[i1] - 1)
        elif(self.states[i].ne == 0):
            for i1 in range(len(self.all_transitions)):
                for kappa in range(len(self.tlead)):
                    if((self.all_transitions[i1].i2==i)&(self.all_transitions[i1].ne1==self.states[i].ne+1)):
                        self.eqn2_matrix[i][i] += -self.Gamma(int(kappa), i, self.all_transitions[i1].i1)*self.fermif(mu[int(kappa)],T)[i1]
                        self.eqn2_matrix[i][self.all_transitions[i1].i1] += self.Gamma(int(kappa), i, self.all_transitions[i1].i1)*(1-self.fermif(mu[int(kappa)],T)[i1])
                        #print(self.Gamma(int(kappa), i, self.all_transitions[i1].i1), self.fermif(mu[int(kappa)],T)[i1], (self.all_transitions[i1].ne1, self.all_transitions[i1].i1+1), (1 - self.fermif(mu[int(kappa)],T)[i1]), (self.all_transitions[i1].ne2, self.all_transitions[i1].i2+1))
        else:
            for i1 in range(len(self.all_transitions)):
                for kappa in range(len(self.tlead)):
                    if((self.all_transitions[i1].i1==i)&(self.all_transitions[i1].ne2==self.states[i].ne-1)):
                        self.eqn2_matrix[i][self.all_transitions[i1].i2] += self.Gamma(int(kappa),self.all_transitions[i1].i2,i)*self.fermif(mu[int(kappa)],T)[i1]
                        self.eqn2_matrix[i][i] += self.Gamma(int(kappa),self.all_transitions[i1].i2,i)*(self.fermif(mu[int(kappa)],T)[i1] - 1)
                        #print(self.Gamma(int(kappa),self.all_transitions[i1].i2,i), self.fermif(mu[int(kappa)],T)[i1], (self.all_transitions[i1].ne1, self.all_transitions[i1].i1+1), (1 - self.fermif(mu[int(kappa)],T)[i1]), (self.all_transitions[i1].ne2, self.all_transitions[i1].i2+1))
                    if((self.all_transitions[i1].i2==i)&(self.all_transitions[i1].ne1==self.states[i].ne+1)):
                        self.eqn2_matrix[i][i] += -self.Gamma(int(kappa), i, self.all_transitions[i1].i1)*self.fermif(mu[int(kappa)],T)[i1]
                        self.eqn2_matrix[i][self.all_transitions[i1].i1] += self.Gamma(int(kappa), i, self.all_transitions[i1].i1)*(1-self.fermif(mu[int(kappa)],T)[i1])
                        #print(self.Gamma(int(kappa), i, self.all_transitions[i1].i1)*self.fermif(mu[int(kappa)],T)[i1], self.fermif(mu[int(kappa)],T)[i1], (self.all_transitions[i1].ne2, self.all_transitions[i1].i2+1), (1-self.fermif(mu[int(kappa)],T)[i1]), (self.all_transitions[i1].ne1, self.all_transitions[i1].i1+1))

    def fill_eqn2_matrix(self, mu, T):
        self.eqn2_matrix = np.zeros((self.N,self.N))
        for i in range(self.N):
            self.eqn2(i, mu, T)

    def solv_eqn(self, mu, T, plot = False, navplot = False):

        self.fill_eqn2_matrix(mu, T)
        def equation(y, t):
            dydt = []
            for i in range(self.N):
                dydt.append(0)
                for j in range(self.N):
                    dydt[-1]+=(self.eqn2_matrix[i][j]*y[j])
            return(dydt)

        t = 10000
        sol = odeint(equation, self.P0, t)
        sol = np.transpose(sol)

        nav = np.zeros(len(sol[0]))

        for i in range(self.N):
            nav += (np.abs(self.states[i].ne*sol[i]))

        if(plot):
            for i in range(self.N):
                plt.plot(t,sol[:,i])
            plt.plot(t,np.sum(sol,axis=1))
            plt.show()

        if(navplot):
            for i in range(self.N):
                plt.plot(t,self.states[i].ne*sol[i])
            plt.plot(t,nav,'k')
            plt.show()

        return(t,sol,nav)
    
    # Population number vs time plots
    def poptime(self, time, T, mu):
        linestyles = ["solid", "dotted", "dashed", "dashdot"]
        fig = plt.figure(figsize=(12,6))
        for TT in T:
            ss = []
            uu = []
            for q in time:
                ti, sol, nav = self.solv_eqn(mu, TT)
                r=np.transpose(sol)[int(q)][:].tolist()
                ss.append(r+[sum(np.transpose(sol)[int(q)][:])])
                uu.append((q*.13).tolist())  
            for i in range(self.N):
                plt.plot(uu, np.array(ss)[:,i], label=str(Ket("$\\mathcal{N}\, =\,$" + str(self.ecount(i)) + " , " + "$n\, = \,$"+str(i))), linestyle=linestyles[i%4])
        plt.plot(uu, np.array(ss)[:,self.N], label="Total occupation", linestyle=linestyles[i%4])
        plt.legend(fontsize = 18, ncol = 2 * self.l - 1, loc= "upper left", bbox_to_anchor=(1,1))
        plt.grid()
        plt.xlabel("$t\,(ps)$", fontsize=26)
        plt.ylabel("$P_{Nn}$", fontsize=26)
        plt.tick_params(axis='both', which='major', labelsize=24)
        plt.title("Population plot as a function of time $t$, $T = $"+str(TT)+" K"+" , "+"$V_{imp} = $"+str(round(mu/71.5, 3))+" V",fontsize=20)
        plt.show()
        return fig    
    
    # Population number vs chemical potential plots
    def popplot(self, u, T, q):
        linestyles = ["solid", "dotted", "dashed", "dashdot"]
        fig = plt.figure(figsize=(12,6))
        for TT in T:
            ss = []
            uu = []
            for mu in u:
                ti, sol, nav = self.solv_eqn(mu, TT)
                r=np.transpose(sol)[:][q].tolist()
                ss.append(r+[sum(np.transpose(sol)[:][q])])
                uu.append((mu/71.5).tolist())  
            for i in range(self.N):
                plt.plot(uu, np.array(ss)[:,i], label=str(Ket("$\\mathcal{N}\, =\,$" + str(self.ecount(i)) + " , " + "$n\, = \,$"+str(i))), linestyle=linestyles[i%4])
        plt.plot(uu, np.array(ss)[:,self.N], label="Total occupation", linestyle=linestyles[i%4])
        plt.legend(fontsize = 18, ncol = 2 * self.l - 1, loc= "upper left", bbox_to_anchor=(1,1))
        plt.grid()
        plt.xlabel("$V_{imp}\,(V)$", fontsize=26)
        plt.ylabel("$P_{Nn}$", fontsize=26)
        plt.xticks([-.1,0,.1,.2,.3])
        plt.xlim(-.11,.31)
        plt.tick_params(axis='both', which='major', labelsize=24)
        plt.title("Population plot as a function of bias $\mu$, $T = $"+str(TT)+" K"+" , "+"t = "+str(q*0.13)+" ps",fontsize=20)
        plt.show()
        return fig
    
    def popplot2(self, u, T, q):
        linestyles = ["solid", "dotted", "dashed", "dashdot"]
        fig = plt.figure(figsize=(12,6))
        for TT in T:
            ss = []
            uu = []
            for mu in u:
                ti, sol, nav = self.solv_eqn(mu, TT)
                r=np.transpose(sol)[:][q].tolist()
                ss.append(r+[sum(np.transpose(sol)[:][q])])
                uu.append(mu.tolist())  
            for i in range(2**(2)):
                plt.plot(uu, np.array(ss)[:,i], label=self.symbolicstate(i), linestyle=linestyles[i%4])
        plt.plot(uu, np.array(ss)[:,2**(2*self.l)], label="Total occupation", linestyle=linestyles[i%4])
        plt.legend(fontsize = 18, ncol = 2, loc= "upper left", bbox_to_anchor=(1,1))
        plt.grid()
        plt.xlabel("$\mu\,(meV)$", fontsize=26)
        plt.ylabel("$P_{Nn}$", fontsize=26)
        plt.tick_params(axis='both', which='major', labelsize=24)
        plt.title("Population plot as a function of bias $\mu$, $T = $"+str(TT)+" K"+" , "+"t = "+str(q*0.13)+" ps",fontsize=20)
        plt.show()
        return fig
    
    def popplot3(self, u, T, q):
        linestyles = ["solid", "dotted", "dashed", "dashdot"]
        fig = plt.figure(figsize=(12,6))
        for TT in T:
            ss = []
            uu = []
            for mu in u:
                ti, sol, nav = self.solv_eqn(mu, TT)
                r=np.transpose(sol)[:][q].tolist()
                ss.append(r+[sum(np.transpose(sol)[:][q])])
                uu.append(mu.tolist())   
            maxz = heapq.nlargest(10,ss)
            idx = []
            for j in range(10):
                idx.append([i  for i, vl in enumerate(ss) if vl == maxz[j]])
            for i in range(len(idx)):
                plt.plot(uu, np.array(ss)[:, idx[i][0]], label=str(Ket("$\\mathcal{N}\, =\,$" + str(self.ecount(idx[i][0])) + " , " + "$n\, = \,$"+str(idx[i][0]))), linestyle=linestyles[i%4])
        plt.plot(uu, np.array(ss)[:,2**(2*self.l)], label="Total occupation", linestyle=linestyles[i%4])
        plt.legend(fontsize = 18, ncol = 2, loc= "upper left", bbox_to_anchor=(1,1))
        plt.grid()
        plt.xlabel("$\mu\,(meV)$", fontsize=26)
        plt.ylabel("$P_{Nn}$", fontsize=26)
        plt.tick_params(axis='both', which='major', labelsize=24)
        plt.title("Population plot as a function of bias $\mu$, $T = $"+str(TT)+" K"+" , "+"t = "+str(q*0.13)+" ps",fontsize=20)
        plt.show()
        return fig
            
    def usurface(self, u, T):
        for TT in T:
            tt = []
            ss = []
            uu = []
            for mu in u:
                ti, sol, nav = self.solv_eqn(mu, TT)
                ss.append((-nav).tolist())
                uu.append((mu*np.ones(len(nav))).tolist())
                tt.append(ti.tolist())
            cm = plt.pcolormesh(tt,uu,ss, cmap="jet")
            plt.contour(tt,uu,ss,colors="k")
            cb = plt.colorbar(cm)
            cb.ax.set_ylabel("$N$")
            plt.xlabel("$t$")
            plt.ylabel("$\mu$")
            plt.title("$T = $"+str(TT))
            plt.show()

    def uplots(self, u, T):
        for TT in T:
            ss = []
            uu = []
            for mu in u:
                ti, sol, nav = self.solv_eqn(mu, TT)
                ss.append(-nav[-1])
                uu.append(mu)
            plt.plot(uu, ss, label="$T = $"+str(TT))

        plt.xlabel("$\mu$")
        plt.ylabel("$N$")
        plt.legend()
        plt.show()

        
    def charge_stability(self, u1, u2, q):
        linestyles = ["solid", "dotted", "dashed", "dashdot"]
        fig = plt.figure(figsize=(12,6))
        ss = []
        uu = []
        for mu1 in u1:
            for mu2 in u2:
                ti, sol, nav = self.solv_eqn([mu1,mu2], 10000)
            r=np.transpose(sol)[:][q].tolist()
            ss.append(r+[sum(np.transpose(sol)[:][q])])
            uu.append(mu.tolist())   
            maxz = heapq.nlargest(10,ss)
            idx = []
            for j in range(10):
                idx.append([i  for i, vl in enumerate(ss) if vl == maxz[j]])
            for i in range(len(idx)):
                plt.plot(uu, np.array(ss)[:, idx[i][0]], label=str(Ket("$\\mathcal{N}\, =\,$" + str(self.ecount(idx[i][0])) + " , " + "$n\, = \,$"+str(idx[i][0]))), linestyle=linestyles[i%4])
        plt.plot(uu, np.array(ss)[:,2**(2*self.l)], label="Total occupation", linestyle=linestyles[i%4])
        plt.legend(fontsize = 18, ncol = 2, loc= "upper left", bbox_to_anchor=(1,1))
        plt.grid()
        plt.xlabel("$\mu\,(meV)$", fontsize=26)
        plt.ylabel("$P_{Nn}$", fontsize=26)
        plt.tick_params(axis='both', which='major', labelsize=24)
        plt.title("Population plot as a function of bias $\mu$, $T = $"+str(TT)+" K"+" , "+"t = "+str(q*0.13)+" ps",fontsize=20)
        plt.show()
        return fig

    def current(self, lr, mu, T):
        ls = []
        sols = []
        I = 0
        for i in range(4**self.l):
            for j in range(4**self.l):
                sols = self.solv_eqn(mu, T)[1]
                I += np.imag(1j * self.Gamma(lr, i, j) * (self.fermif(i, j, mu, T) * sols[j][-1]) - (1 -  self.fermif(i, j, mu, T)) * sols[i][-1])
        return I
#QS = QuantumSystem(1, 0, [-1., -1.], 3.0, 3.0, 0.0)
#QS.usurface(np.linspace(-5,10,101),[0.2, 3.7, 20])
#QS.uplots(np.linspace(-5,10,101),[0.2,3.7,20])
