import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

# Define the mesh spacing (in nm)
delta_x = 1e-9

# Define the rest mass of an electron
m0 = 9.11e-31

# Define the value of Pi
PI = np.pi

# Define the value of h
H = 6.626e-34

# Define the value of eps0
eps0 = 8.85e-12

# Define the value of h_bar
H_BAR = H/(2 * PI)

# Define the value of K * T at 300 K
K_T = 1.38e-23 * 300

# Define the value of the charge on an electron
Q = 1.6e-19

# (also account for delta_Ec (dfference in Ec levels when Ef for two materials is kept constant))
# Ef=0, Ec = -qV, Ev = -qV - (Eg)

# Define utility functions for device independent calculations
def calculate_n(Ec, Nc):

    # Declare a result variable
    n = np.zeros(Ec.shape[0])

    # Integrate
    for i in range(0, n.shape[0]):
        eqn = lambda E : (np.sqrt(E) / (1 + np.exp(E + (Ec[i] / K_T))))
        n[i], err = integrate.quad(eqn, 0, np.inf)
        n[i] = n[i] * Nc[i] * 2 / np.sqrt(np.pi)

    return n

def calculate_p(Ev, Nv):
    
    # Initialise the result
    p = np.zeros(Ev.shape[0])
    
    # Integrate 
    for i in range(0, p.shape[0]):
        eqn = lambda E : (np.sqrt(E) / (1 + np.exp(E - (Ev[i] / K_T))))
        p[i], err = integrate.quad(eqn, 0, np.inf)
        p[i] = p[i] * Nv[i] * 2 / np.sqrt(np.pi)
        
    return p


# Define a class which stores the properties of the material
class material:
    def __init__(self, eps, Eg, Nc, Nv, chi):
        self.eps = eps  # epsilon
        self.Eg = Eg  # band gap
        self.chi = chi  # electron affinity

        # Calculate the values of Nc and Nv
        self.Nc = Nc
        self.Nv = Nv

        # Caculate ni and Ei (with respect to Evac)
        self.ni = np.sqrt(self.Nc * self.Nv * np.exp( (-1.0) * self.Eg / K_T))


# Define a class which stores the information related to a slab of material in the structure
class layer:
    def __init__(self, mat, Na, Nd, t):
        self.mat = mat
        self.Na = Na
        self.Nd = Nd
        self.t = t


# Define a class for the entire device
class structure:
    def __init__(self, L, layers):
        self.L = L  # no of materials in layer
        self.layers = layers    # array of layers

        thick = 0
        for i in range(0, L):
            thick = thick + layers[i].t

        self.N = int(thick) * delta_x # length in nm

        self.M = int(self.N/delta_x)   # no. of mesh points

        self.struct_eps = np.zeros(self.M)
        self.struct_Na = np.zeros(self.M)
        self.struct_Nd = np.zeros(self.M)
        self.struct_ni = np.zeros(self.M)
        self.struct_Eg = np.zeros(self.M)
        self.struct_delta_Ec = np.zeros(self.M)
        self.struct_Nc = np.zeros(self.M)
        self.struct_Nv = np.zeros(self.M)

        reference_chi = layers[0].mat.chi

        # Generate the material structure
        index = 0
        for i in range(0, L):
            self.struct_eps[index: index + layers[i].t] = layers[i].mat.eps
            self.struct_ni[index: index + layers[i].t] = layers[i].mat.ni
            self.struct_Eg[index: index + layers[i].t] = layers[i].mat.Eg
            self.struct_Nc[index: index + layers[i].t] = layers[i].mat.Nc
            self.struct_Nv[index: index + layers[i].t] = layers[i].mat.Nv
            self.struct_Na[index: index + layers[i].t] = layers[i].Na
            self.struct_Nd[index: index + layers[i].t] = layers[i].Nd
            self.struct_delta_Ec[index: index +
                                 layers[i].t] = layers[i].mat.chi - reference_chi
            index = index + layers[i].t

            # Add an array to store the value of delta_Ec for each structure with respect to the first
            # Ei = -qV + delta_Ec

    # Function to initialise the electrostatic potential vector
    def init_pot(self, device, Na, Nd, Nc, Eg, left_bc, right_bc):

        # Initialise the result
        V = np.zeros(device.M)
        
        # Initialise the potential
        for i in range(0, device.M):
            if (Nc[i] != 0):
               # Calculate qV from doping: V = kT/q * ln(n/Nc)
               V[i] = (K_T/Q * np.log(Nd[i]/Nc[i]))
            # If the material has zero Nc, Nv   
            else:
                # Find the first occurence of dEc = 0
                index = np.where(self.struct_delta_Ec != 0)[0][0]
                # Set Nc = value of this material
                Nc_temp = Nc[index]
                Nd_temp = Nd[index]
                
                # Initialise the potential
                V[i] = (K_T/Q * np.log(Nd_temp/Nc_temp))
        
        # Add the boundary conditions
        V[0] = V[0] + left_bc
        V[device.M - 1] = V[device.M - 1] + right_bc

        return V

    # Function to solve the Poisson Equation and Carrier Concentration
    def poisson_solve(self, N, left_bc, right_bc):

        # Set the number of mesh points
        N = self.M

        # Set the band gaps of the materials used
        Egs = self.struct_Eg

        # Set the doping profile
        Nd = self.struct_Nd
        Na = self.struct_Na
        
        # Set the effective density of states
        Nc = self.struct_Nc
        Nv = self.struct_Nv

        # Initialise electron and hole concentrations
        n = Nd
        p = Na

        # Initialise the values of delta_Ec
        dEc = self.struct_delta_Ec
        
        # Initialise the dielectric constants
        struct = self.struct_eps

        # Initialise the potential
        V = self.init_pot(self, Na, Nd, Nc, Egs, left_bc, right_bc)
        V_old = V

        # Initialise parameters for the NR loop
        tol = 1e-5
        max_iter = 1
        count = 0

        # Start the Newton Raphson Method
        for j in range(0, max_iter):

            print("Currently on iteration ", j)

            # Set the value of Ec and Ev with respect to Ei
            Ec = -1 * Q * V + dEc
            Ev = Ec - (Egs)

            # Calculate n,p
            n = Nc * np.exp((-1) * (Ec) / K_T)
            
            # Initialiasation of p for the first iteration
            if (j==0):
                p = Na
            else:
                p = Nv * np.exp((Ev) / K_T)

            # Initialise rho
            rho = Q * (Nd + p - n - Na)

            # Initialise the Jacobian
            jac = np.zeros((N, N))
            
            # Define the derivative of rho with respect to the potential
            drho_dv = ((Q * Q)/(K_T)) * (n + p)
            
            # Define the finite difference formulation
            dv_dx2 = np.zeros(N)
            for i in range(1, N - 1):
                dv_dx2[i] = ((struct[i + 1] + struct[i]) * V[i + 1] - (struct[i + 1] + 2 * struct[i] + struct[i - 1]) * V[i] + (struct[i] + struct[i-1]) * V[i-1]) / (2 * delta_x * delta_x)

            dv_dx2[0] = ((struct[0] + struct[1]) * V[1] - (struct[0] + struct[1]) * V[0]) / (2 * delta_x * delta_x)
            dv_dx2[N - 1] = ((struct[N-2] + struct[N-1]) * V[N-2] - (struct[N-2] + struct[N-1]) * V[N-1]) / (2 * delta_x * delta_x)

            # Generate the Jacobian
            for i in range(1, N - 1):
                jac[i][i - 1] = (struct[i] + struct[i-1]) / (2 * delta_x * delta_x)
                jac[i][i] = -1 * (struct[i + 1] + 2 * struct[i] + struct[i - 1]) / (2 * delta_x * delta_x) - drho_dv[i]
                jac[i][i + 1] = (struct[i + 1] + struct[i]) / (2 * delta_x * delta_x)

            jac[0][0] = 1
            jac[N - 1][N-1] = 1
        
            # Make the RHS of the Poisson Equation
            R = rho + dv_dx2
            R[0] = 0
            R[N-1] = 0

            # Solve for the Electrostatic Potential
            delta_v = np.linalg.solve(jac, R)
            # delta_v[0] = 0
            # delta_v[N-1] = 0
            
            # Update the value of V
            V = V_old - delta_v

            # Stopping condition
            if (np.max(np.abs(delta_v)) < tol):
                break

            # Prepare for the next iteration
            V_old = V
            count = count + 1

        # Print the number of iterations requires
        print("Exited Loop at Iteration", str(count))

        return [V, n, p]


# Create material objects to use
Si = material(eps=11.9 * eps0,
              Eg=1.12 * Q,
              Nc = 2.4e25,
              Nv = 1.04e25,
              chi=4.05 * Q)

GaAs = material(eps=12.9 * eps0,
                Eg=1.42 * Q,
                Nc=4.7e23,
                Nv=7.0e24,
                chi=4.07 * Q)

HfO2 = material(eps=25 * eps0,
                Eg=5.9 * Q,
                Nc=0,
                Nv=0,
                chi=2.14 * Q)

def main():
    
    # Create the layer structure for a Silicon p/n junction diode
    l0 = layer(Si, 2e22, 1e9, 300)
    l1 = layer(Si, 1e9, 1e22, 350)

    # Stack the layers
    layers = [l0, l1]
    
    # Create the structure and enter the number of mesh points
    device = structure(len(layers), layers)

    # Call the calculation function for a given bias to the device
    [V, n, p] = device.poisson_solve(device.M, 0, 0)
    
    # For plotting the band diagram
    p_Ec = ((-1) * Q * V) - device.struct_delta_Ec
    p_Ev = p_Ec - device.struct_Eg 

    # Plot the final value of the electrostatic potential over the entire structure
    
    # plt.semilogy(p)
    # plt.semilogy(n)
    # plt.plot(V)
    plt.plot(p_Ec, label="Ec")
    plt.plot(p_Ev, label="Ev")
    plt.title("Band Diagram")
    plt.xlabel("X - position (in nm)")
    plt.ylabel("Energy (in Joules)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
