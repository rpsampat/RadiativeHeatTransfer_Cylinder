import matplotlib.pyplot as plt
import numpy as np
import scipy
import math
import cantera as ct
import mesh

class FluidHeatTransfer:
    def __init__(self):
        self.L=1
        self.m_in_annulus = 0.05# kg/s
        self.T_in = 1100 # K
        self.P_in = 10e5 # Pa
        self.cooling_hole_specs()
        self.gas = ct.Solution('air.yaml')

    def convection_internal(self):
        eta = 0.23 # effectiveness of film cooling mechanism in reducing heat transfer; lower is more cooling

    def cooling_hole_specs(self):
        self.hole_axial_loc = np.array([0.2,0.5,0.75]) # distance from burner head as a fraction of liner length
        # mass flow as a fraction of cooling flow at the inlet of the annulus near the burner head.
        # Injection flows correspond to location specified in hole_axial_loc
        self.m_inj_frac = np.array([0.1,0.20,0.1])#np.array([0.0,0.0,0.0])#
        # if np.sum(self.m_inj_frac)>1:
        #     self.m_inj_frac = self.m_inj_frac/np.sum(self.m_inj_frac)

        return 0

    def mass_bal_calc(self, mesh_wall1, mesh_wall2):
        # Pre-calculation of quantities along axial length of annulus
        gas = ct.Solution('air.yaml')
        self.num_axial = mesh_wall2.num_axial
        self.L = mesh_wall2.L
        self.m_inj = np.zeros(self.num_axial)
        axial_loc = (np.array(range(self.num_axial))+1)*self.L/self.num_axial
        for inj_ind in range(len(self.hole_axial_loc)):
            loc = self.hole_axial_loc[inj_ind]*self.L
            ind = np.where(axial_loc>=loc)[0][0]
            self.m_inj[ind] = self.m_inj_frac[inj_ind] * self.m_in_annulus
        mass_bal_mat = np.zeros((self.num_axial, self.num_axial))
        mass_bound = np.zeros(self.num_axial)
        area_channel = np.zeros(self.num_axial)
        perimeter_hydraulic = np.zeros(self.num_axial)
        self.dx = mesh_wall2.d_z[[(i) * mesh_wall2.num_thickness for i in range(self.num_axial)]]
        self.a_wall1 = mesh_wall1.d_z[[(i) * mesh_wall1.num_thickness for i in range(self.num_axial)]]*mesh_wall1.D*math.pi
        self.a_wall2 = mesh_wall2.d_z[[(i) * mesh_wall2.num_thickness for i in range(self.num_axial)]]*mesh_wall2.D*math.pi
        for i in range(self.num_axial):
            r1 = mesh_wall2.y[((i + 1) * mesh_wall2.num_thickness) - 1]
            r2 = mesh_wall1.y[((i) * mesh_wall1.num_thickness)]
            area_channel[i] = (math.pi) * ((r2 ** 2) - (r1 ** 2))
            perimeter_hydraulic[i] = 2 * math.pi * (r2 + r1)
            if i == 0:
                mass_bound[i] = self.m_in_annulus
                mass_bal_mat[i, i] = 1
            else:
                mass_bal_mat[i, i] = 1
                mass_bal_mat[i, i - 1] = -1

        m_out = scipy.linalg.solve(mass_bal_mat, mass_bound - self.m_inj)

        self.m_annulus = m_out
        self.area_channel = area_channel
        self.perimeter_channel = perimeter_hydraulic

        return 0

    def convection_annulus(self, T_annulus, P_annulus, rho_annulus):

        """
        Assumption of axisymmetry along axis of combustor
        :param mesh_wall1: outer wall (casing)
        :param mesh_wall2: inner wall (liner)
        :return:
        """
        # Pre-calculation of heat transfer coefficient along axial length of annulus

        mu = np.zeros(self.num_axial)
        self.Cp = np.zeros(self.num_axial)
        for i in range(self.num_axial):
            if T_annulus[i]<0:
                self.gas.TP = self.T_in,self.P_in#P_annulus[i]
            else:
                self.gas.TP = T_annulus[i], P_annulus[i]
            mu[i] = self.gas.viscosity
            self.Cp[i]= self.gas.cp_mass

        Re = 4*self.m_annulus/(mu*self.perimeter_channel)
        Pr = 0.7
        dia_h = 4*self.area_channel/self.perimeter_channel
        l = self.L
        Nu = ((2 / (1 + 22 * Pr)) ** (1 / 6)) * ((Re * Pr * dia_h / l) ** (1 / 2))
        k = 4e-2
        self.h1 = Nu * k / dia_h # outer wall
        self.h2 = self.h1 # inner wall
        self.u = self.m_annulus/(rho_annulus*self.area_channel)

    def fluid_annulus_matrix_setup(self,T_w1,T_w2,rho_annulus):
        energy_bal_mat = np.zeros((self.num_axial, self.num_axial))
        energy_bound = np.zeros(self.num_axial)
        enthalpy_flow = self.m_annulus*self.Cp*0.5
        enthalpy_inj = self.m_inj*self.Cp*0.5
        conv1 = self.h1*self.a_wall1 # outer wall
        conv2 = self.h2*self.a_wall2 # inner wall
        k=0.026
        for i in range(self.num_axial):
            if i == 0:
                energy_bound[i] = self.m_in_annulus*self.Cp[i]*self.T_in-k*self.area_channel[i]*self.T_in/self.dx[i]-0.5*self.m_annulus[i]*(((self.u[i+1]+self.u[i])/2)**2)
                energy_bal_mat[i, i] = conv1[i] + conv2[i] + enthalpy_flow[i] + enthalpy_inj[i]-k*self.area_channel[i]/self.dx[i]
                energy_bal_mat[i, i + 1] = enthalpy_flow[i]+k*self.area_channel[i+1]/self.dx[i+1]
            elif i==self.num_axial-1:
                energy_bound[i] =  0.5 * self.m_annulus[i-1] * (((self.u[i - 1] + self.u[i]) / 2)**2) - 0.5 * self.m_annulus[i] * ((self.u[i])**2)
                energy_bal_mat[i, i - 1] = -enthalpy_flow[i - 1]+k*self.area_channel[i-1]/self.dx[i-1]
                energy_bal_mat[i, i] = -enthalpy_flow[i-1] + conv1[i] + conv2[i] + 2*enthalpy_flow[i] + enthalpy_inj[i]-k*self.area_channel[i]/self.dx[i]
            else:
                energy_bound[i] = (0.5 * self.m_annulus[i - 1] * (((self.u[i - 1] + self.u[i]) / 2 )**2)
                                   - 0.5 * self.m_annulus[i] * (((self.u[i]+self.u[i+1])/2)**2))
                energy_bal_mat[i, i-1] = -enthalpy_flow[i-1]+k*self.area_channel[i-1]/self.dx[i-1]
                energy_bal_mat[i, i] = -enthalpy_flow[i-1]+conv1[i]+conv2[i]+enthalpy_flow[i]+enthalpy_inj[i]-2*k*self.area_channel[i]/self.dx[i]
                energy_bal_mat[i,i+1] = enthalpy_flow[i]+k*self.area_channel[i+1]/self.dx[i+1]

        enthalpy_wall = conv1*T_w1 + conv2*T_w2 - enthalpy_inj*T_w2
        T_annulus = scipy.linalg.solve(energy_bal_mat,energy_bound+enthalpy_wall)

        return T_annulus

    def solver(self,mesh_wall1, mesh_wall2,T_annulus, P_annulus, rho_annulus,T_w1, T_w2):

        self.mass_bal_calc(mesh_wall1, mesh_wall2)
        alpha = 0.9
        # ax.plot(self.m_annulus)
        # plt.show()
        gamma = 1.4
        for i in range(1000):
            self.convection_annulus(T_annulus, P_annulus, rho_annulus)
            T_annulus1 = self.fluid_annulus_matrix_setup(T_w1, T_w2, rho_annulus)
            T_annulus = T_annulus * alpha + T_annulus1 * (1 - alpha)
            P_annulus0 = np.array(P_annulus)
            for j in range(len(P_annulus0)):
                if j == 0:
                    P_annulus[j] = (self.P_in * ((self.T_in / T_annulus[j]) ** (gamma / (1 - gamma))))
                else:
                    P_annulus[j] = (P_annulus0[j - 1] * (
                                (T_annulus[j - 1] / T_annulus[j]) ** (gamma / (1 - gamma)))) * (1 - alpha) + P_annulus0[
                                       j] * alpha
            # for j in range(len(P_annulus0)):
            #     if j==0:
            #         P_annulus[j] = self.P_in
            #     else:
            #         P_annulus[j] = (P_annulus0[j - 1] + rho_annulus[j-1]*(self.u[j-1]**2) - rho_annulus[j]*(self.u[j]**2))*(1-alpha) + P_annulus0[j]*alpha
            rho_annulus = P_annulus * (28.8 / 8314) / T_annulus
        q_conv_w1 = self.h1 * (T_annulus - T_w1)
        q_conv_w2 = self.h2 * (T_annulus - T_w2)

        return q_conv_w1, q_conv_w2, np.array(T_annulus), np.array(P_annulus), np.array(rho_annulus)

    def main(self):
        # Wall2: liner
        mesh_wall2 = mesh.Surf_Cyl_Wall_Mesh()
        mesh_wall2.D = 0.100
        mesh_wall2.initialize()
        # Wall1: casing
        mesh_wall1 = mesh.Surf_Cyl_Wall_Mesh()
        mesh_wall1.D = 0.110
        mesh_wall1.initialize()
        self.num_axial = mesh_wall2.num_axial
        T_annulus = self.T_in*np.ones(self.num_axial)
        P_annulus = self.P_in * np.ones(self.num_axial)
        rho_annulus = P_annulus*(28.8/8314)/T_annulus
        T_w1 = 1100* np.ones(self.num_axial)
        T_w2 = 1500* np.ones(self.num_axial)

        q_conv_w1, q_conv_w2, T_annulus, P_annulus, rho_annulus = self.solver(mesh_wall1, mesh_wall2,T_annulus, P_annulus, rho_annulus,T_w1, T_w2)

        fig,ax = plt.subplots()
        ax.plot(T_annulus)
        fig1, ax1 = plt.subplots()
        ax1.plot(P_annulus)
        plt.show()




if __name__=="__main__":
    obj = FluidHeatTransfer()
    obj.main()
