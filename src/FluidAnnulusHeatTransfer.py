import matplotlib.pyplot as plt
import numpy as np
import scipy
import math
import cantera as ct
from scipy.stats import alpha

import mesh
import convection_1d_cfd
import FilmCoolingLiner

class FluidHeatTransfer:
    def __init__(self):
        self.L=1
        self.num_axial = 100
        self.m_in_annulus = 0.1# kg/s
        self.T_in = 1100 # K
        self.P_in = 10e5 # Pa
        # Cooling hole specs
        self.hole_axial_loc = np.array([0.2, 0.5, 0.75])  # distance from burner head as a fraction of liner length
        # mass flow as a fraction of cooling flow at the inlet of the annulus near the burner head.
        # Injection flows correspond to location specified in hole_axial_loc
        self.m_inj_frac = np.array([0.1,0.20,0.1])#np.array([0.0, 0.0, 0.0])  #
        # if np.sum(self.m_inj_frac)>1:
        #     self.m_inj_frac = self.m_inj_frac/np.sum(self.m_inj_frac)
        self.gas = ct.Solution('air.yaml')

    # def convection_internal(self):
    #     eta = 0.23 # effectiveness of film cooling mechanism in reducing heat transfer; lower is more cooling

    def mass_bal_calc(self, mesh_wall1, mesh_wall2):
        # Pre-calculation of quantities along axial length of annulus
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
        self.rho_in = self.P_in * (28.8 / 8314) / self.T_in
        self.U_in = self.m_in_annulus / (self.rho_in * self.area_channel[0])


        return 0



    def precalc_annulus(self, T_annulus, P_annulus, rho_annulus,x):
        """
        Assumption of axisymmetry along axis of combustor
        :param T_annulus:
        :param P_annulus:
        :param rho_annulus:
        :return:
        """
        # Pre-calculation of heat transfer coefficient along axial length of annulus

        mu = np.zeros(self.num_axial)
        self.Cp = np.zeros(self.num_axial)
        for i in range(self.num_axial):
            if T_annulus[i]<0 or P_annulus[i]<0 or rho_annulus[i]<0:
                self.gas.TP = self.T_in,self.P_in
                if P_annulus[i]<0:
                    P_annulus[i] =1e5#self.P_in
                if T_annulus[i]<0:
                    T_annulus[i] = self.T_in
                if rho_annulus[i]<0:
                    rho_annulus[i] = self.rho_in
            else:
                self.gas.TP = T_annulus[i], P_annulus[i]
            mu[i] = self.gas.viscosity
            self.Cp[i]= self.gas.cp_mass

        # Re = self.perimeter_channel*self.m_annulus/(mu*self.area_channel*math.pi)
        Pr = 0.7
        dia_h = 4*self.area_channel/self.perimeter_channel
        # l = self.L
        # Nu = ((2 / (1 + 22 * Pr)) ** (1 / 6)) * ((Re * Pr * dia_h / l) ** (1 / 2))
        k = 4e-2
        x = x+x[1]
        Nu_x = FilmCoolingLiner.convection_annulus(self.m_annulus, mu, Pr, dia_h, x)
        self.h1 = Nu_x * k / dia_h # outer wall
        self.h2 = self.h1 # inner wall


        # alpha = 0.6
        # gamma = 1.4
        # for i in range(10):
        #
        #     P_annulus0 = np.array(P_annulus)
        #     for j in range(len(P_annulus0)):
        #         if j == 0:
        #             P_annulus[j] = (self.P_in)# * ((self.T_in / T_annulus[j]) ** (gamma / (1 - gamma))))
        #         else:
        #             P_annulus[j] = (P_annulus0[j - 1] + rho_annulus[j - 1] * (self.u[j - 1] ** 2) - rho_annulus[j] * (
        #                         self.u[j] ** 2)- rho_annulus[j] * (self.u[j] * (self.m_inj[j] / (rho_annulus[j] * (self.perimeter_channel[j] * self.dx[j]))))) * (
        #                                alpha) + P_annulus0[j]
        #     rho_annulus = P_annulus * (28.8 / 8314) / T_annulus
        #     fig, ax = plt.subplots()
        #     ax.plot(self.u)
        #     plt.show()



    def fluid_annulus_matrix_setup(self,T_w1,T_w2,pr):
        energy_bal_mat = np.zeros((self.num_axial, self.num_axial))
        energy_bound = np.zeros(self.num_axial)
        pressure_bound = np.zeros(self.num_axial)
        enthalpy_flow = self.m_annulus*self.Cp*0.5
        enthalpy_inj = self.m_inj*self.Cp*0.5
        conv1 = self.h1*self.a_wall1 # outer wall
        conv2 = self.h2*self.a_wall2 # inner wall
        k=0.04
        for i in range(self.num_axial):

            if i == 0:
                enthalpy_e = (enthalpy_flow[i] + enthalpy_flow[i + 1] / 2)
                energy_bound[i] = (((enthalpy_flow[i]*self.T_in) + conv1[i]*T_w1[i] + conv2[i]*T_w2[i]
                                   + k*self.area_channel[i]*self.T_in/(self.dx[i]) - enthalpy_inj[i]*T_w2[i])
                                   + 0.5*self.m_annulus[i]*(((self.u[i]+self.u[i])/2)**2) - 0.5*self.m_annulus[i]*(((self.u[i+1]+self.u[i])/2)**2))
                energy_bal_mat[i, i] = conv1[i] + conv2[i] + enthalpy_e - enthalpy_flow[i] + enthalpy_inj[i] + 2* k*self.area_channel[i]/(self.dx[i])
                energy_bal_mat[i, i + 1] = enthalpy_e - k * self.area_channel[i+1]/(self.dx[i+1])
                pressure_bound[i] += -(
                            (((self.P_in + pr[i]) / 2) * ((self.U_in + self.u[i]) / 2) * self.area_channel[i]) - (
                                ((self.P_in + pr[i]) / 2) * ((self.P_in + self.u[i]) / 2) * self.area_channel[i]))
            elif i==self.num_axial-1:
                enthalpy_w = (enthalpy_flow[i] + enthalpy_flow[i - 1] / 2)
                energy_bound[i] =   (conv1[i]*T_w1[i] + conv2[i]*T_w2[i] - enthalpy_inj[i]*T_w2[i]
                                     + 0.5 * self.m_annulus[i-1] * (((self.u[i - 1] + self.u[i]) / 2)**2) - 0.5 * self.m_annulus[i] * ((self.u[i])**2))
                energy_bal_mat[i, i - 1] = -enthalpy_flow[i] - enthalpy_w + (1/2) * k*self.area_channel[i]/(self.dx[i]) - k*self.area_channel[i-1]/(self.dx[i-1])
                energy_bal_mat[i, i] = conv1[i] + conv2[i] + (3/2)*enthalpy_flow[i] -enthalpy_w + enthalpy_inj[i]+ (-1/2) * k*self.area_channel[i]/(self.dx[i]) + k*self.area_channel[i]/(self.dx[i])
                pressure_bound[i] += -(
                            (((pr[i] + pr[i]) / 2) * ((self.u[i] + self.u[i]) / 2) * self.area_channel[i]) - (
                                ((pr[i] + pr[i]) / 2) * ((self.u[i] + self.u[i]) / 2) * self.area_channel[i]))
            else:
                enthalpy_e = (enthalpy_flow[i] + enthalpy_flow[i + 1] / 2)
                enthalpy_w = (enthalpy_flow[i] + enthalpy_flow[i - 1] / 2)
                energy_bound[i] = conv1[i]*T_w1[i] + conv2[i]*T_w2[i] + (0.5 * self.m_annulus[i - 1] * (((self.u[i - 1] + self.u[i]) / 2 )**2)
                                   - 0.5 * self.m_annulus[i] * (((self.u[i]+self.u[i+1])/2)**2)) - enthalpy_inj[i]*T_w2[i]
                energy_bal_mat[i, i-1] = -enthalpy_w - k*self.area_channel[i-1]/(self.dx[i-1])
                energy_bal_mat[i, i] =  conv1[i] + conv2[i] + enthalpy_e - enthalpy_w + enthalpy_inj[i] + 2*k*self.area_channel[i]/(self.dx[i])
                energy_bal_mat[i,i+1] = -enthalpy_e - k*self.area_channel[i+1]/(self.dx[i+1])
                pressure_bound[i] += -((((pr[i+1]+pr[i])/2)*((self.u[i+1]+self.u[i])/2)*self.area_channel[i])-(((pr[i-1]+pr[i])/2)*((self.u[i-1]+self.u[i])/2)*self.area_channel[i]))

        enthalpy_wall = conv1*T_w1 + conv2*T_w2 - enthalpy_inj*T_w2
        T_annulus = scipy.linalg.solve(energy_bal_mat,energy_bound+enthalpy_wall+pressure_bound)

        return T_annulus

    def newton_system(self, T0,T_w1,T_w2,P0,gamma):
        alpha = 0.5
        rho_annulus = P0 * (28.8 / 8314) / T0
        P_annulus0 = np.array(P0)
        self.convection_annulus(T0, P_annulus0, rho_annulus)
        T_annulus = self.fluid_annulus_matrix_setup(T_w1, T_w2)

        for j in range(len(P0)):
            if j==0:
                P0[j] = (self.P_in * ((self.T_in / T_annulus[j]) ** (gamma / (1 - gamma))))
            else:
                P0[j] = (P_annulus0[j - 1] * ((T_annulus[j - 1] / T_annulus[j]) ** (gamma / (1 - gamma))))*alpha + P_annulus0[j]
                # P0[j] = ((P_annulus0[j - 1]*self.area_channel[j-1] + rho_annulus[j-1]*(self.u[j-1]**2) - rho_annulus[j]*(self.u[j]**2)
                #                 - rho_annulus[j]*(self.u[j]*(self.m_inj[j]/(rho_annulus[j]*(self.perimeter_channel[j]*self.dx[j])))))/self.area_channel[j])*(1-alpha) + P_annulus0[j]*alpha


        epsi = (T_annulus-T0)/T0

        return epsi

    def solver1(self, mesh_wall1, mesh_wall2, T_annulus, P_annulus, rho_annulus, T_w1, T_w2):
        gamma = 1.4
        self.mass_bal_calc(mesh_wall1, mesh_wall2)
        T_annulus = scipy.optimize.newton(self.newton_system, T_annulus, args=(T_w1,T_w2,P_annulus,gamma))
        q_conv_w1 = self.h1 * (T_annulus - T_w1)
        q_conv_w2 = self.h2 * (T_annulus - T_w2)

        print('h1=', self.h1)

        return q_conv_w1, q_conv_w2, np.array(T_annulus), np.array(P_annulus), np.array(rho_annulus)



    def solver(self,mesh_wall1, mesh_wall2,T_annulus, P_annulus, rho_annulus,T_w1, T_w2):

        self.mass_bal_calc(mesh_wall1, mesh_wall2)
        self.u = self.m_annulus / (rho_annulus * self.area_channel)
        alpha = 0.1
        # ax.plot(self.m_annulus)
        # plt.show()
        gamma = 1.4
        x_vect = mesh_wall1.z[[(i) * mesh_wall1.num_thickness for i in range(self.num_axial)]]
        dx_vect = mesh_wall1.d_z[[(i) * mesh_wall1.num_thickness for i in range(self.num_axial)]]
        volume_cells_channel = self.area_channel*dx_vect
        epsi_list = []

        for i in range(10000):

            # print('Uin=', self.U_in)
            rho_annulus = np.array(P_annulus * (28.8 / 8314) / T_annulus)

            # u_vect, p_vect = convection_1d_cfd.solver(self.u,P_annulus,x_vect,self.m_inj,self.num_axial,
            #                                           rho_annulus,self.area_channel,volume_cells_channel,self.P_in,
            #                                           self.U_in,self.rho_in,self.area_channel[0])
            u_vect, p_vect = convection_1d_cfd.solver_0D(self.u, P_annulus, x_vect, self.m_inj, self.num_axial,
                                                      rho_annulus, self.area_channel, volume_cells_channel, self.P_in,
                                                      self.U_in, self.rho_in, self.area_channel[0], self.a_wall1,self.a_wall2)
            epsi_p = np.max(np.abs((P_annulus - p_vect) / (P_annulus)))
            epsi_u = np.max(np.abs((u_vect-self.u)/self.u))
            epsi_list.append(epsi_p)
            self.u = np.array(self.u + (alpha)*(np.array(u_vect)-self.u))
            P_annulus = np.array((P_annulus) + (alpha*(np.array(p_vect)-P_annulus)))
            P_out = P_annulus

            # rho_annulus = np.array(P_annulus * (28.8 / 8314) / T_annulus)
            self.precalc_annulus(T_annulus, P_annulus, rho_annulus,x_vect)




            # Energy Balance
            conv1 = self.h1 * self.a_wall1 * (T_w1-T_annulus) # outer wall
            conv2 = self.h2 * self.a_wall2 * (T_w2-T_annulus) # inner wall
            T_annulus1 = convection_1d_cfd.solver_energy(self.u, P_annulus, x_vect, self.m_inj, self.num_axial,
                                                      rho_annulus, self.area_channel, volume_cells_channel, self.P_in,
                                                      self.U_in, self.rho_in, self.area_channel[0],self.T_in, T_annulus,
                                                         self.m_annulus, self.Cp, conv1, conv2)

            T0 = np.array(T_annulus)
            epsi_T = np.max(np.abs((T_annulus1-T0)/T0))
            T_annulus = np.array(T_annulus + (T_annulus1-T_annulus) * (alpha))
            # P_annulus0 = np.array(P_annulus)
            # for j in range(len(P_annulus0)):
            #     # if T_annulus[j]<0:
            #     #     T_annulus[j] = 500
            #     if j == 0:
            #         P_annulus[j] = self.P_in * ((self.T_in / T_annulus[j]) ** (gamma / (1 - gamma)))
            #         T_annulus[j] = self.T_in
            #     else:
            #         P_annulus[j] = (P_annulus0[j - 1] * (
            #                     (T_annulus[j - 1] / T_annulus[j]) ** (gamma / (1 - gamma)))) * (1 - alpha) + P_annulus0[
            #                            j] * alpha
            # for j in range(len(P_annulus0)):
            #     if j==0:
            #         P_annulus[j] = (self.P_in * ((self.T_in / T_annulus[j]) ** (gamma / (1 - gamma))))
            #     else:
            #         P_annulus[j] = (P_annulus0[j - 1] + rho_annulus[j-1]*(self.u[j-1]**2) - rho_annulus[j]*(self.u[j]**2)
            #                         - rho_annulus[j]*(self.u[j]*(self.m_inj[j]/(rho_annulus[j]*(self.perimeter_channel[j]*self.dx[j])))))*(alpha) + P_annulus0[j]
            if epsi_p<1e-6 and epsi_u<1e-6 and epsi_T<1e-6 and i>5:
                break

        # fig, ax = plt.subplots()
        # ax.plot(self.u)
        # ax.set_ylabel('Velocity (m/s)')

        # fig1, ax1 = plt.subplots()
        # ax1.plot(P_annulus)
        # ax1.set_ylabel('Pressure (Pa)')

        # fig, ax = plt.subplots()
        # ax.plot(T_annulus)
        # ax.set_ylabel('Temperature (K)')
        #
        # fig, ax = plt.subplots()
        # ax.plot(rho_annulus)
        # ax.set_ylabel('Density (kg/m3)')
        #
        # fig, ax = plt.subplots()
        # ax.plot(epsi_list)
        # ax.set_ylabel('Residual (Pressure)')
        # plt.show()

        q_conv_w1 = self.h1 * (T_annulus - T_w1)
        q_conv_w2 = self.h2 * (T_annulus - T_w2)

        # print('h1=',self.h1)
        print('Epsi_p=',epsi_p)
        print('Epsi_u=',epsi_u)

        return self.h1, self.h2, np.array(T_annulus), np.array(P_out), np.array(rho_annulus)

    def main(self):
        num_len = 30
        L = 0.3
        # Wall2: liner
        mesh_wall2 = mesh.Surf_Cyl_Wall_Mesh()
        mesh_wall2.D = 0.100
        mesh_wall2.L = L
        mesh_wall2.num_axial = num_len
        mesh_wall2.initialize()
        # Wall1: casing
        mesh_wall1 = mesh.Surf_Cyl_Wall_Mesh()
        mesh_wall1.D = 0.110
        mesh_wall1.L = L
        mesh_wall1.num_axial = num_len
        mesh_wall1.initialize()
        self.num_axial = mesh_wall2.num_axial
        T_annulus = self.T_in*np.ones(self.num_axial)
        P_annulus = self.P_in * np.ones(self.num_axial)
        rho_annulus = P_annulus*(28.8/8314)/T_annulus
        T_w1 = 1100* np.ones(self.num_axial)
        T_w2 = 1500* np.ones(self.num_axial)

        q_conv_w1, q_conv_w2, T_annulus, P_annulus, rho_annulus = self.solver(mesh_wall1, mesh_wall2,T_annulus, P_annulus, rho_annulus,T_w1, T_w2)

        fig, ax = plt.subplots()
        ax.plot(self.m_annulus)
        fig, ax = plt.subplots()
        ax.plot(self.u)
        fig,ax = plt.subplots()
        ax.plot(T_annulus)
        fig1, ax1 = plt.subplots()
        ax1.plot(P_annulus/1e5)
        plt.show()




if __name__=="__main__":
    obj = FluidHeatTransfer()
    obj.main()
