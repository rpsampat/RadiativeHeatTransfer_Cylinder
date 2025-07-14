import numpy as np
import math
import scipy
import cantera as ct

def momentum_and_mass_bal(u_vect,p_vect,x_vect,m_jet,num_cells,rho,area,volume,P_in,U_in,rho_in,area_in):
    u = 0
    len_vars= num_cells*2
    A_mat =  np.zeros((len_vars,len_vars))
    B_vect = np.zeros(len_vars)
    for i in range(num_cells):
        if i==0:
            rho_e = (rho[i] + rho[i + 1]) / 2
            rho_w = (rho_in + rho[i]) / 2
            area_e = (area[i] + area[i + 1]) / 2
            area_w = (area_in + area[i]) / 2
            U_e = (u_vect[i] + u_vect[i + 1]) / 2
            U_w = (U_in + u_vect[i]) / 2
        elif i==num_cells-1:
            rho_e = (rho[i] + rho[i]) / 2
            rho_w = (rho[i - 1] + rho[i]) / 2
            area_e = (area[i] + area[i]) / 2
            area_w = (area[i - 1] + area[i]) / 2
            U_e = (u_vect[i] + u_vect[i]) / 2
            U_w = (u_vect[i - 1] + u_vect[i]) / 2
        else:
            rho_e = (rho[i]+rho[i+1])/2
            rho_w = (rho[i-1]+rho[i])/2
            area_e = (area[i]+area[i+1])/2
            area_w = (area[i-1] + area[i]) / 2
            U_e = (u_vect[i]+u_vect[i+1])/2
            U_w = (u_vect[i-1]+u_vect[i])/2

        # Momentum consv
        A_mat[i, i] = ((rho_e * area_e * U_e / 2) - (rho_w * area_w * U_w / 2) + m_jet[i])
        if A_mat[i,i]==0:
            d_P=0
        else:
            d_P = volume[i] / A_mat[i, i]


        if i==0:
            A_mat[i, i] +=rho_w*U_w*area_w/2
            B_vect[i] += U_in*rho_w*U_w*area_w # W
            A_mat[i, i + 1] = rho_e * area_e * U_e / 2 # E
            d_W = d_P
            d_E = volume[i + 1] / A_mat[i, i + 1]
            d_w = (d_W + d_P) / 2
            d_e = (d_E + d_P) / 2
            B_vect[i] += volume[i] * P_in / (2*(x_vect[i + 1] - x_vect[i]))  # P_W
            A_mat[i, num_cells + i + 1] = volume[i] / (2*(x_vect[i + 1] - x_vect[i]))  # P_E
        elif i==num_cells-1:
            A_mat[i, i - 1] = -rho_w * area_w * U_w / 2 # W
            A_mat[i, i] += -rho_e * area_e * U_e / 2 # E U_E=U_P
            d_W = volume[i - 1] / A_mat[i, i - 1]
            d_E = d_P
            d_w = (d_W + d_P) / 2
            d_e = (d_E + d_P) / 2
            A_mat[i, num_cells + i - 1] = -volume[i] / (2*(x_vect[i] - x_vect[i - 1]))  # P_W
            A_mat[i, num_cells + i] += volume[i] / (2 * (x_vect[i] - x_vect[i - 1]))  # P_E
        else:

            A_mat[i,i-1] = -rho_w*area_w*U_w/2 # W
            A_mat[i,i+1]  = rho_e*area_e*U_e/2 # E
            d_W = volume[i - 1] / A_mat[i, i - 1]
            d_E = volume[i + 1] / A_mat[i, i + 1]
            d_w = (d_W+d_P)/2
            d_e = (d_E+d_P)/2
            A_mat[i,num_cells+i-1] = -volume[i]/((x_vect[i+1]-x_vect[i-1])) # P_W
            A_mat[i, num_cells + i + 1] = volume[i] / ((x_vect[i + 1] - x_vect[i - 1])) #P_E

        # Mass consv
        A_mat[num_cells+i,i] = (rho_e*area_e/2)-(rho_w*area_w/2)
        B_vect[num_cells + i] += -m_jet[i]
        if i==0:
            A_mat[num_cells + i, num_cells + i] = (rho_e * area_e * d_e * 5 / (4 * (x_vect[i + 1] - x_vect[i]))) + (
                    rho_w * area_w * d_w * 5 / (4 * (x_vect[i+1] - x_vect[i])))  # P_P
            B_vect[num_cells + i] +=  U_in * (rho_w * area_w / 2)  # W
            A_mat[num_cells + i, i + 1] = (rho_e * area_e / 2)  # E
            A_mat[num_cells + i, num_cells + i + 1] = -(
                        rho_e * area_e * d_e * 5 / (4 * (x_vect[i + 1] - x_vect[i]))) - (rho_w * area_w * d_w * 1 / (
                        4 * (x_vect[i + 1] - x_vect[i])))  # P_E
            B_vect[num_cells + i] += -P_in*((rho_e * area_e * d_e * 1 / (4 * (x_vect[i + 1] - x_vect[i]))) - (
                        rho_w * area_w * d_w * 5 / (4 * (x_vect[i + 1] - x_vect[i]))))  # P_W
            A_mat[num_cells + i, num_cells + i + 2] = (
                        rho_e * area_e * d_e * 1 / (4 * (x_vect[i + 1] - x_vect[i])))  # P_EE
            B_vect[num_cells + i] += P_in*(
                        rho_w * area_w * d_w * 1 / (4 * (x_vect[i + 1] - x_vect[i])))  # P_WW
        elif i==num_cells-1:
            A_mat[num_cells + i, num_cells + i] = (rho_e * area_e * d_e * 5 / (4 * (x_vect[i] - x_vect[i-1]))) + (
                    rho_w * area_w * d_w * 5 / (4 * (x_vect[i] - x_vect[i - 1])))  # P_P
            A_mat[num_cells + i, i - 1] = - (rho_w * area_w / 2)  # W
            B_vect[num_cells + i] += -u_vect[i]*(rho_e * area_e / 2)  # E
            B_vect[num_cells + i] += -p_vect[i]*(-(
                        rho_e * area_e * d_e * 5 / (4 * (x_vect[i] - x_vect[i-1]))) - (rho_w * area_w * d_w * 1 / (
                        4 * (x_vect[i] - x_vect[i - 1]))))  # P_E
            A_mat[num_cells + i, num_cells + i - 1] = (rho_e * area_e * d_e * 1 / (4 * (x_vect[i] - x_vect[i-1]))) - (
                        rho_w * area_w * d_w * 5 / (4 * (x_vect[i] - x_vect[i - 1])))  # P_W
            B_vect[num_cells + i] += -p_vect[i] * (rho_e * area_e * d_e * 1 / (4 * (x_vect[i] - x_vect[i-1])))  # P_EE
            A_mat[num_cells + i, num_cells + i - 2] = - (
                        rho_w * area_w * d_w * 1 / (4 * (x_vect[i] - x_vect[i - 1])))  # P_WW
        else:
            A_mat[num_cells + i, num_cells + i] = (rho_e * area_e * d_e * 5 / (4 * (x_vect[i + 1] - x_vect[i]))) + (
                    rho_w * area_w * d_w * 5 / (4 * (x_vect[i] - x_vect[i - 1])))  # P_P
            A_mat[num_cells + i, i-1] = - (rho_w * area_w / 2) # W
            A_mat[num_cells + i, i+1] = (rho_e * area_e / 2) # E
            A_mat[num_cells + i, num_cells + i + 1] = -(rho_e*area_e*d_e*5/(4*(x_vect[i+1]-x_vect[i]))) - (rho_w*area_w*d_w*1/(4*(x_vect[i]-x_vect[i-1]))) # P_E
            A_mat[num_cells + i, num_cells + i - 1] = (rho_e*area_e*d_e*1/(4*(x_vect[i+1]-x_vect[i]))) - (rho_w*area_w*d_w*5/(4*(x_vect[i]-x_vect[i-1]))) # P_W
            if i==1:
                A_mat[num_cells + i, num_cells + i + 2] = (
                            rho_e * area_e * d_e * 1 / (4 * (x_vect[i + 1] - x_vect[i])))  # P_EE
                B_vect[num_cells + i] += P_in * (
                            rho_w * area_w * d_w * 1 / (4 * (x_vect[i] - x_vect[i - 1])))  # P_WW
            elif i==num_cells-2:
                B_vect[num_cells + i] = -p_vect[i] * (
                            rho_e * area_e * d_e * 1 / (4 * (x_vect[i + 1] - x_vect[i])))  # P_EE
                A_mat[num_cells + i, num_cells + i - 2] = - (
                            rho_w * area_w * d_w * 1 / (4 * (x_vect[i] - x_vect[i - 1])))  # P_WW
            else:
                A_mat[num_cells + i, num_cells + i + 2] = (rho_e * area_e * d_e * 1 / (4 * (x_vect[i + 1] - x_vect[i]))) # P_EE
                A_mat[num_cells + i, num_cells + i - 2] = - (rho_w * area_w * d_e * 1 / (4 * (x_vect[i] - x_vect[i - 1]))) # P_WW

    return A_mat, B_vect


def solver(u_vect,p_vect,x_vect,m_jet,num_cells,rho,area,volume,P_in,U_in,rho_in,area_in):
    A_mat, B_vect = momentum_and_mass_bal(u_vect,p_vect,x_vect,m_jet,num_cells,rho,area,volume,P_in,U_in,rho_in,area_in)
    vect_out = scipy.linalg.solve(A_mat,B_vect)
    u_out = vect_out[0:num_cells]
    p_out = vect_out[num_cells:]

    return u_out, p_out

def solver_0D(u_vect,p_vect,x_vect,m_jet,num_cells,rho,area,volume,P_in,U_in,rho_in,area_in,area_w1,area_w2,area_jet):
    u2=np.zeros(num_cells)
    P2 = np.zeros(num_cells)
    for i in range(num_cells):
        if m_jet[i]>0:
            u_jet = m_jet[i]/(rho[i]*area_jet[i])
        else:
            u_jet= 0
        if i==0:
            u2[i] = np.sqrt(((m_jet[i] * u_jet) + (rho_in * (U_in ** 2) * area_in)) / (rho[i] * area[i]))
            P2[i] = (((((rho_in * area_in) / (rho[i] * area[i])) - 1) * (rho[i] * area[i] * (u_vect[i] ** 2))) + (
                        P_in * area_in) - m_jet[i] * u_vect[i])/area[i]
        else:
            u2[i] = np.sqrt(((m_jet[i]*u_jet)+(rho[i-1]*(u_vect[i-1]**2)*area[i-1]))/(rho[i]*area[i]))
            P2[i] =( ((((rho[i-1]*area[i-1])/(rho[i]*area[i]))-1)*(rho[i]*area[i]*(u_vect[i]**2))) + (p_vect[i-1]*area[i-1]) - m_jet[i]*u_vect[i])/area[i]
        if math.isnan(P2[i]) or math.isnan(u2[i]):
            print(area[i])
    return u2, P2

def solver_energy_upwind(u_vect,p_vect,x_vect,m_jet,num_cells,rho,area,volume,P_in,U_in,rho_in,area_in,T_in, T_vect, mdot, Cp, conv1, conv2,area_jet):
    T2 = np.zeros(num_cells)
    for i in range(num_cells):
        if m_jet[i] > 0:
            u_jet = m_jet[i] / (rho[i] * area_jet[i])
            m_inj = m_jet[i]
        else:
            u_jet=0
            m_inj = 0

        if i==0:
            u_w = U_in#(U_in+u_vect[i])/2
            m_w = rho_in*area_in*(U_in)#rho_in*area_in*(U_in+u_vect[i])/2
            Cp_w = (Cp[i] + Cp[i]) / 2
            T_w = T_in#(T_vect[i] + T_in) / 2


            u_e = u_vect[i]#(u_vect[i + 1] + u_vect[i]) / 2
            m_e = mdot[i]#(mdot[i + 1] + mdot[i]) / 2# - m_jet[i]
            Cp_e = Cp[i]#(Cp[i + 1] + Cp[i]) / 2
            # T_e = (T_vect[i + 1] + T_vect[i]) / 2

            T_e = ((m_w * Cp_w * T_w) + (0.5 * m_w * (u_w ** 2)) + (conv1[i]) + conv2[i]
                  - (0.5 * m_jet[i] * (u_jet ** 2)) - (0.5 * m_e * (u_e ** 2)) + (
                              p_vect[i - 1] * u_vect[i - 1] * area[i - 1])
                  - (p_vect[i] * u_vect[i] * area[i])) / ((m_e * Cp_e) + (m_jet[i] * Cp[i]))
            T2[i] = T_e#2*T_e-T_vect[i+1]
        elif i==num_cells-1:
            u_e = (2*u_vect[i]) / 2
            m_e = mdot[i]#-m_jet[i]
            Cp_e = (Cp[i] + Cp[i]) / 2
            # T_e = (2*T_vect[i]) / 2

            u_w = u_vect[i - 1]#(u_vect[i] + u_vect[i - 1]) / 2
            m_w = mdot[i - 1]#(mdot[i] + mdot[i - 1]) / 2
            Cp_w = Cp[i - 1]#(Cp[i] + Cp[i - 1]) / 2
            T_w = T_vect[i - 1]#(T_vect[i] + T_vect[i - 1]) / 2


            T_e = ((m_w * Cp_w * T_w) + (0.5 * m_w * (u_w ** 2)) + (conv1[i]) + conv2[i]
                   - (0.5 * m_jet[i] * (u_jet ** 2)) - (0.5 * m_e * (u_e ** 2)) + (
                           p_vect[i - 1] * u_vect[i - 1] * area[i - 1])
                   - (p_vect[i] * u_vect[i] * area[i])) / ((m_e * Cp_e) + (m_jet[i] * Cp[i]))
            T2[i] = T_e
        else:
            u_w = u_vect[i - 1]#(u_vect[i] + u_vect[i - 1]) / 2
            u_e = u_vect[i]#(u_vect[i + 1] + u_vect[i]) / 2
            m_w = mdot[i - 1]#(mdot[i] + mdot[i - 1]) / 2
            m_e = mdot[i]#(mdot[i + 1] + mdot[i] ) / 2#- m_jet[i]) / 2
            Cp_w = Cp[i - 1]#(Cp[i] + Cp[i - 1]) / 2
            Cp_e = Cp[i]#(Cp[i + 1] + Cp[i]) / 2
            T_w = T_vect[i - 1]#(T_vect[i] + T_vect[i - 1]) / 2
            # T_e = (T_vect[i + 1] + T_vect[i]) / 2
            T_e = ((m_w*Cp_w*T_w) + (0.5*m_w*(u_w**2)) + (conv1[i]) + conv2[i]
                     - (0.5*m_jet[i]*(u_jet**2)) - (0.5*m_e*(u_e**2))+(p_vect[i-1]*u_vect[i-1]*area[i-1])
                     - (p_vect[i]*u_vect[i]*area[i]))/((m_e*Cp_e) + (m_jet[i]*Cp[i]))
            T2[i] = T_e#2*T_e-T_vect[i+1]
            if math.isnan(T_e) or T_e < 0:
                T_e_try = ((m_w * Cp_w * T_w) + (0.5 * m_w * (u_w ** 2)) + (conv1[i]) + conv2[i]
                       - (0.5 * m_jet[i] * (u_jet ** 2)) - (0.5 * m_e * (u_e ** 2)))
                print(m_e)

    return T2

def solver_energy_upwind_nopressure(u_vect,p_vect,x_vect,m_jet,num_cells,rho,area,volume,P_in,U_in,rho_in,area_in,T_in, T_vect, mdot, Cp, conv1, conv2,area_jet):
    T2 = np.zeros(num_cells)
    for i in range(num_cells):
        if m_jet[i] > 0:
            u_jet = m_jet[i] / (rho[i] * area_jet[i])
            m_inj = m_jet[i]
        else:
            u_jet=0
            m_inj = 0

        if i==0:
            u_w = U_in#(U_in+u_vect[i])/2
            m_w = rho_in*area_in*(U_in)#rho_in*area_in*(U_in+u_vect[i])/2
            Cp_w = (Cp[i] + Cp[i]) / 2
            T_w = T_in#(T_vect[i] + T_in) / 2


            u_e = u_vect[i]#(u_vect[i + 1] + u_vect[i]) / 2
            m_e = mdot[i]#(mdot[i + 1] + mdot[i]) / 2# - m_jet[i]
            Cp_e = Cp[i]#(Cp[i + 1] + Cp[i]) / 2
            # T_e = (T_vect[i + 1] + T_vect[i]) / 2

            T_e = ((m_w * Cp_w * T_w) + (0.5 * m_w * (u_w ** 2)) + (conv1[i]) + conv2[i]
                  - (0.5 * m_jet[i] * (u_jet ** 2)) - (0.5 * m_e * (u_e ** 2))) / ((m_e * Cp_e) + (m_jet[i] * Cp[i]))
            T2[i] = T_e#2*T_e-T_vect[i+1]
        elif i==num_cells-1:
            u_e = (2*u_vect[i]) / 2
            m_e = mdot[i]#-m_jet[i]
            Cp_e = (Cp[i] + Cp[i]) / 2
            # T_e = (2*T_vect[i]) / 2

            u_w = u_vect[i - 1]#(u_vect[i] + u_vect[i - 1]) / 2
            m_w = mdot[i - 1]#(mdot[i] + mdot[i - 1]) / 2
            Cp_w = Cp[i - 1]#(Cp[i] + Cp[i - 1]) / 2
            T_w = T_vect[i - 1]#(T_vect[i] + T_vect[i - 1]) / 2


            T_e = ((m_w * Cp_w * T_w) + (0.5 * m_w * (u_w ** 2)) + (conv1[i]) + conv2[i]
                   - (0.5 * m_jet[i] * (u_jet ** 2)) - (0.5 * m_e * (u_e ** 2))) / ((m_e * Cp_e) + (m_jet[i] * Cp[i]))
            T2[i] = T_e
        else:
            u_w = u_vect[i - 1]#(u_vect[i] + u_vect[i - 1]) / 2
            u_e = u_vect[i]#(u_vect[i + 1] + u_vect[i]) / 2
            m_w = mdot[i - 1]#(mdot[i] + mdot[i - 1]) / 2
            m_e = mdot[i]#(mdot[i + 1] + mdot[i] ) / 2#- m_jet[i]) / 2
            Cp_w = Cp[i - 1]#(Cp[i] + Cp[i - 1]) / 2
            Cp_e = Cp[i]#(Cp[i + 1] + Cp[i]) / 2
            T_w = T_vect[i - 1]#(T_vect[i] + T_vect[i - 1]) / 2
            # T_e = (T_vect[i + 1] + T_vect[i]) / 2
            T_e = ((m_w*Cp_w*T_w) + (0.5*m_w*(u_w**2)) + (conv1[i]) + conv2[i]
                     - (0.5*m_jet[i]*(u_jet**2)) - (0.5*m_e*(u_e**2)))/((m_e*Cp_e) + (m_jet[i]*Cp[i]))
            T2[i] = T_e#2*T_e-T_vect[i+1]
            if math.isnan(T_e) or T_e < 0:
                T_e_try = ((m_w * Cp_w * T_w) + (0.5 * m_w * (u_w ** 2)) + (conv1[i]) + conv2[i]
                       - (0.5 * m_jet[i] * (u_jet ** 2)) - (0.5 * m_e * (u_e ** 2)))
                print(m_e)

    return T2

def solver_energy(u_vect,p_vect,x_vect,m_jet,num_cells,rho,area,volume,P_in,U_in,rho_in,area_in,T_in, T_vect, mdot, Cp, conv1, conv2,area_jet):
    T2 = np.zeros(num_cells)
    for i in range(num_cells):
        if m_jet[i] > 0:
            u_jet = m_jet[i] / (rho[i] * area_jet[i])
            m_inj = m_jet[i]
        else:
            u_jet=0
            m_inj = 0

        if i==0:
            u_w = (U_in+u_vect[i])/2
            m_w = rho_in*area_in*(U_in+u_vect[i])/2
            Cp_w = (Cp[i] + Cp[i]) / 2
            T_w = (T_vect[i] + T_in) / 2


            u_e = (u_vect[i + 1] + u_vect[i]) / 2
            m_e = (mdot[i + 1] + mdot[i]) / 2# - m_jet[i]
            Cp_e = (Cp[i + 1] + Cp[i]) / 2
            # T_e = (T_vect[i + 1] + T_vect[i]) / 2

            T_e = ((m_w * Cp_w * T_w) + (0.5 * m_w * (u_w ** 2)) + (conv1[i]) + conv2[i]
                  - (0.5 * m_jet[i] * (u_jet ** 2)) - (0.5 * m_e * (u_e ** 2)) + (
                              p_vect[i - 1] * u_vect[i - 1] * area[i - 1])
                  - (p_vect[i] * u_vect[i] * area[i])) / ((m_e * Cp_e) + (m_jet[i] * Cp[i]))
            T2[i] = 2*T_e-T_vect[i+1]
        elif i==num_cells-1:
            u_e = (2*u_vect[i]) / 2
            m_e = mdot[i]#-m_jet[i]
            Cp_e = (Cp[i] + Cp[i]) / 2
            # T_e = (2*T_vect[i]) / 2

            u_w = (u_vect[i] + u_vect[i - 1]) / 2
            m_w = (mdot[i] + mdot[i - 1]) / 2
            Cp_w = (Cp[i] + Cp[i - 1]) / 2
            T_w = (T_vect[i] + T_vect[i - 1]) / 2


            T_e = ((m_w * Cp_w * T_w) + (0.5 * m_w * (u_w ** 2)) + (conv1[i]) + conv2[i]
                   - (0.5 * m_jet[i] * (u_jet ** 2)) - (0.5 * m_e * (u_e ** 2)) + (
                           p_vect[i - 1] * u_vect[i - 1] * area[i - 1])
                   - (p_vect[i] * u_vect[i] * area[i])) / ((m_e * Cp_e) + (m_jet[i] * Cp[i]))
            T2[i] = T_e
        else:
            u_w = (u_vect[i] + u_vect[i - 1]) / 2
            u_e = (u_vect[i + 1] + u_vect[i]) / 2
            m_w = (mdot[i] + mdot[i - 1]) / 2
            m_e = (mdot[i + 1] + mdot[i] ) / 2#- m_jet[i]) / 2
            Cp_w = (Cp[i] + Cp[i - 1]) / 2
            Cp_e = (Cp[i + 1] + Cp[i]) / 2
            T_w = (T_vect[i] + T_vect[i - 1]) / 2
            # T_e = (T_vect[i + 1] + T_vect[i]) / 2
            T_e = ((m_w*Cp_w*T_w) + (0.5*m_w*(u_w**2)) + (conv1[i]) + conv2[i]
                     - (0.5*m_jet[i]*(u_jet**2)) - (0.5*m_e*(u_e**2))+(p_vect[i-1]*u_vect[i-1]*area[i-1])
                     - (p_vect[i]*u_vect[i]*area[i]))/((m_e*Cp_e) + (m_jet[i]*Cp[i]))
            T2[i] = 2*T_e-T_vect[i+1]
            if math.isnan(T_e) or T_e < 0:
                T_e_try = ((m_w * Cp_w * T_w) + (0.5 * m_w * (u_w ** 2)) + (conv1[i]) + conv2[i]
                       - (0.5 * m_jet[i] * (u_jet ** 2)) - (0.5 * m_e * (u_e ** 2)))
                print(m_e)
    return T2







