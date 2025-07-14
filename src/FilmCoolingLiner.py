import math
import numpy as np
import matplotlib.pyplot as plt

def Nusselt_local_lam_developing_Tw_const(Re,Pr,dia,x):
    Nu_x_1 = 3.66
    beta= Re*Pr*dia/x
    Nu_x_2 = 1.077*(beta)**(1/3)
    Nu_x_3 = 0.5*((2/(1+(22*Pr)))**(1/6))*((beta)**(1/2))
    Nu_x = ((Nu_x_1**3) + (0.7**3) + ((Nu_x_2-0.7)**3) + ((Nu_x_3)**3))**(1/3)

    return Nu_x

def Nusselt_local_turb_fullydeveloped(Re, Pr, dia, x):
    Xi = ((1.8*np.log10(Re))-1.5)**-2
    Nu_x = ((Xi*Re*Pr/8)/(1+12.7*((Xi/8)**(1/2))*((Pr**(2/3))-1)))*(1+((1/3)*((dia/x)**(2/3))))

    return Nu_x

def convection_developing_pipeflow(mdot_comb,mu,Pr, dia,x):
    Re = 4 * mdot_comb / (math.pi * mu * dia)
    gamma = (Re - 2300) / ((1e4) - 2400)
    Nu_x = ((1 - gamma) * Nusselt_local_lam_developing_Tw_const(2300 * np.ones(len(x)), Pr, dia, x)) + (
                gamma * Nusselt_local_turb_fullydeveloped((1e4) * np.ones(len(x)), Pr, dia, x))

    return Nu_x

def convection_annulus(mdot_comb,mu,Pr, dia,x):
    # Steps: 1) h0: heat transfer coefficient without cooling assuming pipe flow
    # 2) film cooling effectiveness (eta) based on data from literature
    # 3) h: correction of h0 with eta
    # Nusselt number calculation based on bulk flow in chamber

    # print("Gamma=",gamma)
    Nu_x=np.zeros(len(mdot_comb))
    for i in range(len(mdot_comb)):

        Re = 4 * (mdot_comb[i]) / (math.pi * mu[i] * dia)

        # print("Re=",Re)
        gamma = (Re - 2300) / ((1e4) - 2300)
        # if x[i]/dia[i]<5:
        #     Nu = 0.023*(Re[i]**0.8)*(Pr[i]**0.385)*((x[i]/dia[i])**-0.0054)
        if gamma[i]<0:
            Nu = Nusselt_local_lam_developing_Tw_const(Re[i],Pr[i],dia[i],x[i])
        elif gamma[i]>1:
            Nu = Nusselt_local_turb_fullydeveloped(Re[i], Pr[i], dia[i], x[i])
        else:
            Nu = ((1-gamma[i])*Nusselt_local_lam_developing_Tw_const(2300,Pr[i],dia[i],x[i])) + (gamma[i]*Nusselt_local_turb_fullydeveloped((1e4),Pr[i],dia[i],x[i]))


        Nu_x[i] = (Nu)

    return Nu_x


def convection_liner(mdot_comb,dia, x,T,P,X,gas):
    # Steps: 1) h0: heat transfer coefficient without cooling assuming pipe flow
    # 2) film cooling effectiveness (eta) based on data from literature
    # 3) h: correction of h0 with eta
    # Nusselt number calculation based on bulk flow in chamber

    h_x = np.zeros(len(T))
    for i in range(len(T)):
        gas.TPX = T[i],P[i],X[i]
        Re = 4 * mdot_comb[i] / (math.pi * gas.viscosity * dia)
        Pr = gas.viscosity*gas.cp_mass/gas.thermal_conductivity
        # print(Pr)
        # print("Re=", Re)
        gamma = (Re - 2300) / ((1e4) - 2300)
        # print("Gamma=", gamma)
        if gamma < 0:
            Nu = Nusselt_local_lam_developing_Tw_const(Re, Pr, dia, x[i])
        elif gamma > 1:
            Nu = Nusselt_local_turb_fullydeveloped(Re, Pr, dia, x[i])
        else:
            Nu = ((1 - gamma) * Nusselt_local_lam_developing_Tw_const(2300, Pr, dia, x[i])) + (
                        gamma * Nusselt_local_turb_fullydeveloped((1e4), Pr, dia, x[i]))

        h_x[i] = (Nu*gas.thermal_conductivity/dia)

    return h_x

def main():
    Re = 6500
    Pr = 0.7
    dia = 0.1
    mdot_comb = 0.1 #kg/s
    mu = 3.26*1e-5
    x = np.linspace(0.001,0.3,50)
    Nu_x = convection_developing_pipeflow(mdot_comb,mu,Pr,dia,x)
    fig, ax = plt.subplots()
    plt.plot(x, Nu_x)
    plt.show()

if __name__=="__main__":
    main()