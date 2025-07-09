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
    Re = 4*mdot_comb/(math.pi*mu*dia)
    # print("Re=",Re)
    gamma = (Re-2300)/((1e4)-2300)
    # print("Gamma=",gamma)
    Nu_x=np.zeros(len(gamma))
    for i in range(len(gamma)):
        if gamma[i]<0:
            Nu = Nusselt_local_lam_developing_Tw_const(Re[i],Pr,dia[i],x[i])
        elif gamma[i]>1:
            Nu = Nusselt_local_turb_fullydeveloped(Re[i], Pr, dia[i], x[i])
        else:
            Nu = ((1-gamma[i])*Nusselt_local_lam_developing_Tw_const(2300,Pr,dia[i],x[i])) + (gamma[i]*Nusselt_local_turb_fullydeveloped((1e4),Pr,dia[i],x[i]))

        Nu_x[i] = (Nu)

    return Nu_x


def convection_liner(mdot_comb, mu, Pr, dia, x):
    # Steps: 1) h0: heat transfer coefficient without cooling assuming pipe flow
    # 2) film cooling effectiveness (eta) based on data from literature
    # 3) h: correction of h0 with eta
    # Nusselt number calculation based on bulk flow in chamber
    Re = 4 * mdot_comb / (math.pi * mu * dia)
    # print("Re=", Re)
    gamma = (Re - 2300) / ((1e4) - 2300)
    # print("Gamma=", gamma)
    Nu_x = np.zeros(len(gamma))
    for i in range(len(gamma)):
        if gamma[i] < 0:
            Nu = Nusselt_local_lam_developing_Tw_const(Re[i], Pr, dia, x[i])
        elif gamma[i] > 1:
            Nu = Nusselt_local_turb_fullydeveloped(Re[i], Pr, dia, x[i])
        else:
            Nu = ((1 - gamma[i]) * Nusselt_local_lam_developing_Tw_const(2300, Pr, dia, x[i])) + (
                        gamma[i] * Nusselt_local_turb_fullydeveloped((1e4), Pr, dia, x[i]))

        Nu_x[i] = (Nu)

    return Nu_x

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