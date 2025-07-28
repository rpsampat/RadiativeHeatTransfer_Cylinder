import cantera as ct
import math
import numpy as np

class CombustorCore:
    """
    Evaluates combustor core temperature and major species using a Plug Flow reactor model.
    The PFR is composed of many PSRs in series
    """
    def __init__(self, num_axial,L, D, m_cooling):
        self.num_axial = num_axial # number of reactors in the axial direction
        self.L = L # combustor length
        self.D= D # combustor diameter
        self.chem_mech = 'H2_11_20.yaml'
        self.hole_axial_loc = np.array([0.2, 0.5, 0.75])  # distance from burner head as a fraction of liner length
        # mass flow as a fraction of cooling flow at the inlet of the annulus near the burner head.
        # Injection flows correspond to location specified in hole_axial_loc
        self.m_inj_frac = np.array([0.1, 0.20, 0.1])  # np.array([0.0, 0.0, 0.0])  #
        self.m_in_annulus = m_cooling

    def m_inj_calc(self):
        self.m_inj = np.zeros(self.num_axial)
        axial_loc = (np.array(range(self.num_axial)) + 1) * self.L / self.num_axial
        for inj_ind in range(len(self.hole_axial_loc)):
            loc = self.hole_axial_loc[inj_ind] * self.L
            ind = np.where(axial_loc >= loc)[0][0]
            self.m_inj[ind] = self.m_inj_frac[inj_ind] * self.m_in_annulus

    def crn(self,T_cooling, P_cooling,T_in,P_in,X_in,mdot_in):
        gas = ct.Solution(self.chem_mech)

        gas.TPX = T_in,P_in,X_in
        Reactor_list=[]
        volume_reactor =(math.pi/4)*(self.D**2)*self.L
        res_in = ct.Reservoir(gas)
        res_out = ct.Reservoir(gas)
        mdot = mdot_in # kg/s
        gas_cooling = ct.Solution(self.chem_mech)
        gas_cooling.TPX = 1100, ct.one_atm,{'O2':0.21, 'N2':0.79}

        for i in range(self.num_axial):
            gas.TP = 1500, P_in
            r1 = ct.IdealGasReactor(gas)
            r1.volume = volume_reactor/self.num_axial
            Reactor_list.append(r1)
            if i==0:
                mfc = ct.MassFlowController(res_in, Reactor_list[i], mdot = mdot)
            else:
                mfc = ct.MassFlowController(Reactor_list[i-1],Reactor_list[i], mdot = mdot)
            # cooling flow
            m_cooling = self.m_inj[i]
            if m_cooling>0:
                gas_cooling.TP = T_cooling[i],P_cooling[i]
                res = ct.Reservoir(gas_cooling)
                mfc_cool = ct.MassFlowController(res,Reactor_list[i],mdot = m_cooling)
                mdot +=m_cooling

        mfc_out = ct.MassFlowController(Reactor_list[i], res_out,mdot = mdot)
        net = ct.ReactorNet(Reactor_list)
        # r2.thermo.T=2000
        # print(r1.thermo)
        # print(r1.thermo.T)
        t=0
        net.advance_to_steady_state()
        T_final =[]
        P_final = []
        X_final = []
        for i in range(self.num_axial):
            T_final.append(Reactor_list[i].thermo.T)
            P_final.append(Reactor_list[i].thermo.P)
            X_final.append(Reactor_list[i].thermo.X)
        # print(T_final)

        return T_final, P_final, X_final

    def main(self,T_cool,P_cool):
        self.m_inj_calc()
        T_in = 1100
        P_in = 10*ct.one_atm
        X_in = {'H2': 0.2, 'O2': 0.21, 'N2': 0.79}
        self.crn(T_cool,P_cool,T_in, P_in, X_in,mdot_in=0.5)

if __name__=="__main__":
    num_axial = 30
    T_cool = 1100*np.ones(num_axial)
    P_cool = 10e5*np.ones(num_axial)
    m_cool = 0.1 #kg/s
    CC = CombustorCore(num_axial,0.3,0.1,m_cool)
    CC.main(T_cool,P_cool)
