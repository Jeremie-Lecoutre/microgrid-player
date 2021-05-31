# python 3
# this class combines all basic features of a generic player
import numpy as np
import pandas as pd
import os


class Player:

    def __init__(self):
        # some player might not have parameters
        self.parameters = 0

        # Constantes du problème
        self.dt = 0.5
        self.horizon = 48
        self.EER = 4  # Energy efficiency ratio
        self.COPcs = self.EER + 1  # Coefficient of performance (cooling system)
        self.Tr = 308.15  # Entry temperature
        self.Tcom = 333.15  # Comfort temperature
        self.e = 0.4  # Thermal efficiency
        self.COPhp = self.Tcom / (self.Tcom - self.Tr) * self.e  # Coefficient of performance (heat production)
        self.MaxProd = 10  # kWh

        # Données en entrée
        self.prices = {"purchase": np.ones(self.horizon), "sale": np.ones(self.horizon)}
        self.prices = 20 + 12 * np.random.rand(2 * self.horizon)

        # Puissances thermiques
        self.hDC = np.ones(self.horizon)
        self.hr = np.ones(self.horizon)
        self.hIT = np.ones(self.horizon)

        # Charges
        self.lIT = 20 + 12 * np.random.rand(self.horizon)  # Supposé connu
        """l_i = pd.read_csv("C:\\Ponts\\COV\\Optimisation et énergie\\data_center_scenarios.csv", sep=";")"""
        l_i = pd.read_csv(os.path.join(os.getcwd(),"data_center_scenarios.csv"), sep=";")
        self.lIT = l_i[l_i["scenario"] == 1]["cons (kW)"]

        self.lNF = np.zeros(self.horizon)
        self.lHP = np.zeros(self.horizon)

        # Variable à optimiser
        self.alpha = np.zeros(self.horizon)

    def set_scenario(self, scenario_data):
        self.data = scenario_data

    def set_prices(self, prices):
        self.prices = prices

    def compute_all_load(self):
        load = np.zeros(self.horizon)
        for time in range(self.horizon):
            load[time] = self.compute_load(time)
        return load

    def take_decision(self, time):
        # Update des charges et de la chaleur rejetée indépendantes de alpha
        self.hIT[time] = self.lIT[time]
        self.hr[time] = self.COPcs / self.EER * self.lIT[time]
        self.lNF[time] = (1 + 1 / (self.EER * self.dt)) * self.lIT[time]

        # Décision quant à l'activation du système d'exploitation de la chaleur rejetée
        if self.prices[time] * self.hr[time] < self.MaxProd * self.prices[time + self.horizon]:
            self.hDC[time] = self.MaxProd  # Vaut 0 si alpha vaut 0
            self.lHP[time] = self.hDC[time] / (self.COPhp * self.dt)  # Vaut 0 si alpha vaut 0
            self.alpha[time] = self.lHP[time] * (self.COPhp - 1) * self.dt / self.hr[time]  # Alpha vaut 0 par défaut
        return self.alpha[time]

    def compute_load(self, time):
        load = 0
        alpha = self.take_decision(time)
        load += alpha * self.hr[time] / ((self.COPhp - 1) * self.dt)  # Heat production load
        # load+=self.lIT[time] #IT load
        load += self.lNF[time]  # Non-flexible electricity load

        return load

    def reset(self):
        # reset all observed data
        pass



## ***************** Feasability ********************

wrong_format_score = 1000
pu_infeas_score = 0.1
default_pu_infeas_score = 0.01


def calculate_infeas_score(n_infeas_check: int, infeas_list: list,
                           n_default_infeas: int) -> float:
    """
    Calculate infeasibility score

    :param n_infeas_check: number of infeasibility check (number of constraints
    to be respected)
    :param infeas_list: list of infeasibility values, relatively to the NONZERO
    values to be respected
    :param n_default_infeas: number of infeas. corresponding to ZERO values to
    be respected
    """

    return np.sum(infeas_list) / n_infeas_check * pu_infeas_score \
           + n_default_infeas * default_pu_infeas_score


def check_data_center_feasibility(load_profile: np.ndarray, cop_cs: float,
                                  cop_hp: float, eer: float, n_ts: int,
                                  delta_t_s: int, it_load_profile: np.ndarray) -> float:
    """
    Check heat pump load profile obtained from the DC module

    :param load_profile: vector (1, n_ts) with heat pump load profile
    :param cop_cs: Coeff. Of Perf. of the Cooling System
    :param cop_hp: Coeff. Of Perf. of the Heat Pump
    :param eer: Energy Eff. Ratio of the IT room
    :param n_ts: number of time-slots
    :param delta_t_s: time-slot duration, in seconds
    :return: returns the infeasibility score
    """

    if not (isinstance(load_profile, np.ndarray) and load_profile.shape == (n_ts,)):
        print("Wrong format for Data Center load profile, should be (%i,)" % n_ts)

        return wrong_format_score

    # check that DC load is non-negative and smaller than IT load up to a proportional coeff.

    n_infeas_check = 0  # number of constraints checked (to normalize
    # the infeas. score at the end)

    # identify time-slots with non-zero IT cons.
    nonzero_it_load_ts = np.where(it_load_profile > 0)[0]
    prop_nonzero_it_load = cop_cs / (eer * (cop_hp - 1) * delta_t_s) * it_load_profile[nonzero_it_load_ts]
    infeas_list = list(np.maximum(load_profile[nonzero_it_load_ts] \
                                  - prop_nonzero_it_load, 0) / prop_nonzero_it_load)
    n_infeas_check += len(prop_nonzero_it_load)

    n_default_infeas = 0
    # loop over ts with zero IT load
    for t in range(n_ts):
        if not t in nonzero_it_load_ts and load_profile[t] > 0:
            n_default_infeas += 1

    # Check that HP load prof. be non-negative
    n_default_infeas += len(np.where(load_profile < 0)[0])

    # calculate and return infeasibility score
    return calculate_infeas_score(n_infeas_check, infeas_list, n_default_infeas)



if __name__ == '__main__':
    my_player = Player()
    my_load = my_player.compute_load(0)
    print(my_load)

## **************** Feasability *****************
# general temporal parameters
    n_ts = 48
    delta_t_s = 1800
    import copy
    ##scenario_data = pd.read_csv("ev_scenarios.csv", sep=";", decimal=".")

    # TEST: Data Center feas. check -> from randomly generated data
    cop_cs = 4 + 1
    cop_hp = 60 / (60 - 35) * 0.5
    eer = 4
    dc_load_profile = np.random.rand(n_ts)
    it_load_profile = np.random.rand(n_ts)
    dc_load_profile_good = np.random.rand(n_ts) \
                           * cop_cs / (eer * (cop_hp - 1) * delta_t_s) * it_load_profile
    dc_load_profile_bad = copy.deepcopy(dc_load_profile_good)
    dc_load_profile_bad[24] = cop_cs / (eer * (cop_hp - 1) * delta_t_s) * it_load_profile[24] + 1
    dc_infeas_score = check_data_center_feasibility(dc_load_profile, cop_cs,
                                                    cop_hp, eer, n_ts, delta_t_s,
                                                    it_load_profile)
    dc_infeas_score_good = check_data_center_feasibility(dc_load_profile_good, cop_cs,
                                                         cop_hp, eer, n_ts, delta_t_s,
                                                         it_load_profile)
    dc_infeas_score_bad = check_data_center_feasibility(dc_load_profile_bad, cop_cs,
                                                        cop_hp, eer, n_ts, delta_t_s,
                                                        it_load_profile)
    print("DC infeas score random prof.: ", dc_infeas_score)
    print("DC infeas score good prof.: ", dc_infeas_score_good)
    print("DC infeas score bad prof.: ", dc_infeas_score_bad)