# -*- coding: utf-8 -*-
"""
Created on Fri May 28 19:19:33 2021

@author: B57876
"""

# Check feasibility of different player loads
# 2OJ
# ATTENTION CONVENTION ARR/DEP pour la station de charge ; dans mon code les arr/dep 
# sont des pas de tps dans un modèle à tps discret ; dans les données d'entrée c'est en heures...
import numpy as np

wrong_format_score = 1000
pu_infeas_score = 0.1
default_pu_infeas_score = 0.01 # when 0 value is the one to be obtained, 
                              # no relative deviation can be calc.

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
    
    n_infeas_check = 0 # number of constraints checked (to normalize 
                   # the infeas. score at the end)
    
    # identify time-slots with non-zero IT cons.
    nonzero_it_load_ts = np.where(it_load_profile > 0)[0]
    prop_nonzero_it_load = cop_cs/(eer*(cop_hp-1)*delta_t_s)*it_load_profile[nonzero_it_load_ts]
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

def check_charging_station_feasibility(load_profiles: np.ndarray, n_ev_normal_charging: int,
                                       n_ev_fast_charging: int, t_ev_dep: np.ndarray,
                                       t_ev_arr: np.ndarray, ev_max_powers: dict,
                                       ev_batt_capa: np.ndarray, charge_eff: float,
                                       discharge_eff: float, n_ts: int,
                                       delta_t_s: int, dep_soc_penalty: float,
                                       cs_max_power: float) -> (float, float):
    """
    Check EV load profiles obtained from the charging station module
    
    :param load_profiles: matrix with a line per EV charging profile. EVs with 
    normal charging power are provided first, then EV with fast charging techno
    :param n_ev_normal_charging: number of EVs with normal charging power
    :param n_ev_fast_charging: idem with fast charging power
    :param t_ev_dep: time-slots of dep.
    :param t_ev_arr: idem for arr (after dep. here, back from work)
    :param ev_max_powers: dict. with keys the type of EV ("normal" or "fast")
    and values the associated max charging power
    :param ev_batt_capa: EV battery capacity
    :param charge_eff: charging efficiency
    :param discharge_eff: discharging efficiency
    :param n_ts: number of time-slots
    :param delta_t_s: time-slot duration, in seconds
    :param dep_soc_penalty: value of the penalty to be added to the objective if
    EV SoC at departure is below 25% of battery capa
    :param cs_max_power: charging station max. power
    :return: returns the obj. penalty (for not being charged at a minimum SOC
    of 4kWh at dep.) and the infeasibility score
    """
    
    if not (isinstance(load_profiles, np.ndarray) and load_profiles.shape[0] == n_ev_normal_charging + n_ev_fast_charging \
                                and load_profiles.shape[1] == n_ts):
        print("Wrong format for charging station (per EV) load profiles, should be (%i,%i)" \
              % (n_ev_normal_charging + n_ev_fast_charging, n_ts))
        
        return None, wrong_format_score, {}
    
    infeas_list = []
    n_default_infeas = 0
    n_infeas_by_type = {"ev_max_p": 0, "charge_out_of_cs": 0, "soc_max_bound": 0,
                        "soc_min_bound": 0, "min_soc_at_dep": 0, "cs_max_power": 0}
    n_infeas_check = 0 # number of constraints checked (to normalize 
                       # the infeas. score at the end)
    
    # check 
    # 1. that indiv. charging powers respect the indiv. max. power limit
    # normal charging EVs
    for i_ev in range(n_ev_normal_charging):
        infeas_list.extend(list(np.maximum(np.abs(load_profiles[i_ev,:]) \
                                     - ev_max_powers["normal"], 0) / ev_max_powers["normal"]))
        n_infeas_check += n_ts
        # update infeas by type
        n_infeas_by_type["ev_max_p"] += len(np.where(np.array(infeas_list[-n_ts:]) > 0)[0])

    # fast charging EVs
    for i_ev in range(n_ev_fast_charging):
        infeas_list.extend(list(np.maximum(np.abs(load_profiles[i_ev,:]) \
                                     - ev_max_powers["fast"], 0) / ev_max_powers["fast"]))
        n_infeas_check += n_ts
        # update infeas by type
        n_infeas_by_type["ev_max_p"] += len(np.where(np.array(infeas_list[-n_ts:]) > 0)[0])
    # 0 charging power when EV is not connected (convention that EV leave at the end
    # of time-slot t_dep and arrive at the beginning of t_arr -> can charge in both ts)
    for i_ev in range(n_ev_normal_charging + n_ev_fast_charging):
        n_charge_out_of_cs = \
            len(np.where(np.abs(load_profiles[i_ev,t_ev_dep[i_ev]+1:t_ev_arr[i_ev]-1])>0)[0])
        n_default_infeas += n_charge_out_of_cs
        # update infeas by type
        n_infeas_by_type["charge_out_of_cs"] += n_charge_out_of_cs
        
    # 2. that SoC bounds of each EV is respected, as well as min. charging need at dep.
    cs_dep_soc_penalty = 0
    for i_ev in range(n_ev_normal_charging + n_ev_fast_charging):
        current_batt_soc = (charge_eff*np.cumsum(np.maximum(load_profiles[i_ev,:],0)) \
                               - discharge_eff*np.cumsum(np.maximum(-load_profiles[i_ev,:],0))) \
                                * delta_t_s / 3600
        # diminish SoC when arriving at CS with E quantity consumed when driving
        current_batt_soc[t_ev_arr[i_ev]] -= 4
        
        # max bound (EV batt. capa)
        infeas_list.extend(list(np.maximum(current_batt_soc
                                     - ev_batt_capa[i_ev], 0) / ev_batt_capa[i_ev]))
        n_infeas_check += n_ts
        n_infeas_by_type["soc_max_bound"] += len(np.where(np.array(infeas_list[-n_ts:]) > 0)[0])
        
        # min bound (0)
        n_soc_below_zero = len(np.where(current_batt_soc < 0)[0])
        n_default_infeas += n_soc_below_zero 
        n_infeas_by_type["soc_min_bound"] += n_soc_below_zero
        
        # SoC at dep. is above the minimal level requested
        if current_batt_soc[t_ev_dep[i_ev]] < 0.25 * ev_batt_capa[i_ev]:
            cs_dep_soc_penalty += dep_soc_penalty
            n_infeas_by_type["min_soc_at_dep"] += 1
        n_infeas_check += 1
    
    # 3.that CS power is below the max allowed value
    infeas_list.extend(list(np.maximum(np.abs(np.sum(load_profiles, axis=0)) \
                                           - cs_max_power, 0) / cs_max_power))
    n_infeas_check += n_ts
    n_infeas_by_type["cs_max_power"] += len(np.where(np.array(infeas_list[-n_ts:]) > 0)[0])

    # calculate infeasibility score    
    infeas_score = calculate_infeas_score(n_infeas_check, infeas_list, n_default_infeas)                                

    return cs_dep_soc_penalty, infeas_score, n_infeas_by_type

def check_solar_farm_feasibility(load_profile: np.ndarray, batt_capa: float,
                                 batt_max_power: float, charge_eff: float,
                                 discharge_eff: float, n_ts: int, delta_t_s: int) -> float:
    """
    Check battery load profile obtained from the Solar Farm module. Note: idem
    Industrial Site feas. check in the current version of the modelling
    
    :param load_profile: vector with battery load
    :param batt_capa: battery capacity
    :param batt_max_power: batt max (charge and discharge) power
    :param charge_eff: charging efficiency
    :param discharge_eff: discharging efficiency
    :param n_ts: number of time-slots
    :param delta_t_s: time-slot duration, in seconds
    :return: returns the infeasibility score
    """

    if not (isinstance(load_profile, np.ndarray) and load_profile.shape == (n_ts,)):
        print("Wrong format for Solar Farm load profile, should be (%i,)" % n_ts)
        
        return wrong_format_score, {}

    return check_industrial_cons_feasibility(load_profile, batt_capa, batt_max_power,
                                             charge_eff, discharge_eff, n_ts,
                                             delta_t_s)

def check_industrial_cons_feasibility(load_profile: np.ndarray, batt_capa: float,
                                      batt_max_power: float, charge_eff: float,
                                      discharge_eff: float, n_ts: int,
                                      delta_t_s: int) -> float:
    """
    Check battery load profile obtained from the Industrial Cons. module
    
    :param load_profile: vector with battery load
    :param batt_capa: battery capacity
    :param batt_max_power: batt max (charge and discharge) power
    :param charge_eff: charging efficiency
    :param discharge_eff: discharging efficiency
    :param n_ts: number of time-slots
    :param delta_t_s: time-slot duration, in seconds
    :return: returns the infeasibility score
    """

    if not (isinstance(load_profile, np.ndarray) and load_profile.shape == (n_ts,)):
        print("Wrong format for Industrial Site load profile, should be (%i,)" % n_ts)
        
        return wrong_format_score, {}
    
    infeas_list = []
    n_default_infeas = 0
    n_infeas_by_type = {"batt_max_p": 0, "soc_max_bound": 0, "soc_min_bound": 0}
    n_infeas_check = 0 # number of constraints checked (to normalize 
                       # the infeas. score at the end)
    
    # check 
    # 1. that battery charging powers respect the max. power limit
    infeas_list.extend(list(np.maximum(np.abs(load_profile) - batt_max_power, 0) / batt_max_power))
    n_infeas_check += n_ts
    # update infeas by type
    n_infeas_by_type["batt_max_p"] += len(np.where(np.array(infeas_list[-n_ts:]) > 0)[0])
        
    # 2. that batt. SoC bounds are respected
    batt_soc = (charge_eff*np.cumsum(np.maximum(load_profile,0)) \
                        - discharge_eff*np.cumsum(np.maximum(-load_profile,0))) \
                                    * delta_t_s / 3600
    # max bound (batt. capa)
    infeas_list.extend(list(np.maximum(batt_soc - batt_capa, 0) / batt_capa))
    n_infeas_check += n_ts
    n_infeas_by_type["soc_max_bound"] += len(np.where(np.array(infeas_list[-n_ts:]) > 0)[0])        
    # min bound (0)
    n_soc_below_zero = len(np.where(batt_soc < 0)[0])
    n_default_infeas += n_soc_below_zero 
    n_infeas_by_type["soc_min_bound"] += n_soc_below_zero
        
    # calculate infeasibility score                                    
    infeas_score = calculate_infeas_score(n_infeas_check, infeas_list, n_default_infeas)
    
    return infeas_score, n_infeas_by_type                         

if __name__ == "__main__":
    # general temporal parameters
    n_ts = 48
    delta_t_s = 1800

    import pandas as pd    
    import copy
    scenario_data = pd.read_csv("ev_scenarios.csv", sep=";", decimal=".")
    
    # TEST: charging station feasibility test -> from the code of one of your classmates
    from code_eleves.code_franchino_opti_class_5 import Player    

    p = Player()
    p.set_scenario(scenario_data)
    
    prices_test = [1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3]
    p.set_prices(prices_test)
    
    resultat = p.optimisation()
    l = resultat[0]
    
    # from arr/dep in hours to time-slots
    t_ev_dep = np.array([int(3600/delta_t_s*elt) for elt in p.depart])
    t_ev_arr = np.array([int(3600/delta_t_s*elt) for elt in p.arr])
    
    cs_dep_soc_penalty, cs_infeas_score, n_infeas_by_type = \
        check_charging_station_feasibility(np.array(l), 2, 2, t_ev_dep, t_ev_arr,
                                           {"normal": 3, "fast": 22}, 40*np.ones(4),
                                           0.95, 0.95, 48, delta_t_s, 5, 40)
    print("res code Franchino: ", cs_dep_soc_penalty, cs_infeas_score, n_infeas_by_type)
    
    # TEST: Data Center feas. check -> from randomly generated data
    cop_cs = 4 + 1
    cop_hp = 60 / (60 - 35) * 0.5
    eer = 4
    dc_load_profile = np.random.rand(n_ts)
    it_load_profile = np.random.rand(n_ts)
    dc_load_profile_good = np.random.rand(n_ts) \
                        * cop_cs/(eer*(cop_hp-1)*delta_t_s) * it_load_profile 
    dc_load_profile_bad = copy.deepcopy(dc_load_profile_good)
    dc_load_profile_bad[24] = cop_cs/(eer*(cop_hp-1)*delta_t_s) * it_load_profile[24] + 1
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
    
    # TEST: Industrial Site feas. check -> from randomly generated data
    batt_capa = 60
    batt_max_power = 10
    is_load_profile_good = np.zeros(n_ts)
    for t in range(n_ts):
        is_load_profile_good[t] = \
            min(batt_max_power,
                (batt_capa - np.sum(is_load_profile_good[:t])*delta_t_s/3600)/2)
    is_infeas_score_good, n_infeas_by_type = \
        check_industrial_cons_feasibility(is_load_profile_good, batt_capa,
                                          batt_max_power, 0.95, 0.95, n_ts, delta_t_s)
    print("IS infeas score good prof.: ", is_infeas_score_good)
    
    # TEST: Solar Farm identical to Industrial Site... not done here but function
    # available above
    
    
    



