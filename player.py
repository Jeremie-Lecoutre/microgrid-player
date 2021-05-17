# python 3
# this class combines all basic features of a generic player
import numpy as np


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
        self.lIT = np.ones(self.horizon)  # Supposé connu
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


if __name__ == '__main__':
    my_player = Player()
    my_load = my_player.compute_load(0)
