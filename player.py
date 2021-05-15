# python 3
# this class combines all basic features of a generic player
import numpy as np


class Player:

	def __init__(self):
		# some player might not have parameters
		self.parameters = 0

		#Données en entrée
		self.prices = {"purchase": np.ones(self.horizon), "sale":np.ones(self.horizon)}

		#Constantes du problème
		self.dt=0.5
		self.horizon = 48
		self.EER=4 #Energy efficiency ratio
		self.COPcs=self.EER+1 #Coefficient of performance (cooling system)
		self.Tr=308.15 #Entry temperature
		self.Tcom=333.15 #Comfort temperature
		self.e=0.4 #Thermal efficiency
		self.COPhp=self.Tcom/(self.Tcom-self.Tr)*self.e #Coefficient of performance (heat production)
		self.MaxProd=10 #kWh

		#Puissances thermiques
		self.hDC=np.zeros(self.horizon)
		self.hr=np.zeros(self.horizon)
		self.hIT = np.zeros(self.horizon)

		#Charges
		self.lIT = np.zeros(self.horizon)
		self.lNF=np.zeros(self.horizon)
		self.lHP = np.zeros(self.horizon)

		#Variable à optimiser
		self.alpha = np.zeros(self.horizon)

	def set_scenario(self, scenario_data):
		self.data = scenario_data

	def set_prices(self, prices):
		self.prices = prices

	def compute_all_load(self):
		load = np.zeros(self.horizon)
		# for time in range(self.horizon):
		# 	load[time] = self.compute_load(time)
		return load

	def take_decision(self, time):
		# TO BE COMPLETED
		return 0

	def compute_load(self, time):
		load = self.take_decision(time)
		# do stuff ?
		return load

	def reset(self):
		# reset all observed data
		pass