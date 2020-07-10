# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 21:56:40 2020

@author: Brownie
"""

import numpy as np
import pandas as pd
from sw_GeneticAlgorithm import GeneticAlgorithm

'''
	x: list or np.array of input
'''

def ackley(x): 
	x = np.array(x) 
	n = x.shape[0]
	return 20 + np.exp(1) - 20 * np.exp( -0.2 * np.sqrt( np.sum(x**2) / n) ) - np.exp( np.sum(np.cos(2 * np.pi * x)) / n )

def levy(x):
	x = np.array(x) 
	y = 1 + (x-1)/4 
	return np.sin(np.pi*y[0])**2 + np.sum((y[1:-1] - 1)**2* (1 + 10*np.sin( np.pi*y[1:-1] +1)**2 )) + (y[-1]-1)**2*(1+np.sin(2*np.pi*x[-1])**2)

def quadric(x):
	x = np.array(x) 
	n = x.shape[0]
	return np.sum([ np.sum(x[:i])**2 for i in range(n)])			
		
def rastrigin(x):
	x = np.array(x)
	n = x.shape[0]
	return 10 * n + np.sum((x**2)/4000) - np.prod([np.cos( x[i-1]/np.sqrt(i) ) for i in range(1, n+1)]) + 1

def rosenbrock(x):
	x = np.array(x)
	n = x.shape[0]
	return np.sum([100*(x[i-1]**2 - x[i])**2 + (x[i-1] - 1)**2 for i in range(1, n)])

def schwefel(x):
	x = np.array(x)
	abx = np.abs(x)
	return np.sum(abx) + np.prod(abx)

def sphere(x):
	x = np.array(x)
	return np.sum(x**2)

def sumsquares(x):
	x = np.array(x)
	n = x.shape[0]
	return np.sum([i*x[i-1]**2 for i in range(1, n+1)])

functions = {
		'Ackley': {
						'fitness_function': ackley, 
						'domains': (-10, 10), 
						'optimizer': min
						},
		# 'Levy': {
		# 				'fitness_function': levy, 
		# 				'domains': (-10, 10),	
		# 				'optimizer': min
		# 				},
		# 'Quadric': {
		# 				'fitness_function': quadric, 
		# 				'domains': (-10, 10), 
		# 				'optimizer': min
		# 				},
		# 'Rastrigin': {
		# 				'fitness_function': rastrigin, 
		# 				'domains': (-5.12, 5.12), 
		# 				'optimizer': min
		# 				},
		# 'Rosenbrock': {
		# 				'fitness_function': rosenbrock, 
		# 				'domains': (-5, 10), 
		# 				'optimizer': min
		# 				},
		# 'Schwefel': {
		# 				'fitness_function': schwefel, 
		# 				'domains': (-10, 10), 
		# 				'optimizer': min
		# 				},
		# 'Sphere': {
		# 				'fitness_function': sphere, 
		# 				'domains': (-10, 10), 
		# 				'optimizer': min
		# 				},
		# 'SumSquares': {
		# 				'fitness_function': sumsquares, 
		# 				'domains': (-10, 10), 
		# 				'optimizer': min
		# 				}
		}

parameter = {
		# , 'selection': ['page_rank', 'tournament'], 'roullet_wheel' 
		'encode': ['gray_code'], # , 'binary'
		'pairing': ['random'], 
		# 'domains': (min, max), defualt (0, 100) 
		'n_population': [500], 
		'dimension': [100], #, 20
		# 'n_sample': [500],
		'function': functions.keys()
}	 

k_nearest_neighbors = [20, 50] # 10, 20, 

iteration = 20
all_time = 1 * 10
fst_time = 1 * 2
lst_time = 1 * 3
		
parameters = [{}]

for key, params in parameter.items(): 
	new_parameters = []
	for old_param in parameters: 
		for param in params: 
			new_param = old_param.copy()
			new_param[key] = param
			new_parameters.append(new_param)
	parameters = new_parameters

def genetic_algorithm_report(columns):
	param_function = functions[columns['function']]
	combine_params = dict(**columns, **param_function)

	print(dict(columns).values())
	
	# combine_params['n_sample'] = int(combine_params['n_population'] / 2)
	combine_params['domains'] = [param_function['domains']] * combine_params['dimension']
	del combine_params['dimension'], combine_params['function']
	
	GA_tournament = GeneticAlgorithm(**combine_params)
	initial_population = GA_tournament.create_initial()
	GA_tournament.set_selection('tournament')

	logs = [] 
	
	for knn in k_nearest_neighbors:
		combine_params['k_nearest_neighbor'] = knn
		GA_page_rank = GeneticAlgorithm(**combine_params)
		GA_page_rank.set_selection('page_rank')
		GA_page_rank.set_cache_param()

		log_page_rank = GA_page_rank.optimize_by_time_hybrid(initial_population, all_time, 'tournament', fst_time, 100, 10, 'page_rank', lst_time)
		# log_page_rank = GA_page_rank.optimize_by_time(initial_population, time)
		logs.append(log_page_rank)

	log_basic = GA_tournament.optimize_by_time(initial_population, all_time)
	# log_basic = GA_tournament.optimize_by_time(initial_population, time)
	
	logs = [log_basic] + logs

	result = []
	for log in logs:
		result += [log['time'], len(log['optimization']), log['optimization'][-1]['fitness'], log]
	
	return result
	
name_columns = []
avg_columns = []
selections = ['tournament'] 

for knn in k_nearest_neighbors:
	selections.append('hybrid page_rank k={:}'.format(knn))

for selection in selections:
		for key in ['time', 'generation', 'optimal', 'log']: 
				name_columns.append('{:} {:}'.format(selection, key))
				
fixed_columns = parameters[0].keys()
name_columns_iterations = []

for i in range(1):
		name_columns_iterations.append([name_column + ' {:}'.format(i+1) for name_column in name_columns])

report = pd.DataFrame(columns=list(parameters[0].keys()) + [item for sublist in name_columns_iterations for item in sublist]) 

for index, parameter in enumerate(parameters):
	report = report.append([parameter], ignore_index=True)
  
	for name_columns_iteration in name_columns_iterations:
		report.loc[index, name_columns_iteration] = report.loc[[index], fixed_columns].apply(lambda columns: genetic_algorithm_report(columns), axis = 1).values[0]
    
	report.to_pickle('./report.pkl')
	report.to_excel("./report.xlsx")	

# =============================================================================
# for function in functions:
#	report[[function['name']+' time', function['name']+' generation', function['name']+' optimal', function['name']+' log']] = report.apply(lambda columns: genetic_algorithm_report(columns, function), axis = 1, result_type= 'expand' )
# =============================================================================


