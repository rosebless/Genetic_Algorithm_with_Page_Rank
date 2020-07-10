
import time
import math
import sys
import json
import random
import psutil
import numpy as np
from scipy import sparse
from fast_pagerank import pagerank 

'''
	parameter: 
		selection: 'page_rank', 'roullet_wheel' or 'tournament', defualt = 'tournament'
		encode: 'gray_code' or 'binary', defualt = 'binary' 
		pairing: 'custom' or 'random', defualt = 'random' 
		domains: [(min, max)]*dimension, defualt [(0, 100)]
			a list of tuple or list that have 2 dimention define min and 
			max value for each dimension of domain
		digits: [integer]*dimension, defualt [2] 
			a list of number of digits for each dimension of domain
		n_population: interger, defualt 100 
			a number of population each generation 
		n_sample: interger, defualt 50 
			a number of selection sample	
		fitness_function: function, defualt lambda x: x 
			a function that return fitness value of input chomosomes 
		optimizer: function, defualt min 
			a function that return the optimized value of list 
'''

class GeneticAlgorithm:
	permutation_prob = 0.05
	cross_over_prob = 0.5
	selection_fn = None 
	encoding = None
	decoding = None
	pairing_fn = None	
	domains = [(0, 100)]
	n_population = 100
	fitness_function = lambda x: x
	optimizer = min
	chromosome_length = 10
	k_nearest_neighbor = 5
	cache_decode_transfrom = {}
	time_set_cache = 0
	transform_ratio = 1023
	get_cache_table_key = lambda x: x
	find_distance = None
	encode = 'binary'

	def set_selection(self, selection):
		self.selection = selection
		if selection == 'roullet_wheel': 
			self.selection_fn = self.roullet_wheel_selection
		elif selection == 'page_rank': 
			self.selection_fn = self.page_rank_selection 
		else: 
			self.selection_fn = self.tournament_selection 
	
	def set_cache_param(self):
		start_time = time.time()
		posible_domain = GeneticAlgorithm.create_posible_domain(['0'*self.chromosome_length])
		for c in posible_domain:
			number = self.gray_code_to_number(c) if self.encode == 'gray_code' else self.binary_to_number(c)
			if len(set(self.domains)) == 1:
				self.get_cache_table_key = lambda c, domain: c
				self.cache_decode_transfrom[self.get_cache_table_key(c, self.domains[0])] = self.inv_transform_decimal(number, self.domains[0])
			else:
				self.get_cache_table_key = lambda c, domain: '_'.join([str(d) for d in domain] + [c])
				for domain in self.domains:
					self.cache_decode_transfrom[self.get_cache_table_key(c, domain)] = self.inv_transform_decimal(number, self.domain)
		self.time_set_cache = time.time() - start_time

	def __init__(self, selection='tournament', encode='binary', pairing='random', chromosome_length=10, domains=[(0, 100)], n_population=100, fitness_function=lambda x: x, optimizer=min, k_nearest_neighbor=5):
		# selection 
		self.set_selection(selection) 

		# encode 
		if encode == 'gray_code': 
			self.encoding = self.decimals_to_gray_code 
			self.decoding = self.gray_code_to_decimals
		else: 
			self.encoding = self.decimals_to_binary 
			self.decoding = self.binary_to_decimals

		# pairing 
		if pairing == 'custom':	
			self.pairing_fn = self.random_pairing
		else: 
			self.pairing_fn = self.random_pairing 

		# domains, digits, n_population, n_sample, fitness_function, optimizer
		for domain in domains:
			if domain[1] < domain[0]:
				raise Exception('minimum of domain must less than maximum of domain. but max is {:} and min is {:}.'.format(domain[1], domain[0]))
		if type(chromosome_length) != int:
			raise Exception('chromosome_length must be a integers.')
		if type(n_population) != int:
			raise Exception('n_population must be a integers.')
		if type(k_nearest_neighbor) != int:
			raise Exception('k_nearest_neighbor must be a integers.')
		if not callable(fitness_function):
			raise Exception('fitness_function must be a function.')
		if not callable(optimizer):
			raise Exception('optimizer must be a function.') 

		self.chromosome_length = chromosome_length
		self.domains = domains
		self.n_population = n_population
		self.k_nearest_neighbor = k_nearest_neighbor
		self.fitness_function = fitness_function
		self.optimizer = optimizer
		self.encode = encode

		# max_length
		self.transform_ratio = (2**self.chromosome_length) - 1
		# self.chromosome_lengths = [math.ceil(math.log((domain[1] - domain[0]) * 10 ** digits[index] , 2)) for index, domain in enumerate(domains)] 

	def optimize_by_iterations(self, initial_population = np.array([]), n_iterations=100):
		start_time = time.time()
		optimization_log = []

		len_n_iterations = len(str(n_iterations))
		
		if self.selection == 'page_rank':
			ga_name = 'GA page_rank k = {:}.'.format(self.k_nearest_neighbor)
		else:
			ga_name = 'GA {:}.'.format(self.selection)

		print('processing {:}'.format(ga_name))

		if len(self.cache_decode_transfrom) == 0:
			self.find_distance = self.euclidean_distance
		else: 
			self.find_distance = self.find_distance_with_cache
		
		population = self.optimize_initial(initial_population)
		current_time = time.time()
		optimization_log.append(dict(self.optimization(population), **{'time': current_time - start_time}))
		sys.stdout.write('\r Generated {:>{:}} / {:} use {:>6.2f} second.'.format(0, len_n_iterations, n_iterations, time.time() - start_time))

		# loop for next generation 
		for i in range(n_iterations):
			population = self.optimize_iteration(population)
			
			# add new optimize 
			previous_optimize = optimization_log[-1] 
			population[0].append(previous_optimize['chromosome']) # chromosome list
			population[1].append(previous_optimize['fitness']) # fitness list

			current_time = time.time()
			optimization_log.append(dict(self.optimization(population), **{'time': current_time - start_time}))
			sys.stdout.write('\r Generated {:>{:}} / {:} use {:>6.2f} second.'.format(i+1, len_n_iterations, n_iterations, time.time() - start_time))
			# sys.stdout.write('\r Generated {:>{:}} / {:} use {:>6.2f} second cache table size {:8.3f} mb {:} items.'.format(i+1, len_n_iterations, n_iterations, time.time() - start_time, sys.getsizeof(self.cache_distance_table)/(10**6), len(list(self.cache_distance_table))))
		
		print('\nfinished {:}'.format(ga_name))

		# return the log and time use in seconds for create report 
		log = {
				'optimization': optimization_log, 
				'time': time.time() - start_time,
				'init_cache_time': self.time_set_cache
		}
		return log 

	def optimize_by_time(self, initial_population = np.array([]), limit_time=300):
		start_time = time.time()
		iteration = 0
		optimization_log = []
		
		if self.selection == 'page_rank':
			ga_name = 'GA page_rank k = {:}.'.format(self.k_nearest_neighbor)
		else:
			ga_name = 'GA {:}.'.format(self.selection)

		print('processing {:}'.format(ga_name))

		if len(self.cache_decode_transfrom) == 0:
			self.find_distance = self.euclidean_distance
		else: 
			self.find_distance = self.find_distance_with_cache
		
		population = self.optimize_initial(initial_population)
		current_time = time.time()
		optimization_log.append(dict(self.optimization(population), **{'time': current_time - start_time}))
		sys.stdout.write('\r Generated {:} use {:>6.2f} / {:} second.'.format(iteration, current_time - start_time, limit_time))

		# loop for next generation 
		while current_time - start_time < limit_time:
			for i in range(10):
				iteration += 1
				population = self.optimize_iteration(population)

				# add new optimize 
				previous_optimize = optimization_log[-1] 
				population[0].append(previous_optimize['chromosome']) # chromosome list
				population[1].append(previous_optimize['fitness']) # fitness list

				current_time = time.time()
				optimization_log.append(dict(self.optimization(population), **{'time': current_time - start_time}))
				sys.stdout.write('\r Generated {:} use {:>6.2f} / {:} second.'.format(iteration, current_time - start_time, limit_time))
				# sys.stdout.write('\r Generated {:>{:}} / {:} use {:>6.2f} second cache table size {:8.3f} mb {:} items.'.format(i+1, len_n_iterations, n_iterations, time.time() - start_time, sys.getsizeof(self.cache_distance_table)/(10**6), len(list(self.cache_distance_table))))
		
		print('\nfinished {:}'.format(ga_name))

		# return the log and time use in seconds for create report 
		log = {
				'optimization': optimization_log, 
				'time': time.time() - start_time,
				'init_cache_time': self.time_set_cache
		}
		return log 

	def optimize_by_time_hybrid(self, initial_population = np.array([]), limit_time=300, fisrt='tournament', time_fisrt=0, n_tournament=8, n_page_rank=2, last='page_rank', time_last=0):
		start_time = time.time()
		iteration = 0
		optimization_log = []
		time_mid = limit_time - time_fisrt - time_last

		print('processing GA hybrid with fisrt={:}, time_fisrt={:}, n_tournament={:}, n_page_rank={:}, last={:}, time_last={:}.'.format(fisrt, time_fisrt, n_tournament, n_page_rank, last, time_last))

		if len(self.cache_decode_transfrom) == 0:
			self.find_distance = self.euclidean_distance
		else: 
			self.find_distance = self.find_distance_with_cache
		
		population = self.optimize_initial(initial_population)
		current_time = time.time()
		optimization_log.append(dict(self.optimization(population), **{'time': current_time - start_time}))
		sys.stdout.write('\r Generated {:} Generation, use {:>6.2f} / {:} second.'.format(iteration, current_time - start_time, limit_time))

		def iterations_process(n, iteration, population, optimization_log, current_time, start_time, optimize_iteration, optimization):
			for i in range(n):
				iteration += 1 
				population = optimize_iteration(population)

				# add new optimize 
				previous_optimize = optimization_log[-1] 
				population[0].append(previous_optimize['chromosome']) # chromosome list
				population[1].append(previous_optimize['fitness']) # fitness list

				current_time = time.time()
				optimization_log.append(dict(optimization(population), **{'time': current_time - start_time}))
				sys.stdout.write('\r Generated {:} Generation, use {:>6.2f} / {:} second.'.format(iteration, current_time - start_time, limit_time))
			return iteration, population, optimization_log, current_time, start_time

		if time_fisrt != 0:
			self.selection_fn = self.page_rank_selection if fisrt == 'page_rank' else self.tournament_selection 
			while current_time - start_time < time_fisrt:
				iteration_update = iterations_process(10, iteration, population, optimization_log, current_time, start_time, self.optimize_iteration, self.optimization)
				iteration, population, optimization_log, current_time, start_time = iteration_update

		if time_mid != 0:		
			while current_time - start_time < time_mid:
				self.selection_fn = self.tournament_selection
				iteration_update = iterations_process(n_tournament, iteration, population, optimization_log, current_time, start_time, self.optimize_iteration, self.optimization)
				iteration, population, optimization_log, current_time, start_time = iteration_update

				self.selection_fn = self.page_rank_selection
				iteration_update = iterations_process(n_page_rank, iteration, population, optimization_log, current_time, start_time, self.optimize_iteration, self.optimization)
				iteration, population, optimization_log, current_time, start_time = iteration_update		
		
		if time_last != 0:
			self.selection_fn = self.page_rank_selection if last == 'page_rank' else self.tournament_selection 
			while current_time - start_time < time_last:
				iteration_update = iterations_process(10, iteration, population, optimization_log, current_time, start_time, self.optimize_iteration, self.optimization)
				iteration, population, optimization_log, current_time, start_time = iteration_update
				
		print('\nfinished GA swap.')

		# return the log and time use in seconds for create report 
		log = {
				'optimization': optimization_log, 
				'time': time.time() - start_time,
				'init_cache_time': self.time_set_cache
		}
		return log 
	
	def optimize_by_same_results(self, initial_population = np.array([]), n_same_results=10):
		start_time = time.time()
		optimization_log = [] 
		
		print('GA processing.')
		
		if len(self.cache_decode_transfrom) == 0:
			self.find_distance = self.euclidean_distance
		else: 
			self.find_distance = self.find_distance_with_cache
		
		population = self.optimize_initial(initial_population)
		current_time = time.time()
		optimization_log.append(dict(self.optimization(population), **{'time': current_time - start_time}))
		sys.stdout.write('\r Generated {:>3} use {:>6.2f} second.'.format(len(optimization_log), time.time() - start_time))
		
		# loop for next generation 
		while self.is_not_optimize(optimization_log, n_same_results): 
			population = self.optimize_iteration(population)
			
			# add new optimize 
			previous_optimize = optimization_log[-1] 
			population[previous_optimize['chromosome']] = previous_optimize['fitness']

			current_time = time.time()
			optimization_log.append(dict(self.optimization(population), **{'time': current_time - start_time}))
			sys.stdout.write('\r Generated {:>3} use {:>6.2f} second.'.format(len(optimization_log), time.time() - start_time))
		
		print('\nGA finished.')
		# return the log and time use in seconds for create report 
		log = {
				'optimization': optimization_log, 
				'time': time.time() - start_time,
				'init_cache_time': self.time_set_cache
		}
		return log 
	
	def optimize_initial(self, initial_population = np.array([])):
		if type(initial_population) != np.ndarray:
			raise Exception('initial_population must be a np.array.')
		
		# initailization first generation 
		initial_population = self.create_initial() if initial_population.size == 0 else initial_population # decimal 
		chromosome_list = [self.encoding(decimals) for decimals in initial_population] # gray_code
		fitness_list = [self.fitness_function(decimals) for decimals in initial_population] 
		population = [chromosome_list, fitness_list]
		
		return population
	
	def optimize_iteration(self, population):
		# selection 
		selected_chomosome = self.selection_fn(population, self.optimizer) # gray_code
		l = self.chromosome_length
		chromosome_list = []
		fitness_list = []
		# sampling 
		for t in range(self.n_population): 
			parent_chomosome = self.pairing_fn(selected_chomosome) 
			child_chomosome = self.cross_over(parent_chomosome[0], parent_chomosome[1])
			child_chomosome = self.permutation(child_chomosome)
			chromosome_list.append(child_chomosome) # gray_code
			
			# evaluation fitness value 
			# fitness_value = self.fitness_function(self.decoding(child_chomosome))
			fitness_value = self.fitness_function([self.cache_decode_transfrom[child_chomosome[i*l:(i+1)*l]] for i in range(len(self.domains))])
			fitness_list.append(fitness_value) # fitness_value
			
		new_population = [chromosome_list, fitness_list]

		return new_population
		

############################################################# Selection Methods #############################################################
	
	@staticmethod
	def bit_distance(chromosome_one, chromosome_two): 
		distance = [0 if c1 == c2 else 1 for c1, c2 in zip(chromosome_one, chromosome_two)]
		distance = sum(distance)
		return distance 
	
	@staticmethod 
	def add_sort(items, new):
		for index, item in enumerate(items):
			if new < item:
				return items[:index] + [new] + items[index:-1]
		return items
	
	@staticmethod 
	def find_min(items, n):
		mins = sorted(items[:n], reverse=False)
		for item in items[n:]:
			if item < mins[-1]:
				mins = GeneticAlgorithm.add_sort(mins, item)
		return mins
	
	def find_distance_with_cache(self, chromosome_one, chromosome_two):
		distance = 0
		l = self.chromosome_length
		for i, domain in enumerate(self.domains):
			decimal_one = self.cache_decode_transfrom[self.get_cache_table_key(chromosome_one[i*l:(i+1)*l], domain)]
			decimal_two = self.cache_decode_transfrom[self.get_cache_table_key(chromosome_two[i*l:(i+1)*l], domain)]
			distance += (decimal_one - decimal_two)**2
		return distance 		
		
	def euclidean_distance(self, chromosome_one, chromosome_two): 
		decimals_one = self.decoding(chromosome_one)
		decimals_two = self.decoding(chromosome_two)
		distance = [(decimal_one - decimal_two)**2 for decimal_one, decimal_two in zip(decimals_one, decimals_two)]
		distance = math.sqrt(sum(distance))
		return distance 

	def directed_graph_cal_first(self, points, values, optimizer):
		k = self.k_nearest_neighbor
		direction_list = []
		
		distance_table = np.full((self.n_population, self.n_population), len(points[0]), dtype=int)
		for i in range(self.n_population):
			for j in range(i+1, self.n_population):
				distance = self.find_distance(points[i], points[j])
				distance_table[i, j] = distance
				distance_table[j, i] = distance
	
		for i in range(self.n_population): 
			distances = distance_table[i, :]
			mins_k = self.find_min(distances, k)
			unique_distances = list(set(mins_k))
			j_candidate = []
			is_break = False
			for distance in unique_distances:
				j_indexs = np.where(distances==distance)[0].tolist()
				for t in range(len(j_indexs)):
					j = random.choice(j_indexs)
					j_indexs.remove(j)
					j_candidate.append(j)
					optimize = optimizer([values[i], values[j]])
					if values[i] == optimize:
						direction_list.append([j, i])
					if values[j] == optimize:
						direction_list.append([i, j])
					if len(j_candidate) == k:
						is_break = True
						break
				if is_break:
					break
		return direction_list

	def page_rank(self, chromosome_list, fitness_list, optimizer):
		direction_list = np.array(self.directed_graph_cal_first(chromosome_list, fitness_list, optimizer))
		weights = np.ones(direction_list.shape[0], dtype=int) 
		G = sparse.csr_matrix((weights, (direction_list[:,0], direction_list[:,1])), shape=(self.n_population, self.n_population))
		return pagerank(G, p=0.85)

	def page_rank_selection(self, population, optimizer): 
		chromosome_list, fitness_list = population
		fitness_list = self.page_rank(chromosome_list, fitness_list, optimizer) 
		population = [chromosome_list, fitness_list]
		return self.tournament_selection(population, max)

	def roullet_wheel_selection(self, population, optimizer):
		chromosome_list, fitness_list = population
		prop_normal = np.array(fitness_list) / sum(fitness_list) 
		return np.random.choice(a=chromosome_list, size=self.n_population, replace=True, p=prop_normal) 

	def tournament_selection(self, population, optimizer): 
		chromosome_list, fitness_list = population
		selected_chromosome = []
		for i in range(self.n_population):
			index_pair = np.random.choice(range(self.n_population), size=2, replace=False) 
			value_one = fitness_list[index_pair[0]]
			value_two = fitness_list[index_pair[1]]
			if value_one == optimizer([value_one, value_two]):
				selected_chromosome.append(chromosome_list[index_pair[0]])
			else: 
				selected_chromosome.append(chromosome_list[index_pair[1]])
		return selected_chromosome

################################################################### Tools ###################################################################		
		
	def cross_over(self, parent_one, parent_two): 
		child = list(parent_one)
		for index in range(min(len(parent_one), len(parent_two))): 
			rand = np.random.random_sample() 
			if rand <= self.cross_over_prob: 
				child[index] = parent_two[index]
				# child_one[index], child_two[index] = child_two[index], child_one[index]	
		child = ''.join(child)
		return child
		
	def permutation(self, chromosome):
		chromosome = list(chromosome)
		for index in range(len(chromosome)): 
			rand = np.random.random_sample() 
			if rand <= self.permutation_prob: 
				if chromosome[index] == '0':
					inverse_gene = '1' 
				else: 
					inverse_gene = '0' 
				chromosome[index] = inverse_gene
		chromosome = ''.join(chromosome)
		return chromosome 

	def create_initial(self): 
		return np.transpose([(domain[1] - domain[0]) * np.random.random_sample(self.n_population) + domain[0] for index, domain in enumerate(self.domains)])

	@staticmethod
	def random_pairing(sample):
		return	np.random.choice(sample, size=2, replace=False) 

	def optimization(self, population): 
		optimize_chromosome, optimize_fitness = self.optimize_by_value(population) 
		l = self.chromosome_length 
		optimize = {
				'chromosome': optimize_chromosome,
				# 'input': self.decoding(optimize_chromosome), 
				'input': [self.cache_decode_transfrom[optimize_chromosome[i*l:(i+1)*l]] for i in range(len(self.domains))],
				'fitness': optimize_fitness
			}
		return optimize 

	def optimize_by_value(self, population): 
		keys, values = population
		optimize_value = self.optimizer(values)
		return keys[values.index(optimize_value)], optimize_value 

	@staticmethod
	def is_not_optimize(optimization_log, n_check): 
		n_log = len(optimization_log)
		if n_log < n_check:
			return True 
		for i in range(1, n_check): 
			for j in range(i, n_check): 
				index_i = n_log - i
				index_j = n_log - (j+1)
				if optimization_log[index_i]['fitness'] != optimization_log[index_j]['fitness']: 
					return True 
		return False 
	
	@staticmethod 
	def create_posible_domain(roots, i=0): 
		current_domain = []
		for root in roots: 
			root_zero = root[:i] + '0' + root[i+1:]
			root_one = root[:i] + '1' + root[i+1:]
			current_domain += [root_zero, root_one]
		if i != len(roots[0])-1: 
			return GeneticAlgorithm.create_posible_domain(current_domain, i+1)
		else:
			return current_domain 
		
	@staticmethod
	def create_distace_key(chromosome_one, chromosome_two):
		return '_'.join([chromosome_one, chromosome_two]) if chromosome_one < chromosome_two else '_'.join([chromosome_two, chromosome_one])

	@staticmethod
	def create_cache_table(length):
		distance_table = {}
		posible_domain = GeneticAlgorithm.create_posible_domain(['0'*length])
		for c1 in posible_domain:
			for c2 in posible_domain:
				key = GeneticAlgorithm.create_distace_key(c1, c2)
				if key not in distance_table:
					distance_table[key] = GeneticAlgorithm.bit_distance(c1, c2)
		return distance_table
		
	@staticmethod
	def inverse_binary(binary): 
		return '0' if binary == '1' else '1'

	@staticmethod
	def number_to_binary(number, max_length): 
		binary = bin(number)[2:].zfill(max_length) 
		return	binary

	@staticmethod
	def binary_to_number(binary): 
		number = int(binary, 2) 
		return number

	@staticmethod
	def number_to_gray_code(number, max_length): 
		binary = GeneticAlgorithm.number_to_binary(number, max_length)
		b = ['0'] + list(binary) 
		g = [b[i] if b[i-1] == '0' else GeneticAlgorithm.inverse_binary(b[i]) for i in range(1, max_length + 1)]
		gray = ''.join(g)
		return gray

	@staticmethod
	def gray_code_to_number(gray):	
		g = list(gray) 
		b = [str(g[:i+1].count('1') % 2) for i in range(len(g))] 
		binary = ''.join(b)
		number = GeneticAlgorithm.binary_to_number(binary)
		return number
	
	def transform_decimal(self, decimal, domain):
		return (decimal - domain[0]) * self.transform_ratio / (domain[1] - domain[0]) # x_b = (x_a-min_a) * (max_b - min_b) / (max_a - min_a) + min_b

	def inv_transform_decimal(self, decimal, domain):
		return decimal * (domain[1] - domain[0]) / self.transform_ratio + domain[0]

	def decimals_to_binary(self, decimals):
		binarys = ''
		for index, domain in enumerate(self.domains):
			number = round(self.transform_decimal(decimals[index], domain))
			binarys += self.number_to_binary(number, self.chromosome_length) 
		return binarys

	def binary_to_decimals(self, binarys):
		decimals = []
		for index, domain in enumerate(self.domains): 
			binary = binarys[index*self.chromosome_length:(index+1)*self.chromosome_length]
			number = self.binary_to_number(binary)
			decimals.append(self.inv_transform_decimal(number, domain))
		return decimals

	def decimals_to_gray_code(self, decimals):
		grays = ''
		for index, domain in enumerate(self.domains):
			number = int(round(self.transform_decimal(decimals[index], domain)))
			grays += self.number_to_gray_code(number, self.chromosome_length) 
		return grays

	def gray_code_to_decimals(self, grays): 
		decimals = []
		for index, domain in enumerate(self.domains): 
			gray = grays[index*self.chromosome_length:(index+1)*self.chromosome_length]
			number = self.gray_code_to_number(gray)
			decimals.append(self.inv_transform_decimal(number, domain))
		return decimals

