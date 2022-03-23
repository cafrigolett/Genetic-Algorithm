import random
import time
import Reporter
import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby
# Modify the class name to match your student number.
class r0123456:

	def __init__(self):
		self.reporter = Reporter.Reporter(self.__class__.__name__)
		self.populationSize = 1000
		self.elitismRate = 0.01
		self.tournamentSize = 100
		self.mutationRate = 0.1
		self.scramSize = 3




	#helperfunctie voor mezelf
	def distanceM(self,filename):
		file = open(filename)
		distanceMatrix = np.loadtxt(file, delimiter=",")
		file.close()
		return distanceMatrix

	# The evolutionary algorithm's main loop
	def optimize(self, filename):
		# Read distance matrix from file.		
		file = open(filename)
		distanceMatrix = np.loadtxt(file, delimiter=",")
		file.close()

		population = self.createInitialPopulation(distanceMatrix)
		previousWinner = population[0]
		ct = 0
		meanObjective = 0.0
		bestObjective = 0.0
		objectiveFunction = []
		meanObjectiveFunction = []
		timeList = []

		while ( ct != 5):
			startTime = time.time()
			winners = self.takeBestOnes(population)
			matingPool = [self.selection(population) for _ in range(self.populationSize)]
			matingPoolTours = [a_tuple[0] for a_tuple in matingPool]
			crossovers = self.crossover(matingPoolTours)
			mutations = self.mutate(crossovers)
			leftovers = self.eliminate(mutations,distanceMatrix)
			newPopulation = leftovers + winners
			population = sorted(newPopulation, key=lambda x: x[1])
			newWinner = population[0]
			epsilon = previousWinner[1] - newWinner[1]
			print(len(crossovers))
			if epsilon < 0.0001:
				ct = ct + 1
			else:
				ct = 0
			previousWinner = newWinner

			bestObjective = newWinner[1]
			meanObjective = self.calcMean(population)

			meanObjectiveFunction.append(meanObjective)
			objectiveFunction.append(bestObjective)
			timeList.append(time.time() - startTime)
			print(newWinner, "Mean objective:",meanObjective)

		print(newWinner, "Mean objective:", meanObjective, "Time:",np.cumsum(timeList)[-1])
		self.plotobjective(objectiveFunction, meanObjectiveFunction, timeList)
		# Your code here.
		# yourConvergenceTestsHere = True
		# while( yourConvergenceTestsHere ):
		# 	meanObjective = 0.0
		# 	bestObjective = 0.0
		# 	bestSolution = np.array([1,2,3,4,5])

			# Your code here.

			# Call the reporter with:
			#  - the mean objective function value of the population
			#  - the best objective function value of the population
			#  - a 1D numpy array in the cycle notation containing the best solution
			#    with city numbering starting from 0
			#timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
			#if timeLeft < 0:
				#break

		# Your code here.
		return 0

	def createInitialPopulation(self,distanceMatrix):
		size = len(distanceMatrix)
		population = []
		for i in range(0,self.populationSize):
			tour = self.createTour(size)
			distance = self.length(distanceMatrix,tour)
			population.append((tour,distance))
		population = sorted(population, key=lambda x: x[1])
		return population

	def createTour(self,number):
		numbers = list(range(1,number+1))
		new = random.sample(numbers,len(numbers))
		return new

	def length(self,distanceMatrix,tour):
		totalDistance = 0
		for i in range(0,len(tour)):
			fromCity = tour[i]
			toCity = None
			if i+1 < len(tour):
				toCity = tour[i+1]
			else:
				toCity = tour[0]
			totalDistance += distanceMatrix[fromCity-1][toCity-1]
		return totalDistance

	def takeBestOnes(self,population):
		count = self.elitismRate*self.populationSize
		bestOnes = population[:int(count)]
		return bestOnes

	def selection(self,population):
		parents = random.choices(population, k=self.tournamentSize)
		parents = sorted(parents, key=lambda x: x[1])
		bestparent = parents[0]
		return bestparent

	def crossover(self,matingPoolTours):
		random.shuffle(matingPoolTours)
		crossovers = []
		while len(matingPoolTours) >= 2:
			parent1 = matingPoolTours.pop()
			parent2 = matingPoolTours.pop()
			child1= self.edge_crossover(parent1,parent2)
			crossovers.append(child1)
			
		return crossovers + matingPoolTours


	def mutate(self,crossovers):
		mutants = []
		candMutants = []
		mutantsSize = round(self.mutationRate*len(crossovers))
		#print(mutantsSize)
		tmpNums = list(range(0, len(crossovers)))
		#print(tmpNums)
		indMutant = random.sample(tmpNums, mutantsSize)
		#print(len(crossovers))
		#randomly select n = mutantsSize and pop them from crossovers
		indMutant = sorted(indMutant, reverse = True)
		#print(indMutant)
		for i in range(len(indMutant)):	
			candMutant = crossovers.pop(indMutant[i])
			candMutants.append(candMutant)
		#print(candMutants)
		#Mutate each  candMutant in Candmutants
		for j in range(len(candMutants)) : 
			mutant = self.scramble(candMutants[j])
			#print(mutant) 
			mutants.append(mutant)
		#print(mutants)
		#print(crossovers)
		return mutants + crossovers 

	def scramble(self, tour):
		randNum = random.randint(0, len(tour)-self.scramSize)
		copy = tour[randNum:randNum+self.scramSize]
		random.shuffle(copy)  
		tour[randNum:randNum+self.scramSize] = copy
		return tour

	def eliminate(self,mutations,distanceMatrix):
		population = []
		for i in range(len(mutations)):
			tour = mutations[i]
			distance = self.length(distanceMatrix,tour)
			population.append((tour,distance))
		population = sorted(population, key=lambda x: x[1] )
		amount = self.populationSize*(1-self.elitismRate)
		population = population[:int(amount)]
		return population

	def calcMean(self, population):
		objectives = []
		for i in population:
			objectives.append(i[1])
		return(np.mean(objectives))

	def plotobjective(self, objectives, meanObjectives, timeList):
		#x = np.arange(1, len(objectives)+1)
		timeList = np.cumsum(timeList) # The timeList contains each iteration time. To plot it as a function of time, we need to take the cumsum()
		plt.plot(timeList, objectives, label = 'best objective')
		plt.plot(timeList, meanObjectives, label = 'mean objective')
		plt.xlabel("Time (s)")
		plt.ylabel("Fitness")
		plt.legend()
		plt.show()
		
	def edge_crossover(self, parent1, parent2):
		edges = find_edges(parent1, parent2)

		return crossover(parent1, parent2, edges)		

#Gets edges for parent1, parent2
def find_edges(parent1, parent2):
  parent1_edges = calc_edges(parent1)
  parent2_edges = calc_edges(parent2)
  merged_edges = merge_edges(parent1_edges, parent2_edges)  

  return merged_edges

#calculates edges for an individual
def calc_edges(individual):
  edges = []
  
  for position in range(len(individual)):
    if position == 0:
      edges.append([individual[position], (individual[-1], individual[position+1])])
    elif position < len(individual)-1:
      edges.append([individual[position], (individual[position-1], individual[position+1])])
    else:
      edges.append([individual[position], (individual[position-1], individual[0])])
  
  return edges

#sort the edges    
def sort_edges(individual):
  return sorted(individual, key=lambda x: x[0])

#perform an union on two parents
def merge_edges(parent1, parent2):
  parent1 = sort_edges(parent1)
  parent2 = sort_edges(parent2)

  edges = []
  for val in range(len(parent1)):
    edges.append([parent1[val][0], union(parent1[val][1], parent2[val][1])])
  
  return edges

#part of merge_edges - unions 2 individual
def union(individual1, individual2):
  edges = list(individual1)

  for val in individual2:
    if val not in edges:
      edges.append(val)
  return edges


parent1 = [4, 3, 1, 2, 5, 6]
parent2 = [1, 2, 3, 4, 6, 5]

#Edge recombination operator 
def crossover(parent1, parent2, edges):
  k = []
  previous = None
  current = random.choice([parent1[0], parent2[0]])

  while True:
    k.append(current)

    if(len(k) == len(parent1)):
      break
    
    previous = remove_node_from_neighbouring_list(current, edges)
    current_neighbour = get_current_neighbour(previous, edges)

    next_node = None
    if len(current_neighbour) > 0:
      next_node = get_best_neighbour(current_neighbour)
    else:
      next_node = get_next_random_neighbour(k, edges)
   
    current = next_node[0]
  return k

def remove_node_from_neighbouring_list(node, neighbour_list):
  removed_node = None

  for n in neighbour_list:
    if n[0] == node:
      removed_node = n
      neighbour_list.remove(n)
    
    if node in n[1]:
      n[1].remove(node)
  
  return removed_node

#return neighbours for a give node(s)
def get_current_neighbour(nodes, neighbour_lists):
  neighbours = []

  if nodes is not None:
    for node in nodes[1]:
      for neighbour in neighbour_lists:
        if node == neighbour[0]:
          neighbours.append(neighbour)

  return neighbours

#returns the best possible neighbour
def get_best_neighbour(neighbour):
  if len(neighbour) == 1:
    return neighbour[0]
  else:
    group_neighbour = group_neighbours(neighbour)
    return random.choice(group_neighbour[0])[1]

#part of get_best_neighbour   
def group_neighbours(neighbours):
  sorted_neighbours = []

  #store length of each individual neighbour + neighbour in a list
  for neighbour in neighbours:
    sorted_neighbours.append((len(neighbour[1]), neighbour))
  
  #sort the new list
  sort_edges(sorted_neighbours)

  #group the neighbour by their size
  groups = []
  for k, g in groupby(sorted_neighbours, lambda x: x[0]):
    groups.append(list(g))

  return groups

#returns a random neighbour from remaining_edges that does not exist in current_path
def get_next_random_neighbour(current_path, remaining_edges):
  random_node = None

  while random_node is None:
    tmp_node = random.choice(remaining_edges)

    if tmp_node[0] not in current_path:
      random_node = tmp_node
  
  return random_node



		
	
	
	
	











'''
	def PMX(self,parent1,parent2):

		zeros = [0]*len(parent1)

		firstCrossPoint = random.randint(0, len(parent1) - 1)
		secondCrossPoint = random.randint(firstCrossPoint + 1, len(parent1))

		parent1MiddleCross = parent1[firstCrossPoint:secondCrossPoint]
		parent2MiddleCross = parent2[firstCrossPoint:secondCrossPoint]

		child1 = (zeros[:firstCrossPoint] + parent1MiddleCross + zeros[secondCrossPoint:])
		child2 = (zeros[:firstCrossPoint] + parent2MiddleCross + zeros[secondCrossPoint:])

		#i1 are the elements that occur in child2 and not in child1
		#i2 are the elements that occur in child1 and not in child2
		i1 = []  # i1 = [8,2]
		i2 = []  # i2 = [4,7]
		for k in range(len(parent2MiddleCross)):
			if parent2MiddleCross[k] not in parent1MiddleCross:
				i1.append(parent2MiddleCross[k])
			if parent1MiddleCross[k] not in parent2MiddleCross:
				i2.append(parent1MiddleCross[k])

		#j1 are the elements of child1 that are on the same position of the elements of i1 in child2
		#j2 are the elements of child2 that are on the same position of the elements of i1 in child1
		j1 = []  # j1 = [4,5]
		j2 = []	 # j2 = [8,5]
		for k in range(len(i1)):
			index = parent2.index(i1[k])
			j1.append(child1[index])

		for k in range(len(i2)):
			index = parent1.index(i2[k])
			j2.append(child2[index])


		for k in range(len(i1)):
			index = parent2.index(j1[k])
			number = child1[index]
			while number != 0:
				index = parent2.index(number)
				number = child1[index]
			child1[index] = i1[k]

		for k in range(len(i2)):
			index = parent1.index(j2[k])
			number = child2[index]
			while number != 0:
				index = parent1.index(number)
				number = child2[index]
			child2[index] = i2[k]

		for n in range(len(child1)): # 0 .. 8
			if child1[n] == 0:
				child1[n] = parent2[n]
			if child2[n] == 0:
				child2[n] = parent1[n]

		return (child1,child2)

'''