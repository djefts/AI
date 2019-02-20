import copy
import math
import numpy as np

# Andrew Forste & David Jefts
# CS455 Spring 2019
# HW2, question #3

''' TODO:
        Initial chromosome population
        Crosover
        Return final result
'''


class multiKnapsack:
    def __init__(self, p, w, W, k):
        if len(p) != len(w):
            print("Please ensure number of weights and profit match")
            return
        if k < 1 or W < 1:
            print("Please supply positive weight and number of knapsacks")
            return
        
        items = []
        for i in range(len(p)):
            items.append(Item(p, w))
        
        aChromosome = Chromosome(k, items, W)
        aChromosome.mutate()
        print(aChromosome)
        aChromosome.mutate()
        print(aChromosome)
        aChromosome.mutate()
        print(aChromosome)
        aChromosome.mutate()
        print(aChromosome)
        # for i in range(len(p)):
        #   print(p[i])


class Item:
    def __init__(self, p, w):
        self.p = p
        self.w = w


class Chromosome:
    def __init__(self, numKnapSacks, items, maxWeight, chromosome=None):
        self.k = numKnapSacks
        self.W = maxWeight
        self.n = len(items)
        self.items = items
        self.bitsPerGene = math.ceil(math.log2(k))
        self.knapsacks = [(0, 0)] * numKnapSacks
        self.chromosome = []
        
        #Build new
        if chromosome==None:
            gene = [0] * self.bitsPerGene
            self.chromosome = gene * self.n
            print("Created chromosome: " + str(self))
        #Copy parent
        else:
            self.chromosome=chromosome[:]

    def mutate(self):
        #Find a place to mutate
        locationOfMutation = np.random.randint(0,len(self.chromosome))
        bitSwap = self.chromosome[locationOfMutation]

        #Mutate if will create legal child only
        tempChromosome = self.clone()
        if bitSwap==0:
            tempChromosome.chromosome[locationOfMutation]=1
            if tempChromosome.verifyLegal():
                self.chromosome=tempChromosome.chromosome
            else:
                self.mutate()
        else:
            tempChromosome.chromosome[locationOfMutation]=0
            if tempChromosome.verifyLegal():
                self.chromosome=tempChromosome.chromosome
            else:
                self.mutate()
    
    #Used to determine if object is legal
    def verifyLegal(self):
        binary = ''
        for gene in range(0, self.n, self.bitsPerGene):
            for bit in range(self.bitsPerGene):
                if self.chromosome[gene + bit] == 0:
                    binary += '0'
                else:
                    binary += '1'
            if int(binary, 2) > k:
                return False
        return True
    
    def set_knapsacks(self):
        """
        set the profit and weight for each knapsack at the time of calling
        each knapsack set as knapsack[n] = (curr_profit, curr_weight)
        """
        profit = 0
        weight = 0
        for i in range(0, self.k, self.bitsPerGene):
            bag_num = ""
            for a in range(self.bitsPerGene):
                bag_num += self.chromosome[a]
            if not bag_num == "000":  # if the item is placed in a bag
                profit += self.items[i].p
                for item in self.items:
                    weight += item.w
                self.knapsacks[int(bag_num)] = (profit, weight)
    
    def fitness(self):
        fitness = 0
        self.set_knapsacks()
        for bag in self.knapsacks:
            profit = bag[0]
            weight_ratio = 1 + ((self.W - weight) / self.W)
            fitness += profit * weight_ratio
        return fitness
    
    def clone(self):
        return Chromosome(self.k, self.items, self.W, self.chromosome)

    def __str__(self):
        return str(self.chromosome)


class GA:
    def __init__(self):
        self.populationSize = 100
        self.numGenerations = 1000
        self.probC = .7
        self.probM = .01
        self.bests = []
        self.averages = []
        
        # Initialize list class variables for population and roulette wheel
        self.population = [None for i in range(0, self.populationSize)]
        self.roulette_min = [0 for i in range(0, self.populationSize)]
        self.roulette_max = [0 for i in range(0, self.populationSize)]
        
        # problem definitions
    
    def buildPopulation(self):
        pass
    
    def pickChromosome(self):
        pass
    
    def reproductionLoop(self):
        pass
    
    def getBest(self):
        return 0
    
    def results(self):
        return ""


# Put Parameters Here:
# Profit for each item
p = (1, 2, 3, 4)
# Weight of each item
w = (5, 6, 7, 8)
# Weight Capacity of bag
W = 5
# Number of Knapsacks
k = 6
multiKnap = multiKnapsack(p, w, W, k)
