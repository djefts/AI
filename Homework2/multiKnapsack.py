import copy
import math

#Andrew Forste & David Jefts
#CS455 Spring 2019
#HW2, question #3

''' TODO:
        Initial chromosome population
        Crosover
        Mutuation
        Return final result
'''
class multiKnapsack:
    def __init__(self, p,w,W,k):
        if len(p)!=len(w):
            print("Please ensure number of weights and profit match")
            return
        if k<1 or W <1:
            print("Please supply postitive weight and number of knapsacks")
            return

        items = []
        for i in range(len(p)):
            items.append(Item(p,w))

        aChromosome = Chromosome(k,items, W)
        
        #for i in range(len(p)):
         #   print(p[i])

class Item:
    def __init__(self, p, w):
        self.p = p
        self.w = w

class Chromosome:
    def __init__(self, numKnapSacks, items, maxWeight):
        self.k = numKnapSacks
        self.W = maxWeight
        self.n = len(items)
        self.items = items
        self.bitsPerGene = math.ceil(math.log2(k))
        
        gene=[0]*self.bitsPerGene
        self.chromosome=gene*self.n
        print("Created chromosome: "+str(self))
        print("Chromosome Legal: "+str(self.verifyLegal()))

    def verifyLegal(self):
        binary=''
        for gene in range(0,self.n,self.bitsPerGene):
            for bit in range(self.bitsPerGene):
                if(self.chromosome[gene+bit]==0):
                    binary+='0'
                else:
                    binary += '1'
            if(int(binary,2)>k):
                return False
        return True

    def fitness(self):
        fitness = 0
        profit = 0
        weight = 0
        for b in range(0, self.k, self.bitsPerGene):
            bag = ""
            for a in range(self.bitsPerGene):
                bag += self.chromosome[a]
            if bag not = "000":
                pass
            for item in self.items:
                profit += item.p
            for item in self.items:
                weight += item.w
            weight_ratio = 1+((self.W-weight)/self.W)
            fitness += profit*weight_ratio
        return fitness

    def __str__(self):
        return str(self.chromosome)

class GA:
    def __init__(self, k, n, items):
        self.populationSize = 100
        self.numGenerations = 1000
        self.probC = .7
        self.probM = .01
        self.bests = []
        self.averages = []
        
        # Initialize list class variables for population and roulette wheel
        self.population = [None for i in range(0,self.populationSize)]
        self.roulette_min = [0 for i in range(0,self.populationSize)]
        self.roulette_max = [0 for i in range(0,self.populationSize)]

        #problem definitions
        self.k = k

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
p = (1,2,3,4)
# Weight of each item
w = (5,6,7,8)
# Weight Capacity of bag
W = 5
# Number of Knapsacks
k = 6
multiKnap = multiKnapsack(p,w,W,k)