import copy
import math
import numpy as np

# Andrew Forste & David Jefts
# CS455 Spring 2019
# HW2, question #3

''' TODO:
        Initial chromosome population
        Run algorithm
        Return final result
'''

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
        self.bitsPerGene = math.ceil(math.log2(k))+1
        self.knapsacks = [(0, 0)] * numKnapSacks  # tuple is (currProfit, currWeight)
        self.chromosome = []
        
        #Build new
        if chromosome==None:
            gene = [0] * self.bitsPerGene
            self.chromosome = gene * self.n
            print("Created chromosome: " + str(self))
        #Copy parent
        else:
            self.chromosome=chromosome[:]

    def crossover(self,otherChrom):
        point = np.random.randint(0,len(self.chromosome))
        print("Chromo length:",len(self.chromosome),"Point:",point)

        # split self
        x1 = self.chromosome[0:point]
        x2 = self.chromosome[point:]
        
        # split other
        y1 = otherChrom.chromosome[0:point]
        y2 = otherChrom.chromosome[point:]
        
        # crossover by appending sublists
        child1 = Chromosome(self.k, self.items,self.W, x1+y2)
        child2 = Chromosome(self.k, self.items,self.W, y1+x2)
        #print("Child1:",child1, "Child2:", child2)

        #Verify the children are legal before returning
        if child1.verifyLegal() and child2.verifyLegal():   
            return child1, child2
        else:
            return self.crossover(otherChrom)

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
        for gene in range(0, len(self.chromosome), self.bitsPerGene):
            for bit in range(self.bitsPerGene):
                if self.chromosome[gene + bit] == 0:
                    binary += '0'
                else:
                    binary += '1'
            if int(binary, 2) > k:
                return False
            binary =''
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
                self.knapsacks[int(bag_num)] = (profit, weight)  # tuple is (currProfit, currWeight)
    
    def fitness(self):
        fitness = 0
        self.set_knapsacks()
        for bag in self.knapsacks:
            profit = bag[0]
            weight = bag[1]
            weight_ratio = 1 + ((self.W - weight) / self.W)
            fitness += profit * weight_ratio
        return fitness
    
    def clone(self):
        return Chromosome(self.k, self.items, self.W, self.chromosome)

    def __str__(self):
        return str(self.chromosome)


class GA:
    def __init__(self, p, w, W, k):
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
        
        # Set the input variables
        self.profitsList = p
        self.weightsList = w
        if W <= 0:
            print("Please use a positive maximum weight value.")
            exit(1)
        self.maxWeight = W
        if k <= 0:
            print("Please use a positive number of knapsacks.")
            exit(1)
        self.numBags = k
        self.items = []
        self.init_items(p, w)
        
        # problem definitions
        if k < 1 or W < 1:
            print("Please supply positive weight and number of knapsacks")
            return
        
        """ DREW'S TESTING CRAP """
        aChromosome = Chromosome(k, self.items, W)
        print("Is legal:",aChromosome.verifyLegal())
        aChromosome.mutate()
        print(aChromosome)
        aChromosome.mutate()
        print(aChromosome)
        aChromosome.mutate()
        print(aChromosome)
        aChromosome.mutate()
        print(aChromosome)

        chromo2 = Chromosome(k, self.items, W, [1,0,1,0])
        chromo3 = Chromosome(k,self.items,W,[0,0,0,0])
        print("Chromo2:",chromo2.verifyLegal())
        chromo4,chromo5=chromo2.crossover(chromo3)
        print("Chromo4: " +str(chromo4))
        print("Chromo5: " +str(chromo5))
        # for i in range(len(p)):
        #   print(p[i])
        """ DREW'S TESTING CRAP """

    def init_items(self, p, w):
        if len(p) != len(w):
            print("ERROR INSTANTIATING ITEMS")
            exit(1)
        for i in range(len(p)):
            self.items.append(Item(p[i], w[i]))
    
    def buildPopulation(self):
        for i in range(self.populationSize):
            bigboi = Chromosome(self.numBags, self.items, self.maxWeight)
            pass

    def calcRoulette(self):
        pass
    
    def pickChromosome(self):
        pass
    
    def reproductionLoop(self):
        pass
    
    def getBest(self):
        return 0
    
    def run(self):
        return ""


# Put Parameters Here:
# Profit for each item
p = [1,2]
# Weight of each item
w = [5,6]
# Weight Capacity of bag
W = 5
# Number of Knapsacks
k = 2
genetic = GA(p, w, W, k)
