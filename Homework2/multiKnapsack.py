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
        self.numBags = numKnapSacks
        self.W = maxWeight
        self.n = len(items)
        self.items = items
        self.bitsPerGene = math.ceil(math.log2(numBags))+1
        self.knapsacks = [[0, 0]] * numKnapSacks  # tuple is (currProfit, currWeight)
        self.chromosome = []
        
        # Default build
        if chromosome==None:
            gene = [0] * self.bitsPerGene
            self.chromosome = gene * self.n
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
        child1 = Chromosome(self.numBags, self.items,self.W, x1+y2)
        child2 = Chromosome(self.numBags, self.items,self.W, y1+x2)
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

    def getBagNum(self, item_num):
        gene=item_num*self.bitsPerGene
        binary = ''
        for bit in range(self.bitsPerGene):
            if self.chromosome[gene + bit] == 0:
                binary += '0'
            else:
                binary += '1'
        return int(binary,2)
    
    #Used to determine if object is legal
    def verifyLegal(self):
        for item in range(len(self.items)):
            if self.getBagNum(item) > numBags:
                return False
        return True
    
    def set_knapsacks(self):
        """
        set the profit and weight for each knapsack at the time of calling
        each knapsack set as knapsack[n] = (curr_profit, curr_weight)
        """
        #reset knapsacks
        for b in range(len(self.knapsacks)):
                self.knapsacks[b][0] = 0   # set the bag's profit to 0
                self.knapsacks[b][1] = 0   # set the bag's weight to 0
        #set knapsacks
        for item in self.items:
            numero = self.getBagNum(i)
            if numero != 0:  # if the item is placed in a bag
                self.knapsacks[numero-1][0] += self.items[i].p
                self.knapsacks[numero-1][1] += self.items[i].w
    
    def calculateFitness(self):
        calculateFitness = 0
        self.set_knapsacks()
        for bag in range(len(self.knapsacks)):
            profit = self.knapsacks[bag][0]
            weight = self.knapsacks[bag][1]
            print("bag",bag,"has profit", profit, "and weight:",weight)
            weight_ratio = 1 + ((self.W - weight) / self.W)
            calculateFitness += profit * weight_ratio
        return calculateFitness
    
    def clone(self):
        return Chromosome(self.numBags, self.items, self.W, self.chromosome)

    def __str__(self):
        return str(self.chromosome)


class GA:
    def __init__(self, p, w, W, numBags):
        self.populationSize = 20
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
        if numBags <= 0:
            print("Please use a positive number of knapsacks.")
            exit(1)
        self.numBags = numBags
        self.items = []
        self.init_items(p, w)
        
        # problem definitions
        if numBags < 1 or W < 1:
            print("Please supply positive weight and number of knapsacks")
            return
        
        """ DREW'S TESTING CRAP """
        #self.buildPopulation()

        """ aChromosome = Chromosome(numBags, self.items, W)
        print("Is legal:",aChromosome.verifyLegal())
        aChromosome.mutate()
        print(aChromosome)
        aChromosome.mutate()
        print(aChromosome)
        aChromosome.mutate()
        print(aChromosome)
        aChromosome.mutate()
        print(aChromosome)

        chromo3 = Chromosome(numBags,self.items,W,[0,0,0,0])
        chromo2 = Chromosome(numBags, self.items, W, [1,0,1,0])
        chromo3 = Chromosome(numBags,self.items,W,[0,0,0,0])
        print("Chromo2:",chromo2.verifyLegal())
        chromo4,chromo5=chromo2.crossover(chromo3)
        print("Chromo4: ", chromo4, chromo4.calculateFitness())
        print("Chromo5: ", chromo5, chromo5.calculateFitness()) """
        #test cases
        self.test()
        
        #True false test
        chromo2 = Chromosome(numBags, self.items, W, [1,0,1,0])
        print("This should say True:",chromo2.verifyLegal())
        chromoIllegal = Chromosome(numBags, self.items,W,[1,0,1,1])
        print("This should say False:",chromoIllegal.verifyLegal())
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
            self.population[i] = Chromosome(self.numBags, self.items, self.maxWeight)
            for bit in range(len(self.population[i].chromosome)):
                rand = np.random.randint(0,100)
                if rand >= 50:
                    self.population[i].chromosome[bit] = 1
                if not self.population[i].verifyLegal():
                    self.population[i].chromosome[bit] = 0
            print("Populated Chromosome as:",self.population[i],self.population[i].calculateFitness())

    def calcRoulette(self):
        pass
    
    def pickChromosome(self):
        pass
    
    def reproductionLoop(self):
        pass
    
    def getBest(self):
        bestF = 0
        best = self.population[0]
        for pop in self.population:
            fit = pop.calculateFitness()
            if fit > bestF:
                best = pop
                bestF = pop.calculateFitness()
        return best
    
    def run(self):
        return ""

    def test(self):
        self.population[0] = Chromosome(self.numBags, self.items, W, [0,0,0,0]) # no items
        self.population[1] = Chromosome(self.numBags, self.items, W, [0,0,0,1]) # 1 item
        self.population[2] = Chromosome(self.numBags, self.items, W, [0,1,0,1]) # overweight
        self.population[3] = Chromosome(self.numBags, self.items, W, [1,0,0,1]) # best
        self.population[4] = Chromosome(self.numBags, self.items, W, [0,1,1,0]) # other best

        print("Empty bag",self.population[0],"with fitness (0):", self.population[0].calculateFitness())
        print("1 item bag",self.population[1],"with fitness (6):", self.population[1].calculateFitness())
        print("Overweight bag",self.population[2],"with fitness (1.6):", self.population[2].calculateFitness())
        print("Best V1",self.population[3],"with fitness (10,2)", self.population[3].calculateFitness())
        print("Best V2",self.population[4],"with fitness (10.2)", self.population[4].calculateFitness())

# Put Parameters Here:
# Profit for each item
p = [3,5]
# Weight of each item
w = [5,4]
# Weight Capacity of bag
W = 5
# Number of Knapsacks
numBags = 2
genetic = GA(p, w, W, numBags)
