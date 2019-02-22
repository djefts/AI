import copy
import math
import random

# Andrew Forste & David Jefts
# CS455 Spring 2019
# HW2, question #3

""" TODO:
        Initial chromosome population
        Run algorithm
        Return final result
"""


class Item:
    def __init__(self, p, w):
        self.p = p
        self.w = w


class Chromosome:
    def __init__(self, num_knap_sacks, items, max_weight, chromosome = None):
        self.numBags = num_knap_sacks
        self.W = max_weight
        self.n = len(items)
        self.items = items
        self.bitsPerGene = math.ceil(math.log2(numBags)) + 1
        self.knapsacks = self.generate_knapsacks()
        self.chromosome = []
        
        # Default build
        if chromosome is None:
            gene = [0] * self.bitsPerGene
            self.chromosome = gene * self.n
        # Copy parent
        else:
            self.chromosome = chromosome[:]
    
    def crossover(self, other_chrom):
        point = random.randint(0, len(self.chromosome) - 1)
        
        # split self
        x1 = self.chromosome[0:point]
        x2 = self.chromosome[point:]
        
        # split other
        y1 = other_chrom.chromosome[0:point]
        y2 = other_chrom.chromosome[point:]
        
        # crossover by appending sublists
        child1 = Chromosome(self.numBags, self.items, self.W, x1 + y2)
        child2 = Chromosome(self.numBags, self.items, self.W, y1 + x2)
        
        # Verify the children are legal before returning
        if child1.verify_legal() and child2.verify_legal():
            return child1, child2
        else:
            return self.crossover(other_chrom)
    
    def mutate(self):
        # Find a place to mutate
        locationOfMutation = random.randint(0, len(self.chromosome) - 1)
        bitSwap = self.chromosome[locationOfMutation]
        
        # Mutate if will create legal child only
        tempChromosome = self.clone()
        if bitSwap == 0:
            tempChromosome.chromosome[locationOfMutation] = 1
            if tempChromosome.verify_legal():
                self.chromosome = tempChromosome.chromosome
            else:
                self.mutate()
        else:
            tempChromosome.chromosome[locationOfMutation] = 0
            if tempChromosome.verify_legal():
                self.chromosome = tempChromosome.chromosome
            else:
                self.mutate()
    
    def get_bag_num(self, item_num):
        gene = item_num * self.bitsPerGene
        binary = ''
        for bit in range(self.bitsPerGene):
            if self.chromosome[gene + bit] == 0:
                binary += '0'
            else:
                binary += '1'
        return int(binary, 2)
    
    # Used to determine if object is legal
    def verify_legal(self):
        for item in range(len(self.items)):
            if self.get_bag_num(item) > numBags:
                return False
        return True
    
    def set_knapsacks(self):
        """
        set the profit and weight for each knapsack at the time of calling
        each knapsack set as knapsack[n] = (curr_profit, curr_weight)
        """
        # reset knapsacks
        for b in range(len(self.knapsacks)):
            self.knapsacks[b][0] = 0  # set the bag's profit to 0
            self.knapsacks[b][1] = 0  # set the bag's weight to 0
        # set knapsacks
        for i in range(len(self.items)):
            numero = self.get_bag_num(i)
            if numero != 0:  # if the item is placed in a bag
                self.knapsacks[numero - 1][0] += self.items[i].p
                self.knapsacks[numero - 1][1] += self.items[i].w
    
    def calculate_fitness(self):
        fitness = 0
        self.set_knapsacks()
        for bag in range(len(self.knapsacks)):
            profit = self.knapsacks[bag][0]
            weight = self.knapsacks[bag][1]
            weight_ratio = 1 + ((self.W - weight) / self.W)
            fitness += profit * weight_ratio
        return fitness
    
    def generate_knapsacks(self):
        knapsacks = [[0, 0]]
        for i in range(self.numBags - 1):
            knapsacks.append([0, 0])
        return knapsacks
    
    def clone(self):
        return Chromosome(self.numBags, self.items, self.W, self.chromosome)
    
    def __str__(self):
        return str(self.chromosome)


class GA:
    def __init__(self, p, w, W, num_bags):
        self.population_size = 20
        self.num_generations = 1000
        self.probC = .7
        self.probM = .01
        self.bests = []
        self.averages = []
        
        # Initialize list class variables for population and roulette wheel
        self.population = [Chromosome] * self.population_size
        self.roulette_min = [0] * self.population_size
        self.roulette_max = [0] * self.population_size
        
        # Set the input variables
        self.profits_list = p
        self.weights_list = w
        if W <= 0:
            print("Please use a positive maximum weight value.")
            exit(1)
        self.maxWeight = W
        if num_bags <= 0:
            print("Please use a positive number of knapsacks.")
            exit(1)
        self.num_bags = num_bags
        self.items = []
        self.init_items(p, w)
        
        # problem definitions
        if num_bags < 1 or W < 1:
            print("Please supply positive weight and number of knapsacks")
            return
    
    def init_items(self, p, w):
        if len(p) != len(w):
            print("ERROR INSTANTIATING ITEMS")
            exit(1)
        for i in range(len(p)):
            self.items.append(Item(p[i], w[i]))
    
    def build_population(self):
        for i in range(self.population_size):
            self.population[i] = Chromosome(self.num_bags, self.items, self.maxWeight)
            for bit in range(len(self.population[i].chromosome)):
                rand = random.randint(0, 100)
                if rand >= 50:
                    self.population[i].chromosome[bit] = 1
                if not self.population[i].verify_legal():
                    self.population[i].chromosome[bit] = 0
            print("Populated Chromosome as:", self.population[i], self.population[i].calculate_fitness())
    
    def calc_roulette(self):
        """
        Constructs a roulette wheel for parent selection.
        """
        
        # Determine the total fitness
        sum = 0
        for chromosome in self.population:
            sum = sum + chromosome.calculate_fitness()
        
        # Generates roulette wheel where roulette_max[i] - roulette_min[i] == chromosome[i].getFitness()
        self.roulette_min[0] = 0
        for i in range(0, self.population_size):
            if i != 0:
                self.roulette_min[i] = self.roulette_max[i - 1]
            self.roulette_max[i] = self.roulette_min[i] + self.population[i].calculate_fitness() / sum
    
    def pick_chromosome(self):
        """
        Using roulette wheel, returns the index of a parent for reproduction.
        @:return index of chromosome to reproduce.
        """
        spin = random.uniform(0, 1)
        for i in range(0, self.population_size):
            print(self.roulette_min, self.roulette_max)
            if self.roulette_min[i] < spin <= self.roulette_max[i]:
                return i
        return self.population_size - 1
    
    def reproduction_loop(self):
        self.calc_roulette()
        parent1 = self.population[self.pick_chromosome()]
        parent2 = self.population[self.pick_chromosome()]
        new_population = []
        for i in range(0, self.population_size, 2):
            x = parent1.clone()
            y = parent2.clone()
            rand = random.randint(0, 100)
            if rand < self.probC * 100:
                x.crossover(y)
            rand = random.randint(0, 100)
            if rand < self.probM * 100:
                x.mutate()
                y.mutate()
            new_population.append(x)
            new_population.append(y)
        self.population = new_population
    
    def get_best(self):
        bestF = 0
        best = self.population[0]
        for pop in self.population:
            fit = pop.calculate_fitness()
            if fit > bestF:
                best = pop
                bestF = pop.calculate_fitness()
        return best
    
    def print_population(self):
        for chrom in self.population:
            print(chrom, chrom.calculate_fitness(), sep = '\t')
    
    def run(self):
        best = Chromosome
        best_overall = Chromosome
        self.build_population()
        for i in range(self.num_generations):
            self.reproduction_loop()
            best = self.get_best()
            if best.calculate_fitness() > best_overall.calculate_fitness():
                best_overall = best
        return ""
    
    def test(self):
        
        # Fitness
        self.population[0] = Chromosome(self.num_bags, self.items, W, [0, 0, 0, 0])  # no items
        self.population[1] = Chromosome(self.num_bags, self.items, W, [0, 0, 0, 1])  # 1 item
        self.population[2] = Chromosome(self.num_bags, self.items, W, [0, 1, 0, 1])  # overweight
        self.population[3] = Chromosome(self.num_bags, self.items, W, [1, 0, 0, 1])  # best
        self.population[4] = Chromosome(self.num_bags, self.items, W, [0, 1, 1, 0])  # other best
        
        print("Empty bag", self.population[0], "with fitness (0):", self.population[0].calculate_fitness())
        print("1 item bag", self.population[1], "with fitness (6):", self.population[1].calculate_fitness())
        print("Overweight bag", self.population[2], "with fitness (1.6):", self.population[2].calculate_fitness())
        print("Best V1", self.population[3], "with fitness (9):", self.population[3].calculate_fitness())
        print("Best V2", self.population[4], "with fitness (9):", self.population[4].calculate_fitness())
        
        # Mutation and crossover
        print("\nMutation:")
        aChromosome = Chromosome(numBags, self.items, W)
        aChromosome.mutate()
        print(aChromosome)
        aChromosome.mutate()
        print(aChromosome)
        aChromosome.mutate()
        print(aChromosome)
        aChromosome.mutate()
        print(aChromosome)
        chromo2 = Chromosome(numBags, self.items, W, [0, 1, 0, 1])
        chromo3 = Chromosome(numBags, self.items, W, [1, 0, 1, 0])
        print("Crossing", chromo2, "with", chromo3)
        chromo4, chromo5 = chromo2.crossover(chromo3)
        print("Chromo4: ", chromo4, chromo4.calculate_fitness())
        print("Chromo5: ", chromo5, chromo5.calculate_fitness())
        
        print("\nTrue false test:")
        chromo2 = Chromosome(numBags, self.items, W, [1, 0, 1, 0])
        print("This should say True:", chromo2.verify_legal())
        chromoIllegal = Chromosome(numBags, self.items, W, [1, 0, 1, 1])
        print("This should say False:", chromoIllegal.verify_legal(), end = '\n\n')
        
        self.build_population()
        self.reproduction_loop()
        self.print_population()


# Put Parameters Here:
# Profit for each item
p = [3, 5]
# Weight of each item
w = [5, 4]
# Weight Capacity of bag
W = 5
# Number of Knapsacks
numBags = 2
genetic = GA(p, w, W, numBags)
genetic.test()
