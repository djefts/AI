import copy
import math
import random


# Andrew Forste & David Jefts
# CS455 Spring 2019
# HW2, question #3

# ===================FOR RUNTIME USE, SEE BOTTOM OF FILE


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
            # print("\tCrossed",self.chromosome,"with",other_chrom.chromosome,"and got",child1.chromosome, "and",
            # child2.chromosome)
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
                # print("\t\tMutating",self.chromosome,"to",tempChromosome.chromosome)
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
            weight_ratio = 1 - abs(((self.W - weight) / self.W))
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
        # Population parameters
        self.population_size = 20  # this number must be even or reproduction causes errors
        self.num_generations = 30
        self.probC = .7
        self.probM = .1
        
        # Initialize list class variables for population and roulette wheel
        self.population = []
        self.roulette_min = [0] * self.population_size
        self.roulette_max = [0] * self.population_size
        
        # Set the input variables
        self.profits_list = p
        self.weights_list = w
        if W <= 0:
            print("Please use a positive maximum weight value.")
            exit(1)
        self.max_weight = W
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
        for i in range(0, self.population_size):
            self.population.append(Chromosome(self.num_bags, self.items, self.max_weight))
            for bit in range(len(self.population[i].chromosome)):
                rand = random.randint(0, 100)
                if rand >= 50:
                    self.population[i].chromosome[bit] = 1
                if not self.population[i].verify_legal():
                    self.population[i].chromosome[bit] = 0
            # print("Populated Chromosome as:", self.population[i], self.population[i].calculate_fitness())
    
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
            if self.roulette_min[i] < spin <= self.roulette_max[i]:
                return i
        return self.population_size - 1
    
    def reproduction_loop(self):
        self.calc_roulette()
        new_population = []
        for i in range(0, self.population_size, 2):
            parent1 = self.population[self.pick_chromosome()]
            parent2 = self.population[self.pick_chromosome()]
            x = parent1.clone()
            y = parent2.clone()
            rand = random.uniform(0, 1)
            # crossover
            if rand <= self.probC:
                x, y = x.crossover(y)
            rand = random.uniform(0, 1)
            # mutate
            if rand <= self.probM:
                x.mutate()
                y.mutate()
            new_population.append(x)
            new_population.append(y)
        self.population = new_population
    
    def get_average(self):
        sum = 0
        for pop in self.population:
            sum += pop.calculate_fitness()
        return sum / self.population_size
    
    def get_best(self):
        bestF = 0
        best = self.population[0]
        for pop in self.population:
            fit = pop.calculate_fitness()
            if fit > bestF:
                best = pop
                bestF = fit
        return best
    
    def run(self):
        best = Chromosome(self.num_bags, self.items, self.max_weight)
        best_overall = Chromosome(self.num_bags, self.items, self.max_weight)
        self.build_population()
        for i in range(self.num_generations):
            print("\nGeneration", i + 1)
            self.reproduction_loop()
            best = self.get_best().clone()
            if best.calculate_fitness() > best_overall.calculate_fitness():
                best_overall = best.clone()
            print("\tBest chromosome:", best)
            print("\tBest fitness:", best.calculate_fitness())
            print("\tAverage:", self.get_average())
        self.print_results(best_overall)
    
    def print_results(self, best):
        print()
        
        print("=====FINAL RESULTS=====")
        for i in range(len(self.items)):
            if best.get_bag_num(i) == 0:
                print("Place item", i + 1, "in no bag")
            else:
                print("Place item", i + 1, "in bag", best.get_bag_num(i))
        
        for i in range(self.num_bags):
            print("Bag", i + 1, "with weight:", best.knapsacks[i][1], "and profit:", best.knapsacks[i][0])
        
        print("Final best fitness:", best.calculate_fitness())


# Put Parameters Here:
# Profit for each item
p = [3, 5, 1, 6]
# Weight of each item
w = [5, 4, 7, 3]
# Weight Capacity of bag
W = 10
# Number of Knapsacks
numBags = 2
# To change mutation, crossover, pop size, and generations, see line 138

# Alogrithm runs
genetic = GA(p, w, W, numBags)
genetic.run()
