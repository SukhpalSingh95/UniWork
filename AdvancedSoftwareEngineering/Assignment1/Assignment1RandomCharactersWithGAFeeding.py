#runs on python 3.5
#body of code reflecting working found from https://www.youtube.com/watch?v=zumC_C0C25c&t=12s
#altered using PyCharm CE

import random

popsize=15                                                                                                              #needs to be larger than number of elite individuals that are passed through to next generation
NoOfEliteIndividuals=1                                                                                                  #infinitly loops if equal to or larger than population size
tournamentselectionsize=3
RateOfMutation = 0.1                                                                                                    #probability rate that a letter does mutation
characters=['H','e','l','o','W','r','d','!',' ']                                                                        #biased selection based on target

#characters='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ !,'                                                   #takes a lot longer to search through entire alphabet
lengthofcharacters=len(characters)

target = 'Hello World!'
lengthoftarget=len(target)



class individual:                                                                                                       #self can be any variable name
    def __init__(self):
        self.letters = []                                                                                               #array representing each letter
        self.fitness=0
        i=0
        while i <target.__len__():                                                                                      #while the counter is less than length of the target
            if random.random()>=0.1:                                                                                    #randomly populate array holding initial population
                self.letters.append(random.choice(characters))
            else:
                self.letters.append(random.choice(characters))
            i+=1

    def getletters(self):
        return self.letters

    def getfitness(self):
        self.fitness=0
        for i in range(self.letters.__len__()):                                                                         # for loop over all letters
            if self.letters[i] == target[i]:
                self.fitness+=1                                                                                         #add 1 to the fitness if the letters is the same as the target
        return self.fitness

    def __str__(self):
        return self.letters.__str__()                                                                                   #returns the contents of the letters array


class Population:
    def __init__(self, size):                                                                                           #takes in size of population of Individuals
        self.individual=[]                                                                                              #new array to hold new individuals
        i=0
        while i< size:
            self.individual.append(individual())                                                                        #put new individuals into an array
            i+=1

    def getindividual(self): return self.individual                                                                     #returns the list of individuals in population


class genalgo:
    @staticmethod                                                                                                       #public, static evolve method?
    def evolve(pop):
        return genalgo.mutatepop(genalgo.crossoverpopulation(pop))

    @staticmethod
    def crossoverpopulation(pop):
        crossoverpop=Population(0)
        for i in range(NoOfEliteIndividuals):
            crossoverpop.getindividual().append(pop.getindividual()[i])                                                 #leaves the elite individuals as is, no mutation
        i=NoOfEliteIndividuals
        while i<popsize:
            individual1 = genalgo.tournamentpopselect(pop).getindividual()[0]
            individual2 = genalgo.tournamentpopselect(pop).getindividual()[0]
            crossoverpop.getindividual().append(genalgo.crossoverindividual(individual1, individual2))
            i+=1
        return crossoverpop


    @staticmethod
    def mutatepop(pop):
        for i in range(NoOfEliteIndividuals,popsize):
            genalgo.mutateindividual(pop.getindividual()[i])
        return pop

    @staticmethod
    def crossoverindividual(individual1,individual2):
        crossoverindividual=individual()
        for i in range(target.__len__()):
            if random.random()>=0.5:
                crossoverindividual.getletters()[i] = individual1.getletters()[i]
            else:
                crossoverindividual.getletters()[i] = individual2.getletters()[i]
        return crossoverindividual

    @staticmethod
    def mutateindividual(individual):                                                                                   #for loop over each passed letter
        for i in range(target.__len__()):
            if random.random()<RateOfMutation:                                                                          #if random number is smaller than the mutation rate, then mutate
                if random.random()<0.5:
                    individual.getletters()[i]=random.choice(characters)
                else:
                    individual.getletters()[i]=random.choice(characters)

    @staticmethod
    def tournamentpopselect(pop):                                                                                       #randomly chooses individuals from past and present population
        tournamentpop=Population(0)
        i=0
        while i < tournamentselectionsize:
            tournamentpop.getindividual().append(pop.getindividual()[random.randrange(0,popsize)])
            i+=1
        tournamentpop.getindividual().sort(key=lambda x: x.getfitness(), reverse=True)
        return tournamentpop


def printpopulation(pop, gennumber):                                                                                    #print population function
    print('\n-------------------------------------------------------')
    print( 'Generation number', gennumber, ', Fitness of highest ranked individual in population: ', pop.getindividual()[0].getfitness())
    print('Target Individual: ', target, 'Highest possible fitness value: ', lengthoftarget)
    print("-------------------------------------------------------")
    i=0
    for x in pop.getindividual():                                                                                       #goes through all the individual in the past pop
        print("Randomly generated individual number:", i, ":", x, "Associated fitness score: ", x.getfitness())
        i +=1


population=Population(popsize)
population.getindividual().sort(key=lambda x: x.getfitness(), reverse=True)                                             #sorts the chromosomes with the individual with fitness in descending order
printpopulation(population,0)

generationnumber = 1
while population.getindividual()[0].getfitness()<target.__len__():                                                      #whilst the fitness of each individual in each population is less than target length
    population = genalgo.evolve(population)
    population.getindividual().sort(key=lambda x: x.getfitness(), reverse=True)
    printpopulation(population,generationnumber)
    generationnumber+=1                                                                                                 #go to next generation

