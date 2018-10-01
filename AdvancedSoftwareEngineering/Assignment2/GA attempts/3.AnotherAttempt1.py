import random
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')

#DataSetIn= pd.read_csv('/Users/sukhpalsingh/PycharmProjects/ASEWorks/Assessment2/Attempt2/TestMaterials/smallfaultmatrix.txt',header=None,sep= ',', comment='#') #https://stackoverflow.com/questions/21546739/load-data-from-txt-with-pandas + https://stackoverflow.com/questions/28200404/pandas-read-table-use-first-column-as-index
DataSetIn=open('/Users/sukhpalsingh/PycharmProjects/ASEWorks/Assessment2/Attempt2/TestMaterials/smallfaultmatrix.txt','r')
characters=DataSetIn.read().splitlines()

target=np.empty(len(characters)); target.fill(int(1))
#print(target)
maxvalue=len(target)
RowSize=4
popsize=10                                                                                                              #needs to be larger than number of elite individuals that are passed through to next generation
lengthofcharacters=len(characters)

#target=np.empty(lengthofcharacters-1); target.fill(int(1))
target=['1','1','1','1','1','1','1','1','1']
lengthoftarget=len(target)


class individual:                                                                                                       #self can be any variable name
    def __init__(self):
        self.letters = []                                                                                               #array representing each letter
        self.fitness=0
        i=0
        while i <RowSize:                                                                                      #while the counter is less than length of the target
            if random.random()>=0.1:                                                                                    #randomly populate array holding initial population
                self.letters.append(random.choice(characters))
            else:
                self.letters.append(random.choice(characters))
            i+=1



    #def getletters(self):
      #  return self.letters

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
        self.individuals=[]   #new array to hold new individuals
        #self.LetterIndividual = []
        #self.TargetComparison = []
        #self.LettersListed = self.individuals

        i=0
        while i< size:
            self.individuals.append(individual())                                                                        #put new individuals into an array
            #self.LetterIndividual = [item[0] for item in enumerate(self.individuals)]
            #self.TargetComparison = [item[1:] for item in enumerate(self.individuals)]
            #hellow.append(individual())
            i+=1
        # EDITED ON THE 14.10.17

        #for i in self.LettersListed:
            #print(i)
        #    self.LetterIndividual = [item[0:3] for item in i]
        #    self.TargetComparison = [item[5:] for item in i]
        #print(self.LetterIndividual)
        #print(self.TargetComparison)


    def getindividual(self): return self.individuals                                                               #returns the list of individuals in population


########################


def printpopulation(pop, gennumber):                                                                                    #print population function

    #print('\n-------------------------------------------------------')
    #print( 'Generation number', gennumber, ', Fitness of highest ranked individual in population: ', pop.getindividual()[0].getfitness())
    #print('Target Individual: ', target, 'Highest possible fitness value: ', lengthoftarget)
    #print("-------------------------------------------------------")
    i=0
    for x in pop.getindividual():                       #goes through all the individual in the past pop
        print("Randomly generated individual number:", i, ":", x, "Associated fitness score: ", x.getfitness())
        i +=1
#        print(LetterIndividual)
#        print(TargetComparison)

#def MessingwithGetIndividual(pop):
   # LetterIndividual = []
  #  TargetComparison = []
#    for x in pop.getindividual()[0].getfitness():
#        print(x)

#    for i in self.LettersListed:
#        print(i)
#        LetterIndividual = [item[0:3] for item in i]
#        TargetComparison = [item[5:] for item in i]
#        print(LetterIndividual)
#        print(TargetComparison)
hellow=[]

population=Population(popsize)
population.getindividual().sort(key=lambda x: x.getfitness(), reverse=True)                                             #sorts the chromosomes with the individual with fitness in descending order
printpopulation(population,0)
#MessingwithGetIndividual(population)

#parameter1=individual()
#parameter1.__init__()

#.sum(axis=0)

#for i in self.LettersListed:
# print(i)
#    self.LetterIndividual = [item[0:3] for item in i]
#    self.TargetComparison = [item[5:] for item in i]
#print(self.LetterIndividual)
#print(self.TargetComparison)
