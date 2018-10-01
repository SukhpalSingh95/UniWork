#runs on python 3.5
#body of code reflecting working found from https://www.youtube.com/watch?v=zumC_C0C25c&t=12s
#altered using PyCharm CE

import random
import pandas as pd

popsize=15                                                                                                              #needs to be larger than number of elite individuals that are passed through to next generation
NoOfEliteIndividuals=1                                                                                                  #infinitly loops if equal to or larger than population size
tournamentselectionsize=3
RateOfMutation = 0.1                                                                                                    #probability rate that a letter does mutation
characters=['t','1','2','3','4','5','6','7','8','9','0',' ']                                                                        #biased selection based on target

#characters='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ !,123456789'                                                   #takes a lot longer to search through entire alphabet
#lengthofcharacters=len(characters)


data=[]
t=0
target = []
stageofproject=0
sectionofproject=0

##############
sectionofproject+=1
stageofproject+=0
print('------------------------------------------------------------------------------------')
print('Current stage in project: Taking in the txt file and making it workable: '+str(sectionofproject)+'.'+str(stageofproject))
print(' ')
data_file = open('/Users/sukhpalsingh/PycharmProjects/ASEWorks/Assessment2/HillClimber/smallfaultmatrix.txt', 'r')
for x in data_file:
    data.append(x.strip().split(','))
    datafilelength=len(data)
print('Length of Data file: ' + str(datafilelength))

data_file= pd.read_csv('/Users/sukhpalsingh/PycharmProjects/ASEWorks/Assessment2/Attempt2/TestMaterials/smallfaultmatrix.txt',header=None,sep= ',',index_col=0, comment='#') #https://stackoverflow.com/questions/21546739/load-data-from-txt-with-pandas + https://stackoverflow.com/questions/28200404/pandas-read-table-use-first-column-as-index
playdata=pd.DataFrame(data=data_file)


lengthofword = len(data[1])
print(' ')
print(' ')

##########################
originaltarget=[]
t = random.randrange(0, datafilelength, 1)
target.append(data[t])
print('------------------------------------------------------------------------------------')
stageofproject+=1
print('Current stage in project: '+str(sectionofproject)+'.'+str(stageofproject))
print(' ')
print('THIS IS WHAT SHALL BE SEARCHED FOR')
print('Randomly selected target: ' + str(target))
print('____________________________________________________________________________________')
print('------------------------------------------------------------------------------------')
print(' ')
print(' ')
originaltarget=target
##########################

letter=target[0]
NumberOfLetters=len(letter)
print('------------------------------------------------------------------------------------')
stageofproject+=1
print('Current stage in project: '+str(sectionofproject)+'.'+str(stageofproject))
print(' ')
print('Letter set: '+str(letter))
print('Number of letters within letter set: ' +str(NumberOfLetters))
print('____________________________________________________________________________________')
print('------------------------------------------------------------------------------------')
print(' ')
print(' ')
targetletterset=[]
targetletterset=[1,1,1,1,1,1,1,1,1,1,1]#letter

##########################

comparelist=[]
stringjoined=''.join(letter)
for count in enumerate(stringjoined):
    comparelist.append(count)
lengthofstringjoined=len(stringjoined)
print('------------------------------------------------------------------------------------')
stageofproject+=1
print('Current stage in project: '+str(sectionofproject)+'.'+str(stageofproject))
print(' ')
print('Decomposition of text line with indexes: '+str(comparelist))
print('Letters Reconstructed together without breaks: '+str(stringjoined))
print('Length of joined up string: '+str(lengthofstringjoined))
print('____________________________________________________________________________________')
print('------------------------------------------------------------------------------------')
print(' ')
print(' ')

target=['1','1','1','1','1','1','1','1','1','1','1']#stringjoined
lengthoftarget=len(target)








############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################



class individual:                                                                                                       #self can be any variable name
    def __init__(self):
        self.letters = []                                                                                               #array representing each letter
        self.fitness=0
        i=0
        while i <target.__len__():                                                                                      #while the counter is less than length of the target
            if random.random()>=0.8:                                                                                    #randomly populate array holding initial population
                self.letters.append(random.choice(data[1:]))
            else:
                self.letters.append(random.choice(data[1:3]))
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

