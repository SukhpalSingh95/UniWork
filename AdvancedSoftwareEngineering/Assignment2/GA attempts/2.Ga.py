#runs on python 3.5
#body of code reflecting working found from https://www.youtube.com/watch?v=zumC_C0C25c&t=12s
#altered using PyCharm CE
#11.10.17 influenced by https://code.activestate.com/recipes/578157-hill-climbing-template-method/ to generate the hill climbing section


import random

popsize=15                                                                                                              #needs to be larger than number of elite individuals that are passed through to next generation
NoOfEliteIndividuals=1                                                                                                  #infinitly loops if equal to or larger than population size
tournamentselectionsize=3
RateOfMutation = 0.1                                                                                                    #probability rate that a letter does mutation
characters=['t','1','2','3','4','5','6','7','8','9','0',' ']                                                                        #biased selection based on target

#characters='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ !,'                                                   #takes a lot longer to search through entire alphabet
#lengthofcharacters=len(characters)

#target = 'Hello World!'
#lengthoftarget=len(target)


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
targetletterset=letter

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

target=stringjoined
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
 #   population = genalgo.evolve(population)
    population.getindividual().sort(key=lambda x: x.getfitness(), reverse=True)
    printpopulation(population,generationnumber)
    generationnumber+=1                                                                                                 #go to next generation

