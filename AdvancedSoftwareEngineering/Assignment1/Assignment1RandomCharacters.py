import random

characters=['H','e','l','o','W','r','d','!',',',' ']
lengthofcharacters=len(characters)

target = 'Hello World!'
lengthoftarget=len(target)

popsize=100

emptystoragelocation2=[]    #random letter
emptystoragelocation3=[]    #split up target
targetsplit=[]

######################################CREATING INITIAL POPULATION#######################################################

class individuals:                                                                                                      #self can be any variable name
    def __init__(self):
        self.letter = []                                                                                                 #create empty list to hold letters
        self.fitness=0                                                                                                  #initial fitness is set to 0
        counter=0
        while counter <target.__len__():                                                                                #while the counter is less than length of the target
            if random.random()>=0.5:                                                                                    #randomly populate array holding initial population
                self.letter.append(random.choice(characters))                                                           #this is where you put in  the letters
            else:
                self.letter.append(random.choice(characters))                                                            #this is where you put in  the letters
            counter+=1

    def getletter(self):                                                                                                 #get new letters
        return self.letter


    def __str__(self):
        return self.letter.__str__()                                                                                     #returns the contents of the letters array




class Population:
    def __init__(self, size):                                                                                           #takes in size of population of individuals
        self.individuals=[]
        counter=0
        while counter< size:
            self.individuals.append(individuals())                                                                      #put new individuals into an array
            counter+=1

    def getindividuals(self): return self.individuals                                                                   #returns the list of individuals



def printpopulation(pop, gennumber):                                                                                    #print population function
    print(' ')
    print('Generation:', gennumber)
    print('Target: ', target)
    print(' ')
    counter=0
    for x in pop.getindividuals():                                                                                      #goes through all the individuals in the past pop
        print("Randomly generated individual:", counter, ":", x)
        counter+=1
    print(' ')


#def getphrasetargetcomparison(self): return self.individuals


population=Population(popsize)
#population.getindividuals().sort(key=lambda x: , reverse=True)                                                         #sorts the individuals with the individuals with fitness in descending order
printpopulation(population,0)


#################################is the population holding a target matcher?############################################


def phrasetotarget(pop):
    i = 0
    for i in individuals:
        if individuals(i) == target(i):
            print('goal achieved')
    else:
        print('goal not achieved')
        i += 1


answer=0

class SplitUpTarget(Population):                                                                                        #splits target into separate words!!!
    words=target.split()
    emptystoragelocation3.append(words)
    print ("Target split into separate words: " + str(emptystoragelocation3))
    for i in emptystoragelocation3:
        targetsplit.append(list(target))
        print ('The components of the target are: '+str(targetsplit))
        #print("target length split: "+str(len(targetsplit)))
        print(' ')


class getnextgen(Population):

    def evolvepop(Population):
        return Population

generationnumber=1
while answer<1:
    if Population==targetsplit:
        print('Target found')
        answer+=1
        #print('answer', answer)
    else:
        print('Target not found, need to run again :-(')
        answer=0
        #print('answer',answer)

    Population=getnextgen.evolvepop(Population)
    printpopulation(population, generationnumber)
    generationnumber +=1

    if generationnumber>5000:
        print('Doubt the target will get found, you\'re at generation', generationnumber-1, ', try running it again if you like until you\'re satisfied. \nOr you could go to the Genetric Algorithm version :-)')
        exit()
