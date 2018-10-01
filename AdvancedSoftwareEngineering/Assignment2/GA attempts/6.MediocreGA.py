import random as rm
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')

#target =['1','1','1','1','1','1','1','1','1']   #represents the ideal solution
#target =np.zeros(1,len())


print(' ')
#DataSetIn= pd.read_csv('/Users/sukhpalsingh/PycharmProjects/ASEWorks/Assessment2/Attempt2/TestMaterials/smallfaultmatrix.txt',header=None,sep= ',',index_col=0, comment='#') #https://stackoverflow.com/questions/21546739/load-data-from-txt-with-pandas + https://stackoverflow.com/questions/28200404/pandas-read-table-use-first-column-as-index
#DataSetIn = pd.read_csv('/Users/sukhpalsingh/PycharmProjects/ASEWorks/Assessment2/Attempt2/TestMaterials/bigfaultmatrix.txt', header=None,sep=',', index_col=0,comment='#')  # https://stackoverflow.com/questions/21546739/load-data-from-txt-with-pandas + https://stackoverflow.com/questions/28200404/pandas-read-table-use-first-column-as-index
DataSetIn= pd.read_csv('/Users/sukhpalsingh/PycharmProjects/ASEWorks/Assessment2/Attempt2/TestMaterials/smallfaultmatrix.txt',header=None,sep= ',', comment='#') #https://stackoverflow.com/questions/21546739/load-data-from-txt-with-pandas + https://stackoverflow.com/questions/28200404/pandas-read-table-use-first-column-as-index
print('Number of Columns in data set: '+str(len(DataSetIn.columns)))
print('Number of rows in data set: '+str(len(DataSetIn)))
print(' ')

target =[]
for i in range(1,len(DataSetIn.columns)):
    target.append(1)

print('Target: '+str(target))
maxvalue=len(target)
print(' ')
print('Maxvalue score of target: '+str(maxvalue))


PopulationSize=3
popsize=PopulationSize
PPOPULATIONSIZE=10

NoOfEliteIndividuals=1                                                                                                  #infinitly loops if equal to or larger than population size
tournamentselectionsize=3
RateOfMutation = 0.1                                                                                                    #probability rate that a letter does mutation

characters=[]



GenerateAPopulation=[]
class InitialisingDataSet:                                                                                                                          #CAN IGNORE THIS FOR THE MOST PART
    def __init__(self):
        self.DataFrameInitial = DataSetIn
        self.DataFrameLength = len(self.DataFrameInitial)
        self.DataFrameColumns = len(self.DataFrameInitial)

        self.PlayData = pd.DataFrame(data=self.DataFrameInitial)


        self.DataFrameIn=self.PlayData                                                                                  #doing it this way reduces dependence on data frame names
        e = self.DataFrameIn.sum(axis=1)                                                                                # https://stackoverflow.com/questions/42886354/pandas-count-values-by-condition-for-row
        #t = pd.Series(e)
        self.DataFrameInModPID = self.DataFrameIn.assign(Score=e)                                                       # https://chrisalbon.com/python/pandas_assign_new_column_dataframe.html

        self.DataFrameInModSOPID = self.DataFrameInModPID.sort_values(self.DataFrameInModPID.columns[-1], ascending=False)


class SearchPopulationLinearly(InitialisingDataSet):                                                                                #NOT ACTUALLY LINEARLY SEARCHING, JUST CREATING INITIAL POPULATION


    def __init__(self):

        self.DataFrameInitial = DataSetIn
        self.DataFrameLength = len(self.DataFrameInitial)
        self.DataFrameColumns = len(self.DataFrameInitial)

        self.PlayData = pd.DataFrame(data=self.DataFrameInitial)

        self.DataFrameIn = self.PlayData  # doing it this way reduces dependence on data frame names
        e = self.DataFrameIn.sum(axis=1)  # https://stackoverflow.com/questions/42886354/pandas-count-values-by-condition-for-row
        # t = pd.Series(e)

        self.DataFrameInModPID = self.DataFrameIn.assign(Score=e)  # https://chrisalbon.com/python/pandas_assign_new_column_dataframe.html
        self.DataFrameInModSOPID = self.DataFrameInModPID.sort_values(self.DataFrameInModPID.columns[-1],
                                                                      ascending=False)

        #FROM SPLITTING DATA UP INTO 2
        self.DataFrameInModSDFIT1=pd.DataFrame()    #CREATES EMPTY DATAFRAME
        self.DataFrameInModSDFIT2=pd.DataFrame()    #CREATES EMPTY DATAFRAME
        self.DataFrameInModSDFIT3=self.DataFrameInModSOPID  #CREATES A DATAFRAME FROM THE DATAFRAME WITH SCORES
        self.SDFIT3Length=len(self.DataFrameInModSDFIT3)
        self.midpoint=(self.SDFIT3Length/2)
        #print(self.midpoint)

        self.DataFrameInModSDFIT1 = self.DataFrameInModSDFIT3[:int(self.midpoint)]
        self.DataFrameInModSDFIT2 = self.DataFrameInModSDFIT3[int(self.midpoint):]



        self.DataFrameInModSDFIT1_1=pd.DataFrame(self.DataFrameInModSDFIT1.drop('Score',axis=1))
        self.DataFrameInModSDFIT2_1=pd.DataFrame(self.DataFrameInModSDFIT2.drop('Score',axis=1))


        #FROM ADDING ROWS TO EACH OTHER
        self.GenerateAPopulation = []
        #self.GenerateAPopulation=pd.DataFrame
        self.DataFrameInARTEO1 = pd.DataFrame()
        t1=self.DataFrameInModSDFIT1_1.values.tolist()                                                                           #TURNS INTO AN ARRAY
        t2 = self.DataFrameInModSDFIT2_1.values.tolist()
        #t1=pd.DataFrame()
        #t2=pd.DataFrame()

        i = 0
        while i < PopulationSize:
            if i < rm.randint(0, self.DataFrameLength):
                self.GenerateAPopulation.append(rm.choice(t1))
                #t1=rm.sample(self.DataFrameInModSDFIT1_1)
            else:
                self.GenerateAPopulation.append(rm.choice(t2))
                #t2=rm.sample(self.DataFrameInModSDFIT2_1)
            i += 1
        self.GenerateAPopulation=pd.DataFrame(self.GenerateAPopulation)
        GenerateAPopulation.append(self.GenerateAPopulation)
        #print(self.GenerateAPopulation)


    def getletters(self):
        return self.GenerateAPopulation



    def __str__(self):
        return self.GenerateAPopulation.__str__()                                                                       #returns the contents of the letters array

    def getfitness(self):
        # FROM COMBING ROWS TOGETHER

        #print('Summed over row' + str(self.SummedOverRow))
        #print(self.t5[0][0])
        #self.SeparatedRowString = self.SummedOverRow[0][0]  # THIS PUTS THE ARRAYS TOGETHER!!!!!!! by summing

        self.fitness=0
        self.Hellllloo=[]
        for i in range(self.GenerateAPopulation.__len__()):                                                                         # for loop over all letter
        #for i in enumerate(self.GenerateAPopulation):
            #self.SummedOverRow = np.sum(self.GenerateAPopulation, axis=1)           #Works, but the comparison doesnt work the way i want it too
            self.SummedOverRow = self.GenerateAPopulation.sum(axis=1)  # Works, but the comparison doesnt work the way i want it too  #self.SummedOverRow.drop(self.SummedOverRow[[0]], axis=1)  #self.SummedOverRow.drop(self.SummedOverRow[[0]], axis=1)
            #print(self.SummedOverRow)
            #print(self.SummedOverRow)
            #self.NewSummedOverRow=np.sum(self.SummedOverRow,axis=1)
            if self.SummedOverRow[i]== target[i]:
                #print(self.SummedOverRow)
                #print(i)
                self.fitness+=1                         #add 1 to the fitness if the letters is the same as the target
        self.Hellllloo.append(self.SummedOverRow)
#######NEED TO ADD SOMETHING ELSE HERE TOO, NOT SURE HOW TO MAKE THE BEST ONES WIN!!!!!
            #print(self.SummedOverRow)
        return self.fitness
        return self.Hellllloo





class CreatingThePopulation(SearchPopulationLinearly):
    def __init__(self):

        self.CreatingRows = []  # holds all the new rows thatre getting made
        v = 0
        while v < PPOPULATIONSIZE:
            self.CreatingRows.append(SearchPopulationLinearly())
            v += 1

    def GetTheRowPopulation(self): return self.CreatingRows






class genalgo:
    @staticmethod  # public, static evolve method?
    def evolve(pop):
        return genalgo.mutatepop(genalgo.crossoverpopulation(pop))

    @staticmethod
    def crossoverpopulation(pop):
        crossoverpop = CreatingThePopulation()
        for i in range(NoOfEliteIndividuals):
            crossoverpop.GetTheRowPopulation().append(
                pop.GetTheRowPopulation()[i])  # leaves the elite individuals as is, no mutation
        i = NoOfEliteIndividuals
        while i < popsize:
            individual1 = genalgo.tournamentpopselect(pop).GetTheRowPopulation()[0]
            individual2 = genalgo.tournamentpopselect(pop).GetTheRowPopulation()[0]
            crossoverpop.GetTheRowPopulation().append(genalgo.crossoverindividual(individual1, individual2))
            i += 1
        return crossoverpop

    @staticmethod
    def mutatepop(pop):
        for i in range(NoOfEliteIndividuals, popsize):
            genalgo.mutateindividual(pop.GetTheRowPopulation()[i])
        return pop

    @staticmethod
    def crossoverindividual(individual1, individual2):
        crossoverindividual = SearchPopulationLinearly()
        for i in range(target.__len__()):
            if rm.random() >= 0.9:
                crossoverindividual.getletters()[i] = individual1.getletters()[i]
            else:
                crossoverindividual.getletters()[i] = individual2.getletters()[i]
        return crossoverindividual

    @staticmethod
    def mutateindividual(SearchPopulationLinearly):  # for loop over each passed letter
        for i in range(target.__len__()):
            if rm.random() < RateOfMutation:  # if random number is smaller than the mutation rate, then mutate
                if rm.random() < 0.2:
                    SearchPopulationLinearly.getletters()[i].append(rm.choice(GenerateAPopulation))
                else:
                    SearchPopulationLinearly.getletters()[i].append(rm.choice(GenerateAPopulation))

    @staticmethod
    def tournamentpopselect(pop):  # randomly chooses individuals from past and present population
        tournamentpop = CreatingThePopulation()
        i = 0
        while i < tournamentselectionsize:
            tournamentpop.GetTheRowPopulation().append(pop.GetTheRowPopulation()[rm.randrange(0, popsize)])
            i += 1
        tournamentpop.GetTheRowPopulation().sort(key=lambda x: x.getfitness(), reverse=True)
        return tournamentpop







def printpopulation(pop, gennumber):                                                                                    #print population function
    Helloww = []
    print("-------------------------------------------------------")
    #print('\n-------------------------------------------------------')
    print('Generation number', gennumber, ', Fitness of highest ranked individual in population: ',pop.GetTheRowPopulation()[0].getfitness())
    print("-------------------------------------------------------")
    i=0
    for x in pop.GetTheRowPopulation():                       #goes through all the individual in the past pop
        print("Randomly generated individual group of tests number:", i, ":", x, "Associated fitness score: ", x.getfitness())
        PopulationsAscribed.append(x.__str__())
        print('')
        #Helloww = pd.DataFrame(PopulationsAscribed)
        i +=1




PopulationsAscribed=[]
PROJECT2=CreatingThePopulation()
PROJECT2.GetTheRowPopulation()
printpopulation(PROJECT2,0)
#PROJECT1=SearchPopulationLinearly()
#PROJECT1.getfitness()


generationnumber = 1
while PROJECT2.GetTheRowPopulation()[0].getfitness()<target.__len__():                                                      #whilst the fitness of each individual in each population is less than target length
    PROJECT3 = genalgo.evolve(PROJECT2)
    PROJECT3.GetTheRowPopulation().sort(key=lambda x: x.getfitness(), reverse=True)
    printpopulation(PROJECT3,generationnumber)
    generationnumber+=1                                                                                                 #go to next generation
