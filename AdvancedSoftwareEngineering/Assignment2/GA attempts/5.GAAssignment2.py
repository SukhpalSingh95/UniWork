import random as rm
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')


print(' ')
DataSetIn= pd.read_csv('/Users/sukhpalsingh/PycharmProjects/ASEWorks/Assessment2/Attempt2/TestMaterials/smallfaultmatrix.txt',header=None,sep= ',',index_col=0, comment='#') #https://stackoverflow.com/questions/21546739/load-data-from-txt-with-pandas + https://stackoverflow.com/questions/28200404/pandas-read-table-use-first-column-as-index
#DataSetIn = pd.read_csv('/Users/sukhpalsingh/PycharmProjects/ASEWorks/Assessment2/Attempt2/TestMaterials/bigfaultmatrix.txt', header=None,sep=',', index_col=0,comment='#')  # https://stackoverflow.com/questions/21546739/load-data-from-txt-with-pandas + https://stackoverflow.com/questions/28200404/pandas-read-table-use-first-column-as-index
print('Number of Columns in data set: '+str(len(DataSetIn.columns)))
print('Number of rows in data set: '+str(len(DataSetIn)))
print(' ')
DataSetIn2= pd.read_csv('/Users/sukhpalsingh/PycharmProjects/ASEWorks/Assessment2/Attempt2/TestMaterials/smallfaultmatrix.txt',header=None,sep= ',', comment='#') #https://stackoverflow.com/questions/21546739/load-data-from-txt-with-pandas + https://stackoverflow.com/questions/28200404/pandas-read-table-use-first-column-as-index

DataFrameStart=pd.DataFrame(DataSetIn2)

#target=np.empty(len(DataSetIn.columns)); target.fill(int(1))
target =['1','1','1','1','1','1','1','1','1']   #represents the ideal solution, this is not what it needs to be though!!!
target =[1,1,1,1,1,1,1,1,1]   #represents the ideal solution, this is not what it needs to be though!!!

print('Target: '+str(target))
maxvalue=len(target)
print(' ')
print('Maxvalue score of target: '+str(maxvalue))

RowPopulationVal=4
PopulationPool=5

class InitialisingDataSet:

    def __init__(self):
        self.DataFrameInitial=DataSetIn
        self.DataFrameLength=len(self.DataFrameInitial)

        self.DataFrameColumns=len(self.DataFrameInitial)
        self.PlayData = pd.DataFrame(data=self.DataFrameInitial)
        #self.PlayData.columns = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']



    def PrintInputData(self):
        print('The input data')
        print(self.PlayData[0:self.DataFrameColumns].head())
        #print(self.PlayData)
        print(' ')
        print(' ')


    def PrepareInputData(self):
        #sdf = playdata.isin(target)            #This hasnt been checked!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.DataFrameIn=self.PlayData                       #doing it this way reduces dependence on data frame names
        e = self.DataFrameIn.sum(axis=1)  # https://stackoverflow.com/questions/42886354/pandas-count-values-by-condition-for-row
        #t = pd.Series(e)
        self.DataFrameInModPID = self.DataFrameIn.assign(Score=e)  # https://chrisalbon.com/python/pandas_assign_new_column_dataframe.html
        print('Initial Population compared to the target with score')
        print(self.DataFrameInModPID.head())
        print(' ')
        print(' ')


    def SortOutPreparedInputData(self):
        self.DataFrameInModSOPID = self.DataFrameInModPID.sort_values(self.DataFrameInModPID.columns[-1], ascending=False)
        print('DATA SORTED IN ASCENDING ORDER OF SCORE: ')
        print(' ')
        print(self.DataFrameInModSOPID.head())
        print('')
        print(' ')


class CreateAPopulation(InitialisingDataSet):

################################################
    #def RowPopulation(self):

    def __init__(self):
        self.MergingRows = []
        #self.t1=np.matrix(DataFrameStart)
        self.t1=np.fromfile('/Users/sukhpalsingh/PycharmProjects/ASEWorks/Assessment2/Attempt2/TestMaterials/smallfaultmatrix.txt')
        self.fitness = 0
        i = 0
        y=0
        while i < RowPopulationVal:  # while the counter is less than length of the target
            if rm.random() >= 0.8:  # randomly populate array holding initial population
                self.MergingRows.append(rm.choice(self.t1[:]))
            else:
                self.MergingRows.append(rm.choice(self.t1[:]))
            i += 1
            y+=1
        print(self.MergingRows)

    def GetTheRows(self):
        return self.MergingRows

    def Fitness(self):
        self.fitness=0
        for i in range(self.MergingRows.__len__()):
            if self.MergingRows[i]==target[i]:
                self.fitness+=1
            return self.fitness

    def __str__(self):
        return self.MergingRows.__str__()

#################################################

class MakingTheActualPopulation:
    def __init__(self,PopulationPool):
        self.CreatingRows=[]                #holds all the new rows thatre getting made
        i=0
        while i<PopulationPool:
            self.CreatingRows.append(CreateAPopulation)
            i+=1

    def GetTheRowPopulation(self): return self.CreatingRows


def printpopulation(pop, gennumber):                                                                                    #print population function
    print('\n-------------------------------------------------------')
    print( 'Generation number', gennumber, ', Fitness of highest ranked individual in population: ', pop.GetTheRowPopulation()[0])
    print('Target Individual: ', target, 'Highest possible fitness value: ', maxvalue)
    print("-------------------------------------------------------")
    i=0
    for x in pop.GetTheRowPopulation():                                                                                       #goes through all the individual in the past pop
        print("Randomly generated individual number:", i, ":", x, "Associated fitness score: ")#, x.Fitness())
        i +=1

######################################

#t1=np.matrix
#DealingWithInputData=InitialisingDataSet()
#PopulationControl=CreateAPopulation()
#DealingWithInputData.PrintInputData()
#PopulationControl1=CreateAPopulation()
##PopulationControl1.RandomlyGenerateSolutions()
#PopulationControl1.RowPopulation()
#PopulationControl1.GetTheRows()
#PopulationControl1.Fitness()

PopulationControl2=MakingTheActualPopulation(PopulationPool)
PopulationControl2.GetTheRowPopulation()#.sort(key=lambda x: x.Fitness(), reverse=True)
printpopulation(PopulationControl2,gennumber=0)