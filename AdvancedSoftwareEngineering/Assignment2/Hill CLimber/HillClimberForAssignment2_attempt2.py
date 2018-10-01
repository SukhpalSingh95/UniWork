import random as rm
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
from itertools import *



style.use('fivethirtyeight')

#target =['1','1','1','1','1','1','1','1','1']   #represents the ideal solution

print(' ')
#DataSetIn= pd.read_csv('/Users/sukhpalsingh/PycharmProjects/ASEWorks/Assessment2/Attempt2/TestMaterials/smallfaultmatrix.txt',header=None,sep= ',',index_col=0, comment='#') #https://stackoverflow.com/questions/21546739/load-data-from-txt-with-pandas + https://stackoverflow.com/questions/28200404/pandas-read-table-use-first-column-as-index
DataSetIn = pd.read_csv('/Users/sukhpalsingh/PycharmProjects/ASEWorks/Assessment2/Attempt2/TestMaterials/bigfaultmatrix.txt', header=None,sep=',',comment='#')  # https://stackoverflow.com/questions/21546739/load-data-from-txt-with-pandas + https://stackoverflow.com/questions/28200404/pandas-read-table-use-first-column-as-index
#DataSetIn= pd.read_csv('/Users/sukhpalsingh/PycharmProjects/ASEWorks/Assessment2/Attempt2/TestMaterials/smallfaultmatrix.txt',header=None,sep= ',', comment='#') #https://stackoverflow.com/questions/21546739/load-data-from-txt-with-pandas + https://stackoverflow.com/questions/28200404/pandas-read-table-use-first-column-as-index
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
#print(' ')
PopulationSize=1


class InitialisingDataSet:

    def __init__(self):
        self.DataFrameInitial=DataSetIn
        self.DataFrameLength=len(self.DataFrameInitial)

        self.DataFrameColumns=len(self.DataFrameInitial)
        self.PlayData = pd.DataFrame(data=self.DataFrameInitial)

        print(' ')
        self.DataFrameIn=self.PlayData                       #doing it this way reduces dependence on data frame names
        e = self.DataFrameIn.sum(axis=0)  # https://stackoverflow.com/questions/42886354/pandas-count-values-by-condition-for-row

        self.GenerateAPopulation=pd.DataFrame([])
        self.RandomLocationInSearchSpace=rm.randrange(0,self.DataFrameColumns)

        List=[self.RandomLocationInSearchSpace-3,self.RandomLocationInSearchSpace-2,self.RandomLocationInSearchSpace-1,self.RandomLocationInSearchSpace,self.RandomLocationInSearchSpace+1,self.RandomLocationInSearchSpace+2,self.RandomLocationInSearchSpace+3,]
        self.GenerateAPopulation1=self.PlayData.ix[List]

        ######################################################################################################
        ######################################################################################################
        ######################################################################################################
        self.HCInitialPopulation=pd.DataFrame(self.GenerateAPopulation1)
        print('Initial HC population')
        print(self.HCInitialPopulation)
        print(' ')

        self.HCt2=self.HCInitialPopulation.sum(axis=1)
        self.HCt2Seriesform=pd.Series(self.HCt2)
        self.HCt2Seriesform=self.HCt2Seriesform.rename('Score')
        self.Genpop=pd.concat([self.HCInitialPopulation,self.HCt2Seriesform],axis=1)
        self.Genpopseries = self.Genpop.sort_values(self.Genpop.columns[-1], ascending=False)
        #self.Genpopseries1 = self.Genpopseries.drop(self.Genpopseries.index[-1], inplace=True)

        self.HCt4=self.Genpopseries.values.tolist()
        #print(self.HCt4[0])
        #print(self.Genpopseries)
        self.t=self.HCt4[0]
        #print(self.t)
        #self.HCt4=self.Genpopseries1.ix[0]

        #List=[self.Genpopseries.ix[0]]
        #print(List)
        self.GenerateAPopulation1=self.t#self.Genpopseries.ix[List]
        print(self.GenerateAPopulation1)

        ######################################################################################################
        ######################################################################################################
        ######################################################################################################

        print('Search Space Population')
        #print(self.GenerateAPopulation1)
        print('Finished with HC')



class HillClimbing(InitialisingDataSet):

    def PopulationAnalysis(self):
        #e = self.GenerateAPopulation1.sum(axis=0)  # https://stackoverflow.com/questions/42886354/pandas-count-values-by-condition-for-row
        #self.GenerateAPopulation1Scored = self.GenerateAPopulation1#.assign(Score=e)  # https://chrisalbon.com/python/pandas_assign_new_column_dataframe.html
        self.GenerateAPopulation1Scored=pd.Series(self.GenerateAPopulation1)
        self.t2=self.GenerateAPopulation1Scored#.sum(axis=0)
        #print(self.GenerateAPopulation1Scored)
        self.v=self.t2.ix[0]
        self.t2.drop(self.t2.index[0],inplace=True)
        self.t2.drop(self.t2.index[-1], inplace=True)
        #print('IM here')
        #print(self.t2)
        self.t3=self.t2.to_frame().T#.values
        #print(self.t3)

        print('')#print(self.t2.to_frame().T.values)
        self.fitness1=0 #number of greater than values
        self.fitness2=0 #number of equal values
        self.fitness3=0 #number of zero values

        self.t4=self.t3.values.tolist()
        self.t5=self.t4[0]
        print(self.t5)
        for i in self.t5:
            if i>target[i]:
                self.fitness1+=1
                #print(q)
            elif i==target[i]:
                self.fitness2+=1
            elif i<target[i]:
                self.fitness3+=1#

        self.OverallFitness=0.1*self.fitness1+0.001*self.fitness2+self.fitness3

        print('Test set combination: '+str(self.v))
        print(' ')
        print('Number of times tests are repeated within test set: '+str(self.fitness1))
        print('Number of times only a test found 1 fault that no others did within test space: '+str(self.fitness2))
        print('Number of test cases that reported no faults: '+str(self.fitness3))
        print('Overall fitness of this set: '+ str(self.OverallFitness))
        self.FinalDataFrameNames = pd.Series(self.v, name='Test Combinations')
        self.FinalDataFrameData = pd.Series(self.t5, name='Data sets for tests')
        self.FinalDataFrameOverallFitness = pd.Series(self.OverallFitness, name='Overall fitness for each test')
        self.FinalDataFrameND = pd.concat([self.FinalDataFrameNames, self.FinalDataFrameData, self.FinalDataFrameOverallFitness], axis=0)

        return self.v, self.t5,self.OverallFitness,self.fitness1,self.fitness2,self.fitness3 #self.FinalDataFrameNames#,self.FinalDataFrameData#,self.FinalDataFrameOverallFitness

Pops=[]
#NumberOfIterations=0

Generation=0
for x in range(10):
    print('--------------------------------------------------------------')
    print('This Generation is: '+str(Generation))
    pol=HillClimbing().PopulationAnalysis()
    pol
    Pops.append(pol)
    Generation+=1

print('---------------------------------------------------------')


ddd=[]
#print(str(Pops))
#print('------------------------')
for i in enumerate(Pops):
    ddd.append(i[1])
        #print(i[1])

print(' ')
#print(ddd)
TheSeries=pd.DataFrame(ddd,columns=['Test Cases','Test faults combined','Overall Fitness','Fitness1','Fitness2','Fitness3'])
TheSeries1=pd.DataFrame(TheSeries)#.sort_values(TheSeries.columns[-1]) #UNSORTE LIST
TheSeries2=TheSeries.sort_values(TheSeries.columns[-1])  #SORTED LIST
print('Unsorted list')
print(' ')
print(TheSeries1)
print('')
print('sorted list')
print(' ')
print(TheSeries2)
print('')
print('')
print('')

unsortedlist1=TheSeries1.groupby('Test Cases')['Fitness1'].sum()

unsortedlist3=TheSeries1['Test Cases']
unsortedlist1=TheSeries1['Fitness1']
unsortedlist2=TheSeries1['Fitness2']
unsortedlist=pd.DataFrame([unsortedlist3,unsortedlist1,unsortedlist2])
tttt=unsortedlist.sum()

sortedlist3=TheSeries2['Test Cases']
sortedlist1=TheSeries2['Fitness1']
sortedlist2=TheSeries2['Fitness2']
sortedlist=pd.DataFrame([sortedlist3,sortedlist1,sortedlist2])
tttt1=sortedlist.sum()

SortedVsUnsorted=pd.DataFrame([sortedlist,unsortedlist])

write=pd.ExcelWriter('Unsorted Test 5.xlsx',engine='xlsxwriter')
unsortedlist.to_excel(write,sheet_name='Sheet1')
write.save()

write=pd.ExcelWriter('Sorted Test 5.xlsx',engine='xlsxwriter')
sortedlist.to_excel(write,sheet_name='Sheet1')
write.save()