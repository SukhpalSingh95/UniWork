######## NOTES #########
# PARETOFRONTS AREN'T ACTUALLY PARETO FRONTS, INSTEAD THEY ARE JUST HIGH ACHIEVERS FROM THE SELECTION PROCESS



import random as rm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math as mh
from itertools import *
#import evoalgos

print('')
print('-----------------------')
print('Importing the text file')
print('-----------------------')
print('')
print('')
#DataSetIn = pd.read_csv('nrp1.txt', header=None, index_col=None)#, index=['Profit Of customer', 'Number of requests','Requirements list']) #sep=' ')#,comment='#')  # https://stackoverflow.com/questions/21546739/load-data-from-txt-with-pandas + https://stackoverflow.com/questions/28200404/pandas-read-table-use-first-column-as-index#
DataSetIn = pd.read_csv('nrp-m4.txt', header=None, index_col=None)#, index=['Profit Of customer', 'Number of requests','Requirements list']) #sep=' ')#,comment='#')  # https://stackoverflow.com/questions/21546739/load-data-from-txt-with-pandas + https://stackoverflow.com/questions/28200404/pandas-read-table-use-first-column-as-index

DataFrameStart=pd.DataFrame(DataSetIn)


class MakingSenseOfTheDataFile:
    def __init__(self):
        print('Reading the csv file')
        print('')
        print(DataSetIn.head())
        print('')
        print('DataFrame')
        DataFrameStart = pd.DataFrame(DataSetIn)
        print(DataFrameStart.tail())
        print('')
        print('')

        ###############################################################################
        ######## making sense of the data #################
        print('')
        print('-------------------------')
        print('Making sense of the data')
        print('-------------------------')
        print('')
        print('')

        # THE FIRST VALUE SAYS HOW MANY LEVELS THERE ARE

        self.NumOfLevels = DataSetIn[0][0]  # .astype(float64)
        print('Number of levels: ' + str(self.NumOfLevels))
        NumOfLevels = int(self.NumOfLevels)
        print('')
        print('')

        ####### READ THE NEXT SET OF LINES AS THE NUMBER OF REQUIREMENTS AND THE COSTS ########
        #### THIS SHOULD ACTUALLY BE 2*NUMBER OF LEVELS ########

        self.TopSectionLength = int(2 * int(self.NumOfLevels) + 1)
        # print(TopSectionLength)
        print('The top section of the dataframe')
        print('--------------------------------')
        print('')
        self.TopSectionOfDataFrame = DataFrameStart[0][1:(self.TopSectionLength)]
        print(str(self.TopSectionOfDataFrame))
        print('')
        print('')

        ########## the last value in this list is the number of entries to read in next #######
        self.NumberOfDependencies = DataFrameStart[0][self.TopSectionLength]
        print('Number of Dependencies: ' + str(self.NumberOfDependencies))
        print('---------------------------')
        print('')
        self.NumberOfDependencies = int(self.NumberOfDependencies)
        self.TotalLengthOfDependenciesETC = int(1 + self.TopSectionLength + self.NumberOfDependencies)
        self.Dependencies = DataFrameStart[0][int(self.TopSectionLength + 1):int(self.TotalLengthOfDependenciesETC)]
        print('Head of the dataframe')
        print('---------------------')

        print(str(self.Dependencies.head()))
        print('')
        print('Tail of the dataframe')
        print('---------------------')
        print(str(self.Dependencies.tail()))




    def DataFrameTypeFormatting(self):
        print('')
        print('')
        print('----------------------')
        print('----------------------')
        print('Now onto the customers')
        print('----------------------')
        print('----------------------')
        print('')
        print('')
        print('')

        self.NumberOfCustomers = DataFrameStart[0][self.TotalLengthOfDependenciesETC]
        print('Number of customers: ' + str(self.NumberOfCustomers))
        print('------------------------')
        print('')

        #NumberOfCustomers = int(NumberOfCustomers)

        print('')
        print('Customer Details: ')
        print('-----------------')

        self.CustomersDetails = DataFrameStart[0][int(self.TotalLengthOfDependenciesETC + 1):]
        print(str(self.CustomersDetails.head()))

        print('')
        print('')
        print('Putting customer details into sorted lists')
        print('------------------------------------------')

        self.CustomerDataFrame = pd.Series(self.CustomersDetails)
        print(self.CustomerDataFrame.head())

        self.ListingCustomerDataFrame = self.CustomerDataFrame.tolist()
        self.NewListOfCustomers = []
        for i in self.ListingCustomerDataFrame:
            self.NewListOfCustomers.append(i.split())

        print('')
        print('')

        print('Putting the customers data into a dataframe:')
        print('--------------------------------------------')
        print('')
        print('')
        print('')
        self.CustomerDataFrame2 = pd.DataFrame(self.NewListOfCustomers)
        for i in range(len(self.CustomerDataFrame2.columns)):
            self.CustomerDataFrame2[i] = pd.to_numeric(self.CustomerDataFrame2[i]).astype(np.float64)
        self.CustomerDataFrame2.fillna(0, inplace=True)

        print('Top of the dataframe with customer details:')
        print('-------------------------------------------')

        print(self.CustomerDataFrame2.head())
        print('')
        print('End of dataframe with customer details')
        print('--------------------------------------')

        print(self.CustomerDataFrame2.tail())
        print('')

        # NORMALISE THE CUSTOMER WEIGHTS

        print('Normalising values for each column')
        print('')
        self.NormalisingValues = self.CustomerDataFrame2.sum(axis=0)
        self.NormalisedCustomerDataFrameCol0 = self.CustomerDataFrame2.loc[:, 0].div(self.NormalisingValues[0],axis=0)  # https://stackoverflow.com/questions/26537878/pandas-sum-across-columns-and-divide-each-cell-from-that-value
        self.NormalisedCustomerDataFrameCol1 = self.CustomerDataFrame2.loc[:, 1].div(self.NormalisingValues[1],axis=0)
        self.CustomerDataFrame3 = self.CustomerDataFrame2.drop([0], axis=1)
        self.NormalisedCustomerDataFrame = pd.concat([self.NormalisedCustomerDataFrameCol0, self.CustomerDataFrame3], axis=1)
        print('')
        print('Normalised customer weighting - normalisedCustomerDataFrame')
        print(self.NormalisedCustomerDataFrame.tail())
        print('')
        # print(NormalisedCustomerDataFrame.sum(axis=0))


    def FindingRepeatingFunctionsValues(self):
        self.ValuesInRequirementsList=self.CustomerDataFrame2.drop([0,1],axis=1)
        self.ValuesInRequirementsList2=self.ValuesInRequirementsList.apply(pd.value_counts)
        print('')
        print('Values in requirements list as a Series')
        self.ValuesInRequirementsListAppearing=pd.Series(self.ValuesInRequirementsList2.index)
        self.ValuesInRequirementsListAppearing=pd.to_numeric(self.ValuesInRequirementsListAppearing).astype(np.float64)

        self.ValuesInRequirementsListAppearing1=self.ValuesInRequirementsListAppearing.drop([0][0])
        print(self.ValuesInRequirementsListAppearing1.head())
        self.T1=self.ValuesInRequirementsListAppearing1.tolist()
        print('')

        ########## NEED TO FIND VALUES ############
        tolist1=[]
        tolist2=[]

        for index, rows in self.ValuesInRequirementsList.iterrows():
            rowvalu=0
            for w in rows:
                tolist1.append(w)
                rowvalu+=1.0
                tolist2.append((1.0/rowvalu))
                #print(tolist2)
        print('Ordered list version of dataframe')
        print(tolist1)
        print('')
        print(tolist2)

        tolist3=[]
        for i in tolist1:
            if i>0:
                tolist3.append(1)
            else:
                tolist3.append(0)

        self.ValueList=[a * b for a, b in zip(tolist2, tolist3)] #https://stackoverflow.com/questions/10271484/how-to-perform-element-wise-multiplication-of-two-lists-in-python
        print('Value list')
        print(self.ValueList)
        print('')
        lengthoftruncation=len(self.ValuesInRequirementsList.columns)
        self.ValueListTruncated=[self.ValueList[i:i + lengthoftruncation] for i in range(0, len(self.ValueList), lengthoftruncation)] #https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
        print('Value list truncated')
        print(self.ValueListTruncated)
        print('')


        self.CustomerWeight=pd.Series(self.NormalisedCustomerDataFrameCol0).tolist()

        self.CustomerWeightList=[]
        numnum=len(self.ValueListTruncated[0])-1

        for i in ((self.CustomerWeight)):
            self.CustomerWeightList.append(i)
            self.CustomerWeightList.extend([i]*numnum)

        print('')
        print('Filled out the customer weighting to create equal lengthed lists')
        print('')
        print(self.CustomerWeightList)
        print('')
        self.CustomerWeightTimesValues=[a * b for a, b in zip(self.CustomerWeightList, self.ValueList)]
        print((self.CustomerWeightTimesValues))
        print('')
        print('Customer weight times values')
        print('')
        self.DataframeForDays=pd.DataFrame([tolist1,self.CustomerWeightTimesValues]).T
        print(self.DataframeForDays.head(12))
        print('')
        print('The indices are were those values where in the lists')
        print('The 0 column is the requirement number/cost')
        print('The 1 column is the number of score of each requirement in ascending order')
        print('')
        self.DataframeForDays.sort_values([0,1],ascending=True, inplace=True)
        print(self.DataframeForDays.tail(12))
        print('Length of this dataframe'+str(len(self.DataframeForDays)))
        print('')
        print('Score values for each requirement. This is found from multiplying the weight of the customer by the value')
        self.DataframeForDays1=self.DataframeForDays.groupby(0).sum()
        print('Dataframe for days1')
        print(self.DataframeForDays1.tail(12))
        print('Length of this dataframe/total number of requirements requested : '+str(len(self.DataframeForDays1)))
        print('')

        print('Section: Putting the scores beside the requirement number in one dataframe')
        print('')
        self.DataframeForDays2=self.DataframeForDays1
        self.DataframeForDays2['Req Num']=self.DataframeForDays2.index
        self.DataframeForDays2['Score/Fitness']=self.DataframeForDays2[1]
        self.DataframeForDays2=self.DataframeForDays2.drop([1],axis=1)
        self.DataframeForDays2=self.DataframeForDays2.reset_index(drop=True)        #https://stackoverflow.com/questions/20490274/how-to-reset-index-in-a-pandas-data-frame
        print(self.DataframeForDays2.head())
        print('')

    def EvaluteCostRrequirements(self):

        #this creates a cost requirement

        print('Evaluating the Cost requirement')
        print('')
        self.DataframeForDays2=self.DataframeForDays2.drop([0],axis=0)  #NEED TO REMOVE ZERO TERM OTHERWISE DIVIDING BY ZERO

        print(self.DataframeForDays2.head())
        print('')
        self.CostFitness=[]
        for i in range(1,len(self.DataframeForDays2)):
            self.CostFitness.append((1.0/float(i)))

        self.SeriesCostFitness=pd.Series(self.CostFitness)
        self.SeriesCostFitness=self.SeriesCostFitness.rename('Cost Fitness',inplace=True)
        self.DataframeForDays2=self.DataframeForDays2.reset_index(drop=True)        #https://stackoverflow.com/questions/20490274/how-to-reset-index-in-a-pandas-data-frame
        self.DataframeForDays3=pd.concat([self.DataframeForDays2,self.SeriesCostFitness],axis=1)
        print(self.DataframeForDays3.head())


    def AverageFitness(self):

        # this makes a list for average by multiplying the score and the cost together
        # THIS IS WHERE I NEED TO MAKE THE CHANGE FOR THE POPULATION
        self.AverageFitness1=[]
        self.AverageFitness2=[]
        for index, row in self.DataframeForDays3.iterrows():
            self.AverageFitness1.append(self.DataframeForDays3['Score/Fitness'][index]) #gets fitnesses from score
            self.AverageFitness2.append(self.DataframeForDays3['Cost Fitness'][index])  #gets fitnesses from cost
        self.AverageFitness3 = [(a+b)/2 for a, b in zip(self.AverageFitness1, self.AverageFitness2)]
        print('')
        self.SeriesAverageFitness3=pd.Series(self.AverageFitness3)
        self.SeriesAverageFitness3=self.SeriesAverageFitness3.rename('Average Fitness',inplace=True)
        self.DataframeForDays4=pd.concat([self.DataframeForDays3, self.SeriesAverageFitness3],axis=1)

        print(self.DataframeForDays4.head())




    def CreateRandomPopulation(self):
        self.PopulationSize=len(self.DataframeForDays4)
        #self.DataframeForDays4=self.DataframeForDays4.sort_values(['Average Fitness'])
        self.RandomPopulationSample=self.DataframeForDays4.sample(n=self.PopulationSize)#.astype(np.float64)
        #self.RandomPopulationSample=self.DataframeForDays4[]

        print('')
        return self.RandomPopulationSample


    def CrowdDistancing(self,populationtoswap):

        ######## isnt actually used


        print('Crowd Distancing')
        self.Populationunderswapping=populationtoswap
        self.populationexchange=self.MutatePop
        self.Populationexchangelength=len(self.populationexchange)
        self.columnsearch=3         #depends on objective, if you want better customer satisfaction, make 1, better cost reduction make 2
        for index in range(0, len(self.Populationunderswapping)-1):
            for i in range(0,self.Populationexchangelength):
                #USING GREATER THAN BECAUSE THIS IS MULTIOBJECTIVE RATHER THAN FOCUSING ON A SINGLE OBJECTIVE.
                if ((self.Populationunderswapping.iloc[index][self.columnsearch])-(self.Populationunderswapping.iloc[index+1][self.columnsearch]))>((self.Populationunderswapping.iloc[index][self.columnsearch])-(self.populationexchange.iloc[i][self.columnsearch])):
                    v=pd.Series(self.Populationunderswapping.loc[index+1])
                    self.Populationunderswapping.loc[index+1], self.populationexchange.loc[i]=self.populationexchange.loc[i].copy(),v.copy()    #https://stackoverflow.com/questions/35113945/how-to-swap-rows-in-2-pandas-dataframes

    def CrowdDistancing1(self,populationtoswap):

        ###### isnt actually used


        print('Crowd Distancing1')
        self.PopulationToRank=populationtoswap
        #print(self.PopulationToRank)
        print('')
        self.PopulationToRanklength=len(self.PopulationToRank)
        print(self.PopulationToRanklength)
        self.EmptyList=[]
        crowdindColumn=1
        for indexer in range(0,self.PopulationToRanklength-1):
            self.EmptyList.append(self.PopulationToRank.iloc[indexer][crowdindColumn]-self.PopulationToRank.iloc[indexer+1][crowdindColumn])
        self.EmptyList.insert(0,0)
        self.PopulationToRank['Crowd distance score'] = self.EmptyList

        self.EmptyList = []
        crowdindColumn = 2
        for indexer in range(0, self.PopulationToRanklength - 1):
            self.EmptyList.append(self.PopulationToRank.iloc[indexer][crowdindColumn] - self.PopulationToRank.iloc[indexer + 1][crowdindColumn])
        self.EmptyList.insert(0, 0)
        self.PopulationToRank['Crowd distance cost'] = self.EmptyList

        self.EmptyList = []
        crowdindColumn = 3
        for indexer in range(0, self.PopulationToRanklength - 1):
            self.EmptyList.append(self.PopulationToRank.iloc[indexer][crowdindColumn] - self.PopulationToRank.iloc[indexer + 1][crowdindColumn])
        self.EmptyList.insert(0, 0)
        self.PopulationToRank['Crowd distance average'] = self.EmptyList

        print(self.PopulationToRank.head())


####### EVERYTHING ABOVE HERE SIMPLY PREPARES THE DATA, IT GENERATES COST FITNESS AND SSCORE FITNESS AND AVERGAE FITNESS

weight=0.5

populations=[]
Step1=MakingSenseOfTheDataFile()
Step1
Step1.DataFrameTypeFormatting()
Step1.FindingRepeatingFunctionsValues()
Step1.EvaluteCostRrequirements()
Step1.AverageFitness()
print(' ')



#sampling the text file to create a population
Generation=0
#PopulationSize=20
NumberOfGenerations=1
for i in range(0,NumberOfGenerations):
    populations.append(Step1.CreateRandomPopulation())
    Generation+=1
    GenerationalSampledPopulation=pd.concat(populations)


InputPopulation = pd.DataFrame(GenerationalSampledPopulation)
#InputPopulation = InputPopulation.sort_values(['Average Fitness'], ascending=False)
print('')
print('The entire sampled population')
print(InputPopulation.head())
print('Population length/sampled' +str(len(InputPopulation)))

####### EVERYTHING ABOVE HERE SIMPLY CREATES A POPULATION





##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
##################################################################################################################################


InputPopulation1=InputPopulation
print(InputPopulation1.head())
print('')
InputPopulationSorers=[]
for i in range(len(InputPopulation)-1):
    if InputPopulation1.iloc[i][1]>InputPopulation1.iloc[i+1][1]:
        InputPopulationSorers.append(InputPopulation1.iloc[i])
        InputPopulationScorersDF = pd.DataFrame(InputPopulationSorers)
print(len(InputPopulationSorers))
InputPopulationScorersDF=InputPopulationScorersDF.sort_values(['Score/Fitness'],ascending=False)
print('Scorers')
print(InputPopulationScorersDF.head())
print('')

#Placement=0

ScoreFitness = []
for i in range(1,len(InputPopulationScorersDF)):
    normaliser=InputPopulationScorersDF['Score/Fitness'].sum()
    ScoreFitness.append((InputPopulationScorersDF.iloc[i][1] / normaliser))
SeriesCostFitness = pd.Series(ScoreFitness)
SeriesCostFitness = SeriesCostFitness.rename('Score Fitness', inplace=True)
InputPopulationScorersDF = InputPopulationScorersDF.reset_index(drop=True)  # https://stackoverflow.com/questions/20490274/how-to-reset-index-in-a-pandas-data-frame
DataframeForDays3 = pd.concat([InputPopulationScorersDF, SeriesCostFitness], axis=1)
InputPopulationScorersDF=DataframeForDays3
#print(InputPopulationScorersDF.head())
print('')
InputPopulationScorersDF=InputPopulationScorersDF[['Req Num','Cost Fitness','Score Fitness','Average Fitness']]
print('Scorers')
print(InputPopulationScorersDF.head())


InputPopulationCosters=[]
for i in range(len(InputPopulation)-1):
    if InputPopulation1.iloc[i][2]>InputPopulation1.iloc[i+1][2]:
        InputPopulationCosters.append(InputPopulation1.iloc[i])
        InputPopulationCostersDF=pd.DataFrame(InputPopulationCosters)
print(len(InputPopulationCosters))
InputPopulationCostersDF=InputPopulationCostersDF.sort_values(['Cost Fitness'], ascending=False)
print('Costers')
InputPopulationCostersDF['Score Fitness']=InputPopulationCostersDF['Score/Fitness']
InputPopulationCostersDF=InputPopulationCostersDF[['Req Num','Cost Fitness', 'Score Fitness','Average Fitness']]
print(InputPopulationCostersDF.head())
print('')


CostFitness = []
InputPopulationCostersDF=InputPopulationCostersDF.rename(columns={'Cost Fitness':'Old CFitness'})
for i in range(1,len(InputPopulationScorersDF)):
    normaliser=InputPopulationCostersDF['Old CFitness'].sum()
    CostFitness.append((InputPopulationScorersDF.iloc[i][1] / normaliser))
SeriesCostFitness = pd.Series(CostFitness)
SeriesCostFitness = SeriesCostFitness.rename('Cost Fitness', inplace=True)
InputPopulationCostersDF = InputPopulationCostersDF.reset_index(drop=True)  # https://stackoverflow.com/questions/20490274/how-to-reset-index-in-a-pandas-data-frame
DataframeForDays3 = pd.concat([InputPopulationCostersDF, SeriesCostFitness], axis=1)
InputPopulationCostersDF=DataframeForDays3
InputPopulationCostersDF=InputPopulationCostersDF[['Req Num','Cost Fitness', 'Score Fitness','Average Fitness']]
print(InputPopulationCostersDF.head())
print('')







trialer=[]
remains=[]
if len(InputPopulationCostersDF)<len(InputPopulationScorersDF):         #means that holder 1 will contain the coster data
    holder2=InputPopulationCostersDF                                    #holder 1 should have the most number of dataframe elements
    print('Holder 2 is the costers')
    holder2columninterest=1
    holder1=InputPopulationScorersDF[:len(holder2)]
    holder1columninterest=2
    print('holder 1 is the scorers')
    case=0

else:
    holder2=InputPopulationScorersDF    #will contain the scorer data
    holder2columninterest=1
    print('Holder 1 is the costers')
    print('holder 2 is the scorers')

    holder1=InputPopulationCostersDF[:len(holder2)]
    holder1columninterest=2
    case=1


print('')
print('holder 1')
print(holder1.head())
print(len(holder1))

print('')
print('holder 2')
print(holder2.head())
print(len(holder2))

print('Case'+str(case))

emptylist1=[]
emptylist2=[]
for i in range(len(holder1)):
   if holder1.iloc[i][1]>holder2.iloc[i][2]:
       emptylist1.append(holder1.iloc[i])
       emptylist2.append(holder2.iloc[i])
       populationdf=pd.DataFrame(emptylist1)
       remainderDF=pd.DataFrame(emptylist2)

   if holder1.iloc[i][1]<holder2.iloc[i][2]:
       emptylist1.append(holder2.iloc[i])
       emptylist2.append(holder1.iloc[i])
       populationdf=pd.DataFrame(emptylist1)
       remainderDF=pd.DataFrame(emptylist2)
print('im here')
print(populationdf.head())
print(remainderDF.head())

InputPopulation=populationdf

MutatePop=remainderDF
############### redo fitness for score



fivepercentpop = int(0.05 * (len(InputPopulation)))
tenpercentpop = int(0.1 * (len(InputPopulation)))
twentypercentpop = int(0.15 * (len(InputPopulation)))
thirtypercentpop = int(0.2 * (len(InputPopulation)))
fourtypercentpop = int(0.25 * (len(InputPopulation)))




print('1st level')
Paretofront1 = InputPopulation[:fivepercentpop]
Paretofront1 = Paretofront1.reset_index(drop=True)  # https://stackoverflow.com/questions/20490274/how-to-reset-index-in-a-pandas-data-frame
print(Paretofront1.tail())
print('')


print('2nd level')
Paretofront2 = InputPopulation[fivepercentpop:tenpercentpop]
Paretofront2 = Paretofront2.reset_index(drop=True)  # https://stackoverflow.com/questions/20490274/how-to-reset-index-in-a-pandas-data-frame
print(Paretofront2.tail())
print('')



print('3rd level')
Paretofront3 = InputPopulation[tenpercentpop:twentypercentpop]
Paretofront3 = Paretofront3.reset_index(drop=True)  # https://stackoverflow.com/questions/20490274/how-to-reset-index-in-a-pandas-data-frame
print(Paretofront3.tail())
print('')


print('4th Level')
Paretofront4 = InputPopulation[twentypercentpop:thirtypercentpop]
Paretofront4 = Paretofront4.reset_index(drop=True)  # https://stackoverflow.com/questions/20490274/how-to-reset-index-in-a-pandas-data-frame
print(Paretofront4.tail())
print('')


print('5th level')
Paretofront5 = InputPopulation[thirtypercentpop:fourtypercentpop]
Paretofront5 = Paretofront5.reset_index(drop=True)  # https://stackoverflow.com/questions/20490274/how-to-reset-index-in-a-pandas-data-frame
print(Paretofront5.tail())
print('')

print('Population for Mutation')
MutatePop = InputPopulation[fourtypercentpop:]
#MutatePop=RemainsDF
MutatePop = MutatePop.reset_index(drop=True)  # https://stackoverflow.com/questions/20490274/how-to-reset-index-in-a-pandas-data-frame



##### MUTATING THE POPULATION ######
NumOfIterations=1   #NumberOfGenerations
populations2=[]
itterationnumber=0

for i in range(0,NumOfIterations):
    t=rm.randint(1,200)
    if t>180:
        PopulationSize = (len(Paretofront3)-1)
        RandomSampleFront3 = Paretofront3.sample(n=1)
        IndexofSample3 = RandomSampleFront3.index[0]
        RandomMutateSample = MutatePop.sample(n=1)
        IndexofRandomMutate = RandomMutateSample.index[0]
        Paretofront3 = Paretofront3.drop(Paretofront3.index[IndexofSample3])
        Paretofront3 = Paretofront3.append(RandomMutateSample)
        MutatePop = MutatePop.drop(MutatePop.index[IndexofRandomMutate])
        MutatePop = MutatePop.append(RandomSampleFront3)
        Paretofront3 = Paretofront3.reset_index(drop=True)
        MutatePop = MutatePop.reset_index(drop=True)


    if t>90 & t<110:
        PopulationSize = (len(Paretofront4)-1)
        RandomSampleFront4 = Paretofront4.sample(n=1)
        IndexofSample4 = RandomSampleFront4.index[0]
        RandomMutateSample = MutatePop.sample(n=1)
        IndexofRandomMutate = RandomMutateSample.index[0]
        Paretofront4 = Paretofront4.drop(Paretofront4.index[IndexofSample4])
        Paretofront4 = Paretofront4.append(RandomMutateSample)
        MutatePop = MutatePop.drop(MutatePop.index[IndexofRandomMutate])
        MutatePop = MutatePop.append(RandomSampleFront4)
        Paretofront4 = Paretofront4.reset_index(drop=True)
        MutatePop = MutatePop.reset_index(drop=True)


    if t>0 &t<20:

        PopulationSize = (len(Paretofront5)-1)
        RandomSampleFront5 = Paretofront5.sample(n=1)

        IndexofSample5 = RandomSampleFront5.index[0]
        RandomMutateSample = MutatePop.sample(n=1)

        IndexofRandomMutate = RandomMutateSample.index[0]
        Paretofront5 = Paretofront5.drop(Paretofront5.index[IndexofSample5])

        Paretofront5 = Paretofront5.append(RandomMutateSample)
        MutatePop = MutatePop.drop(MutatePop.index[IndexofRandomMutate])

        MutatePop = MutatePop.append(RandomSampleFront5)
        Paretofront5 = Paretofront5.reset_index(drop=True)

        MutatePop = MutatePop.reset_index(drop=True)


    itterationnumber+=1

########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
#################################################### Need to put in cross over #########################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################


for u in range(2):
    for i in range(0, len(Paretofront1)-1):
            frontin = Paretofront1
            distance2 = mh.sqrt((frontin.iloc[i-1]['Cost Fitness']-frontin.iloc[i]['Cost Fitness'])**2+(frontin.iloc[i-1]['Score Fitness']-frontin.iloc[i]['Score Fitness'])**2)#(frontin.iloc[i]['Average Fitness']-frontin.iloc[i+1]['Average Fitness'])
            distance1 = mh.sqrt((frontin.iloc[i]['Cost Fitness']-frontin.iloc[i+1]['Cost Fitness'])**2+(frontin.iloc[i]['Score Fitness']-frontin.iloc[i+1]['Score Fitness'])**2)#(frontin.iloc[i-1]['Average Fitness']-frontin.iloc[i]['Average Fitness'])
            if distance2<distance1:

                RandomSampleFront = frontin.iloc[i+1]  # Paretofront5.sample(n=1)
                IndexofSample5 = RandomSampleFront.name  # index[0]
                RandomMutateSample = Paretofront2.iloc[0]
                IndexofRandomMutate = RandomMutateSample.name  # ix[0]
                frontin = frontin.drop(IndexofSample5)
                Paretofront2 = Paretofront2.drop(IndexofRandomMutate)
                frontin = frontin.append(RandomMutateSample)
                Paretofront2 = Paretofront2.append(RandomSampleFront)
                frontin = frontin.reset_index(drop=True)
                Paretofront2 = Paretofront2.reset_index(drop=True)
                Paretofront1=frontin

    for i in range(0, len(Paretofront2)-1):
            frontin = Paretofront2
            distance2 = mh.sqrt((frontin.iloc[i-1]['Cost Fitness']-frontin.iloc[i]['Cost Fitness'])**2+(frontin.iloc[i-1]['Score Fitness']-frontin.iloc[i]['Score Fitness'])**2)#(frontin.iloc[i]['Average Fitness']-frontin.iloc[i+1]['Average Fitness'])
            distance1 = mh.sqrt((frontin.iloc[i]['Cost Fitness']-frontin.iloc[i+1]['Cost Fitness'])**2+(frontin.iloc[i]['Score Fitness']-frontin.iloc[i+1]['Score Fitness'])**2)#(frontin.iloc[i-1]['Average Fitness']-frontin.iloc[i]['Average Fitness'])
            if distance2<distance1:

                RandomSampleFront = frontin.iloc[i+1]  # Paretofront5.sample(n=1)
                IndexofSample5 = RandomSampleFront.name  # index[0]
                RandomMutateSample = Paretofront3.iloc[0]
                IndexofRandomMutate = RandomMutateSample.name  # ix[0]
                frontin = frontin.drop(IndexofSample5)
                Paretofront3 = Paretofront3.drop(IndexofRandomMutate)
                frontin = frontin.append(RandomMutateSample)
                Paretofront3 = Paretofront3.append(RandomSampleFront)
                frontin = frontin.reset_index(drop=True)
                Paretofront3 = Paretofront3.reset_index(drop=True)
                Paretofront2=frontin
    for i in range(0, len(Paretofront3)-1):
            frontin = Paretofront3
            distance2 = mh.sqrt((frontin.iloc[i-1]['Cost Fitness']-frontin.iloc[i]['Cost Fitness'])**2+(frontin.iloc[i-1]['Score Fitness']-frontin.iloc[i]['Score Fitness'])**2)#(frontin.iloc[i]['Average Fitness']-frontin.iloc[i+1]['Average Fitness'])
            distance1 = mh.sqrt((frontin.iloc[i]['Cost Fitness']-frontin.iloc[i+1]['Cost Fitness'])**2+(frontin.iloc[i]['Score Fitness']-frontin.iloc[i+1]['Score Fitness'])**2)#(frontin.iloc[i-1]['Average Fitness']-frontin.iloc[i]['Average Fitness'])
            if distance2<distance1:

                RandomSampleFront = frontin.iloc[i+1]  # Paretofront5.sample(n=1)
                IndexofSample5 = RandomSampleFront.name  # index[0]
                RandomMutateSample = Paretofront4.iloc[0]
                IndexofRandomMutate = RandomMutateSample.name  # ix[0]
                frontin = frontin.drop(IndexofSample5)
                Paretofront4 = Paretofront4.drop(IndexofRandomMutate)
                frontin = frontin.append(RandomMutateSample)
                Paretofront4 = Paretofront4.append(RandomSampleFront)
                frontin = frontin.reset_index(drop=True)
                Paretofront4 = Paretofront4.reset_index(drop=True)
                Paretofront3=frontin

    for i in range(0, len(Paretofront4)-1):
            frontin = Paretofront4
            distance2 = mh.sqrt((frontin.iloc[i-1]['Cost Fitness']-frontin.iloc[i]['Cost Fitness'])**2+(frontin.iloc[i-1]['Score Fitness']-frontin.iloc[i]['Score Fitness'])**2)#(frontin.iloc[i]['Average Fitness']-frontin.iloc[i+1]['Average Fitness'])
            distance1 = mh.sqrt((frontin.iloc[i]['Cost Fitness']-frontin.iloc[i+1]['Cost Fitness'])**2+(frontin.iloc[i]['Score Fitness']-frontin.iloc[i+1]['Score Fitness'])**2)#(frontin.iloc[i-1]['Average Fitness']-frontin.iloc[i]['Average Fitness'])
            if distance2<distance1:

                RandomSampleFront = frontin.iloc[i+1]  # Paretofront5.sample(n=1)
                IndexofSample5 = RandomSampleFront.name  # index[0]
                RandomMutateSample = Paretofront5.iloc[0]
                IndexofRandomMutate = RandomMutateSample.name  # ix[0]
                frontin = frontin.drop(IndexofSample5)
                Paretofront5 = Paretofront5.drop(IndexofRandomMutate)
                frontin = frontin.append(RandomMutateSample)
                Paretofront5 = Paretofront5.append(RandomSampleFront)
                frontin = frontin.reset_index(drop=True)
                Paretofront5 = Paretofront5.reset_index(drop=True)
                Paretofront4=frontin

    for i in range(0, len(Paretofront5)-1):
            frontin = Paretofront5
            distance2 = mh.sqrt((frontin.iloc[i-1]['Cost Fitness']-frontin.iloc[i]['Cost Fitness'])**2+(frontin.iloc[i-1]['Score Fitness']-frontin.iloc[i]['Score Fitness'])**2)#(frontin.iloc[i]['Average Fitness']-frontin.iloc[i+1]['Average Fitness'])
            distance1 = mh.sqrt((frontin.iloc[i]['Cost Fitness']-frontin.iloc[i+1]['Cost Fitness'])**2+(frontin.iloc[i]['Score Fitness']-frontin.iloc[i+1]['Score Fitness'])**2)#(frontin.iloc[i-1]['Average Fitness']-frontin.iloc[i]['Average Fitness'])
            if distance2<distance1:

                RandomSampleFront = frontin.iloc[i+1]  # Paretofront5.sample(n=1)
                IndexofSample5 = RandomSampleFront.name  # index[0]
                RandomMutateSample = MutatePop.iloc[0]
                IndexofRandomMutate = RandomMutateSample.name  # ix[0]
                frontin = frontin.drop(IndexofSample5)
                MutatePop = MutatePop.drop(IndexofRandomMutate)
                frontin = frontin.append(RandomMutateSample)
                MutatePop = MutatePop.append(RandomSampleFront)
                frontin = frontin.reset_index(drop=True)
                MutatePop = MutatePop.reset_index(drop=True)
                Paretofront5=frontin


print('Front 1')
print(Paretofront1.head())
print(len(Paretofront1))

print('Front 2')
print(Paretofront2.head())
print(len(Paretofront2))

print('Front 3')
print(Paretofront3.head())

print(len(Paretofront3))
print('Front 4')
print(Paretofront4.head())
print(len(Paretofront4))

print('Front 5')
print(Paretofront5.head())
print(len(Paretofront5))


##### FINAL SORTING ########



##### PREPARE FOR PLOTTING #######
ax1=Paretofront1['Cost Fitness'].tolist()
ax2=Paretofront2['Cost Fitness'].tolist()
ax3=Paretofront3['Cost Fitness'].tolist()
ax4=Paretofront4['Cost Fitness'].tolist()
ax5=Paretofront5['Cost Fitness'].tolist()

ay1=Paretofront1['Score Fitness'].tolist()
ay2=Paretofront2['Score Fitness'].tolist()
ay3=Paretofront3['Score Fitness'].tolist()
ay4=Paretofront4['Score Fitness'].tolist()
ay5=Paretofront5['Score Fitness'].tolist()

fig=plt.figure()

plt.plot(ax5,ay5,'8',label='5')
plt.plot(ax4,ay4,'>',label='4')
plt.plot(ax3,ay3,'<',label='3')
plt.plot(ax2,ay2,'o',label='2')
plt.plot(ax1,ay1,'x',label='1')

plt.legend(loc='best')
plt.title('Multi Objective - Trial 1 - Realistic-nrp-m4',fontsize=16)
#plt.title('Multi Objective - Equal weighting between fitnesses - Realistic:nrp-m4',fontsize=16)
#plt.title('Multi Objective - Classic - nrp1',fontsize=16)
plt.xlabel('Objective 1 - Cost Fitness',fontsize=16)
plt.ylabel('Objective 2 - Score/Fitness',fontsize=16)
plt.savefig('Multi Objective - Realistic - trial 1.pdf')
#plt.savefig('Multi Objective - equal - Large 4.pdf')

plt.show()
print('')
print('')
##########