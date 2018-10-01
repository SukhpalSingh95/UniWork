import random as rm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import *
#import evoalgos

print('')
print('-----------------------')
print('Importing the text file')
print('-----------------------')
print('')
print('')
DataSetIn = pd.read_csv('nrp1.txt', header=None, index_col=None)#, index=['Profit Of customer', 'Number of requests','Requirements list']) #sep=' ')#,comment='#')  # https://stackoverflow.com/questions/21546739/load-data-from-txt-with-pandas + https://stackoverflow.com/questions/28200404/pandas-read-table-use-first-column-as-index#
#DataSetIn = pd.read_csv('nrp-m4.txt', header=None, index_col=None)#, index=['Profit Of customer', 'Number of requests','Requirements list']) #sep=' ')#,comment='#')  # https://stackoverflow.com/questions/21546739/load-data-from-txt-with-pandas + https://stackoverflow.com/questions/28200404/pandas-read-table-use-first-column-as-index

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
        #self.DataframeForDays.columns=['List of requirements in order','Weight multiplied by value']
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
        print('Evaluating the Cost requirement')
        print('')
        self.DataframeForDays2=self.DataframeForDays2.drop([0],axis=0)  #NEED TO REMOVE ZERO TERM OTHERWISE DIVIDING BY ZERO

        print(self.DataframeForDays2.head())
        print('')
        self.CostFitness=[]
        for i in self.DataframeForDays2['Req Num']:
            self.CostFitness.append((1.0/float(i)))
            #print(i)

        self.SeriesCostFitness=pd.Series(self.CostFitness)
        self.SeriesCostFitness=self.SeriesCostFitness.rename('Cost Fitness',inplace=True)
        #print(self.SeriesCostFitness.head())
        self.DataframeForDays2=self.DataframeForDays2.reset_index(drop=True)        #https://stackoverflow.com/questions/20490274/how-to-reset-index-in-a-pandas-data-frame
        self.DataframeForDays3=pd.concat([self.DataframeForDays2,self.SeriesCostFitness],axis=1)
        print(self.DataframeForDays3.head())


    def AverageFitness(self):
        self.AverageFitness1=[]
        self.AverageFitness2=[]
        for index, row in self.DataframeForDays3.iterrows():
            self.AverageFitness1.append(self.DataframeForDays3['Score/Fitness'][index]) #gets fitnesses from score
            self.AverageFitness2.append(self.DataframeForDays3['Cost Fitness'][index])  #gets fitnesses from cost

        self.AverageFitness3 = [((a + b)/2) for a, b in zip(self.AverageFitness1, self.AverageFitness2)]

        #print(self.AverageFitness3)
        print('')
        self.SeriesAverageFitness3=pd.Series(self.AverageFitness3)
        self.SeriesAverageFitness3=self.SeriesAverageFitness3.rename('Average Fitness',inplace=True)
        self.DataframeForDays4=pd.concat([self.DataframeForDays3, self.SeriesAverageFitness3],axis=1)
        print(self.DataframeForDays4.head())



            ################### NOW I NEED TO STICK EVERYTHING TOGETHER ##################

    def CreateRandomPopulation(self):
        print('Sampled the population')
        self.PopulationSize=len(self.DataframeForDays1)-1
        self.RandomPopulationSample=self.DataframeForDays4.sample(n=self.PopulationSize)#.astype(np.float64)
        print('')
        return self.RandomPopulationSample



    def CreatingTheParetoFront(self,GenerationalSampledPopulation):
        #print('')
        print('Creating the pareto fronts')
        #print('')
        #print(populations)
        self.InputPopulation=GenerationalSampledPopulation
        self.Objective1Population=[]            #keep customers happy
        self.Objective2Population=[]            #keep cost low
        self.WeightingValue=1
        for indexes, rows in self.InputPopulation.iterrows():
            #print(indexes)
            if rows[1]>self.WeightingValue*rows[3]:
                self.Objective1Population.append([rows[0],rows[1],rows[2],rows[3]])
            if rows[2] > self.WeightingValue*rows[3]:
                # print(rows[1])
                self.Objective2Population.append([rows[0],rows[1],rows[2],rows[3]])

        self.obj1=pd.DataFrame(self.Objective1Population,columns=['Req Num','Score/Fitness','Cost Fitness','Average Fitness'])  #keep customers happy
        self.obj2=pd.DataFrame(self.Objective2Population,columns=['Req Num','Score/Fitness','Cost Fitness','Average Fitness'])  #keep costs low

        #print('Population in favour of customer happiness')
        #print(self.obj1.head())
        #print('')
        #print('Population in favour of cutting costs')
        #print(self.obj2.head())

        return self.obj2#, self.obj2





populations=[]
Step1=MakingSenseOfTheDataFile()
Step1
Step1.DataFrameTypeFormatting()
Step1.FindingRepeatingFunctionsValues()
Step1.EvaluteCostRrequirements()
Step1.AverageFitness()
#Step1.CreateRandomPopulation()
print(' ')


Generation=0
#PopulationSize=10
NumberOfGenerations=1
for i in range(0,NumberOfGenerations):
    populations.append(Step1.CreateRandomPopulation())
    Generation+=1
    GenerationalSampledPopulation=pd.concat(populations)


ObjectivesLists=pd.DataFrame(Step1.CreatingTheParetoFront(GenerationalSampledPopulation))
print(ObjectivesLists)
ycomponents=ObjectivesLists['Score/Fitness'].tolist()
xcomponents=ObjectivesLists['Cost Fitness'].tolist()
print(xcomponents)
print(ycomponents)
plt.plot(xcomponents,ycomponents,'x')
plt.xlabel('Objective 1 - Cost Fitness',fontsize=16)
plt.ylabel('Objective 2 - Score/Fitness',fontsize=16)
plt.title('Random search - Cost minimisation orientated - Realistic-nrp')
#plt.savefig('Random search - Cost minimisation orientated - large.pdf')
plt.grid()
#plt.tight_layout()
plt.show()
