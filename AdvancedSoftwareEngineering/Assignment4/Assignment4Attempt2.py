from scipy.io import arff
import pandas as pd
from deap import base
from deap import creator as Creator
from deap import tools
from deap import algorithms
from deap import gp as GenProg
import operator
from sklearn.metrics import mean_squared_error,r2_score
import numpy as np
import matplotlib as mp
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import math
from sklearn import preprocessing, cross_validation, svm,metrics





####################################
# YOU GET TO CHOSE WHICH ONE OF THESE TWO GETS USED FOR MODELLING
data = open('kemerer.arff')
#data = open('albrecht.arff')
trialnumber = 3
#datafilename='albrecht'
datafilename='kemerer'

#####################################


class TheDataFile:  # KEEP
    def LoadTheDataFile(self):
        data1 = arff.loadarff(data)
        DataFile = pd.DataFrame(data1[0])

        Model = DataFile[DataFile.columns[:-1]]
        print(Model)
        print('')

        Efforts = DataFile[DataFile.columns[-1]]
        print(Efforts)
        print('End of file loading')
        print('')
        print('')
        print(len(DataFile))
        return DataFile, Model, Efforts


class UsingDeap:

    def StraightFromDeapTreeCreation(self):  # MAIN BODY OF WORK, deap stuff was used in this, from their website, parts will look similar
        #https://deap.readthedocs.io/en/master/examples/gp_symbreg.html



        #########################################
        # YOU CAN CHOSE THE SIZE OF THE POPULATION YOU WANT AND THE NUMBER OF GENERATIONS TO EVOLVE OVER
        NumberOfGenerations=100          #ideally choose higher than 50
        PopulationSize=60               #ideally choose higher than 50
        tournamentsize=3		#ideally choose higher than 2
        usercxpb=0.3            #probability of mating two individuals
        usermutpb=0.2           #probability of mutating an individual
        #########################################





        trainlength = int(len(datafile2) * 0.75)
        traindata = datafile2.iloc[:trainlength]
        traineffort=datafile3.iloc[:trainlength]
        testdata=datafile2.iloc[trainlength:]
        testeffort=datafile3.iloc[trainlength:]

        minimum = len(datafile2.columns)
        maximum = len(datafile2.columns)

        pset = GenProg.PrimitiveSet('main', 7)
        pset.addPrimitive(max, 2)
        pset.addPrimitive(operator.add, 2)
        pset.addPrimitive(operator.mul, 2)
        pset.addPrimitive(operator.sub, 2)
        pset.addPrimitive(min, 2)
        pset.addPrimitive(math.cos, 1)
        pset.addPrimitive(math.sin, 1)

        Creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        Creator.create("Individual", GenProg.PrimitiveTree, fitness=Creator.FitnessMin, pset=pset)
        toolbox = base.Toolbox()

        toolbox.register("expr", GenProg.genFull, pset=pset, min_=minimum, max_=maximum)
        toolbox.register("individual", tools.initIterate, Creator.Individual, toolbox.expr)
        toolbox.register('population', tools.initRepeat, list, toolbox.individual)
        toolbox.register('compile', GenProg.compile,pset=pset)


##################################################

        # THIS IS THE USER DEFINED PART TO EVALUATE THE FITNESS
        # CAN BE CHANGED IF IMPLEMENTED FITNESS ISNT ADEQUATE


        def Evaluation(individual):
            trainlength=int(len(datafile2)*0.75)
            traindata=datafile2.iloc[:trainlength]


            func = toolbox.compile(expr=individual)
            summer1=0
            summer2=0
            fitnessess=[]
            fitnessess2=[]
            for i in range(0,trainlength):
                fitness1 = func(datafile2.iloc[i][0], datafile2.iloc[i][1], datafile2.iloc[i][2], datafile2.iloc[i][3], datafile2.iloc[i][4], datafile2.iloc[i][5], datafile2.iloc[i][6])
                fitness2=datafile3.iloc[i]
                fitnessess.append(fitness1)
                fitnessess2.append(fitness2)
                summer1+=fitness1
                summer2+=fitness2
            RootMeanSquare=math.sqrt(((summer1-summer2)**2))#math.sqrt(mean_squared_error(fitnessess2,fitnessess))
            fitness1=RootMeanSquare
            toolbox.individual().fitness.values=RootMeanSquare,#fitness1,##### not sure how to give it a valid fitness!!!

            return fitness1,
####################################################





        print('Onto creating populations using GP')
        toolbox.register("evaluate", Evaluation)
        toolbox.register("select", tools.selTournament, tournsize=tournamentsize)
        toolbox.register("mate", GenProg.cxOnePoint)
        toolbox.register("expr_mut", GenProg.genFull, min_=minimum, max_=maximum)
        toolbox.register("mutate", GenProg.mutUniform, expr=toolbox.expr_mut, pset=pset)


        hof=tools.HallOfFame(1)
        pop = toolbox.population(n=PopulationSize)
        pop, log = algorithms.eaSimple(pop, toolbox, cxpb=usercxpb, mutpb=usermutpb, ngen=NumberOfGenerations, halloffame=hof)


        poplist=[]
        fitnesses=[]
        for i in range(0,len(pop)):
            fitnessess3 = []
            fitnessess4 = []
            func1 = toolbox.compile(expr=pop[i])
            poplist.append(str(pop[i]))
            datapop=pd.DataFrame(poplist)
            summer1=0
            summer2=0
            for t in range(0,trainlength):
                fitness10 = func1(datafile2.iloc[t][0], datafile2.iloc[t][1], datafile2.iloc[t][2], datafile2.iloc[t][3],datafile2.iloc[t][4], datafile2.iloc[t][5], datafile2.iloc[t][6])
                fitness20=datafile3.iloc[t]
                fitnessess3.append(fitness10)
                fitnessess4.append(fitness20)
                summer1+=fitness10
                summer2+=fitness20
            variability=math.sqrt(((summer1-summer2)**2))#math.sqrt(mean_squared_error(fitnessess4,fitnessess3))
            fitnesses.append(variability)



        print('')
        print('')
        print('Final section')
        print('')
        print(fitnesses)
        print('')
        print(poplist)
        print('')
        print('Printing the dataframe')
        print(datapop.head())
        print('')

        datapop['Fitnesses']=pd.Series(fitnesses)
        datapop=datapop.sort_values('Fitnesses',ascending=True)
        print(datapop)
        print('')

        #now apply trained model to remaining data
        print('Best model from GP-ing')
        print(datapop.iloc[0][0])
        print(trainlength)
        print(len(datafile2))



        def TestingDeapModel(self):
            print('')
            print('')
            print('Testing Model')
            print(str(pop[0]))
            functiontest=toolbox.compile(expr=pop[0])

            summer10 = 0
            summer20 = 0
            fitnessess5=[]
            fitnessess6=[]
            for t in range(0, len(testdata)):
                fitness101 = functiontest(testdata.iloc[t][0], testdata.iloc[t][1], testdata.iloc[t][2],testdata.iloc[t][3],testdata.iloc[t][4], testdata.iloc[t][5], testdata.iloc[t][6])
                fitness202 = testeffort.iloc[t]
                fitnessess5.append(fitness101)
                fitnessess6.append(fitness202)
                summer10 += fitness101
                summer20 += fitness202
            print('')
            return fitnessess5,fitnessess6

        def TrainingData(self):
            print('')
            print('')
            print('Testing Model')
            print(str(pop[0]))
            functiontest = toolbox.compile(expr=pop[0])

            summer10 = 0
            summer20 = 0
            fitnessess8 = []
            fitnessess9 = []
            for t in range(0, len(traindata)):
                fitness101 = functiontest(traindata.iloc[t][0], traindata.iloc[t][1], traindata.iloc[t][2],
                                          traindata.iloc[t][3], traindata.iloc[t][4], traindata.iloc[t][5],
                                          traindata.iloc[t][6])

                fitness202 = traineffort.iloc[t]
                fitnessess8.append(fitness101)
                fitnessess9.append(fitness202)
                summer10 += fitness101
                summer20 += fitness202

            print('')
            return fitnessess8, fitnessess9



        ##############################################################
        ##############################################################
        ####### THIS IS JUST THE APPLICATION OF THE TREE MODEL #######
        ##############################################################
        ##############################################################


        testingmodel=TestingDeapModel(pop)
        print(testingmodel[0])
        print(testingmodel[1])
        actualeffort=testingmodel[1]
        print(actualeffort)
        predictedefforts=testingmodel[0]
        trainmodel=TrainingData(pop)
        print('train model')
        print(trainmodel)
        actualtraineffort=trainmodel[1]
        print(actualtraineffort)
        predictedtrainefforts=trainmodel[0]

        xaxis=np.arange(0,len(testdata))
        xaxis2=np.arange(0,len(traindata))


        fig1=plt.figure(figsize=(10,6))
        ax1=fig1.add_subplot(211)
        ax1.plot(xaxis,actualeffort,label='Actual test data effort')
        ax1.legend(loc='upper right', fontsize=10)
        plt.title('Testing Data')

        ax2=ax1.twinx()
        ax2.plot(xaxis,predictedefforts,'r-',label='Predicted test data efforts')
        ax2.legend(loc='upper left', fontsize=10)

        ax3=fig1.add_subplot(212)
        ax3.plot(xaxis2,actualtraineffort, label='Actual train data effort')
        ax3.legend(loc='upper right', fontsize=10)
        plt.title('Training Data')

        ax4=ax3.twinx()
        ax4.plot(xaxis2,predictedtrainefforts, 'r-', label='Predicted train data effort')
        ax4.legend(loc='upper left', fontsize=10)

        #fig1.savefig('GP trial ' + str(trialnumber) + ', file '+datafilename+'.pdf')  # Plotting each year, percentage change of volume and market capital first 30 days ' + str(dflist[placer][0]) + '.pdf')



        t=r2_score(actualtraineffort,predictedtrainefforts)
        print('r squared values seems to be legit, this is the variance',t)
        q=np.mean(predictedtrainefforts)
        r=np.mean(actualtraineffort)
        u=r-q
        print('difference of means overall', u)

        print('The predictedtrainefforts is the predicted effort values the model has placed on the training data')
        print('The predictedefforts is the predicted effort values of the model applied to the test sets')



##################################################
        # THIS SECTION IS USED TO EVALUATE THE TREE MODEL BY GETTING MEANS AS REQUESTED IN THE ASSIGNMENT REQUIREMENTS STUFF

        print('')
        print('---------------------------------')
        print('Write this part down')
        print('')

        yy=np.mean(predictedefforts)
        ii=np.mean(actualeffort)
        var1=r2_score(actualeffort,predictedefforts)
        print('tested stuff')
        print('predicted effort mean: ',yy,', actual mean effort: ',ii)
        print('Mean difference',yy-ii)
        print('Variance: ', var1)
        print('')


        uu=np.mean(predictedtrainefforts)
        tt=np.mean(actualtraineffort)
        var2=r2_score(predictedtrainefforts,actualtraineffort)
        print('training stuff')
        print('predicted train effort mean: ',uu,', actual train efforts: ',tt)
        print('Mean difference: ',uu-tt)
        print('Variance: ', var2)
        print('---------------------------------')

        plt.show()


class UsingLinearRegression:

    def LinearRegressioning(self):
        print('')
        print('')
        print('-----------------------------------')
        print('Linear Regression section beginning')
        print('-----------------------------------')
        print('')
        print('')


        X_train, X_test, y_train, y_test = cross_validation.train_test_split(datafile2,datafile3)

        print('Used to develop/test on a model')
        print(X_train.shape)
        print(y_train.shape)
        print('Used to test the developed model')
        print(X_test.shape)
        print(y_test.shape)
        print('')

        LinReg = LinearRegression()
        training = LinReg.fit(X_train, y_train)
        print('')
        print('Onto modelling predict')
        predciteffort = training.predict(X_test)
        PredictedEffort = pd.Series(predciteffort)
        print(PredictedEffort.head())
        print(len(PredictedEffort))
        print('')

        print('')
        print('---------------------------------')
        print('Write this part down')
        print('')
        print('Coeffs',training.coef_)                                          #http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
        print('Mean squared error',mean_squared_error(y_test,predciteffort))    #http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
        print('Variance', r2_score(y_test,predciteffort))                       #http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
        print('---------------------------------')
        print('')

        xaxis = np.arange(0, len(X_train))
        xaxis2 = np.arange(0, len(X_test))

        testdata=datafile3.iloc[len(X_train):]
        traindata=datafile3.iloc[:len(X_train)]

        fig1 = plt.figure(figsize=(10, 6))
        ax1 = fig1.add_subplot(111)
        ax1.plot(xaxis2, testdata, label='Actual test data effort')
        ax1.legend(loc='upper right', fontsize=10)
        plt.title('Testing Data')

        ax2 = ax1.twinx()
        ax2.plot(xaxis2, PredictedEffort, 'r-', label='Predicted test data efforts')
        ax2.legend(loc='upper left', fontsize=10)

        #fig1.savefig('Linear Regression trial ' + str(trialnumber) + ', file '+datafilename+'.pdf')  # Plotting each year, percentage change of volume and market capital first 30 days ' + str(dflist[placer][0]) + '.pdf')





        plt.show()


objective1 = TheDataFile()
datafile1, datafile2, datafile3 = (objective1.LoadTheDataFile())
objective2 = UsingDeap()
treeindividualmodelling = objective2.StraightFromDeapTreeCreation()  # This one is getting used
objective3=UsingLinearRegression()
objective3.LinearRegressioning()