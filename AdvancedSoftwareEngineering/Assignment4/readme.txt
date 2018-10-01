Made using pycharm using python 3.5

imports needed:
scipy
pandas
deap
operator
sklearn
numpy
matplotlib
math


If error about python cannot evaluate a tree high than 90, just run the code again. The files to choose from can be choosen however you want under the import section:


####################################
# YOU GET TO CHOSE WHICH ONE OF THESE TWO GETS USED FOR MODELLING
data = open('kemerer.arff')
#data = open('albrecht.arff')
trialnumber = 3
#datafilename='albrecht'
datafilename='kemerer'

#####################################



Also there are parameters that you can change if you want in the class UsingDeap, in the def StraightFromDeapTreeCreation(self):






        #########################################
        # YOU CAN CHOSE THE SIZE OF THE POPULATION YOU WANT AND THE NUMBER OF GENERATIONS TO EVOLVE OVER
        NumberOfGenerations=100          #ideally choose higher than 50
        PopulationSize=60               #ideally choose higher than 50
        tournamentsize=3		#ideally choose higher than 2
        usercxpb=0.3            #probability of mating two individuals
        usermutpb=0.2           #probability of mutating an individual
        #########################################



saving plots have been # so no graphs will get saved.