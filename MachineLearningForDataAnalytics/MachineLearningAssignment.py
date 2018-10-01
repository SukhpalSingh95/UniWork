import pandas as pd
#import matplotlib as plt
import geopy as geopy
import matplotlib.pyplot as plt
from geopy.exc import GeocoderTimedOut
import sklearn as sklearn
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, cross_validation


import numpy as np
#import ggplot as ggplt
#import scipy as scipy
#import collections as cllctns
#import seaborn as sbrn
#import datetime
#import random
#import image
#import geos
#import mpl_toolkits.basemap as Basemap
#import cartopy
import cartopy.crs as ccrs


#CHANGE THE STUFF INSIDE " ", OTHERWISE THIS IS UNIVERSAL
OpenCSVFile=pd.DataFrame.from_csv("flavors_of_cacao.csv", index_col=None)
#ChocolateCSVFile=open("/Users/sukhpalsingh/PycharmProjects/MachineLearningForDataAnalytics/flavors_of_cacao.csv", 'r')



#CLASS IS UNIVERSAL - DOES NOT NEED TO BE CHANGED FOR OTHER TYPES OF DATASETS
class InitialInvestigation:
    print('-----------------------------------------')
    print('-----------------------------------------')
    print('-----------------------------------------')
    print('STEP 1 - INITIAL INVESTIGATION!!!')
    print('')
    print('Top 5 rows')
    print('')
    print(OpenCSVFile.head())
    print('')
    #print('Not sure what this shows')
    #print('')
    #print(OpenCSVFile.head().all())
    #print('')
    print('Shows a description of the dataset')
    print('')
    print(OpenCSVFile.describe())
    print('')
    print('Shows a information on the dataset')
    print('')
    print(OpenCSVFile.info())
    print('')

    #THIS IS ONLY USEFUL FOR THE ASSIGNMENT, NEED TO COMMENT OUT THE DECLARATION IF REUSED
    def MostSuccessfulChocolate(self):
        # USED TO COUNT THE OCCURENCES OF EACH bean
        print('-----------------------------------------------------------------------------')
        print('-----------------------------------------------------------------------------')
        print('Dataframe ordered by rating')
        DataSet=OpenCSVFile.sort_values(['Rating'],ascending=True)
        DataSetRating=DataSet['Rating']
        DataSetChocolate=DataSet['Company \n(Maker-if known)']
        print(DataSet.head())
        print('-----------------------------------------------------------------------------')
        print('-----------------------------------------------------------------------------')
        print('')

    MostSuccessfulChocolate(OpenCSVFile)
    print('END OF SECTION - INITIAL INVESTIGATION!!!')
    print('*****************************************')
    print('*****************************************')
    print('*****************************************')
    print('')
    print('')
    print('')

class DataErrors:
    print('-----------------------------------------')
    print('-----------------------------------------')
    print('-----------------------------------------')
    print('STEP 2 - DATA ERRORS???')
    print('')
    print('Finding the problem columns')
    print('')
    print(OpenCSVFile.isnull().sum())
    print('')


    #FOR FUTURE USE, CHANGE THIS DEPENDING ON YOUR USE CASE - THIS IS EXPLICIT TO THIS ONE ASSIGNMENT
    def DealingWithTheNulls(self):
        ReplaceErroredData1=OpenCSVFile['Bean\nType'].replace('¬†', 'ertihwfdkjhdfkjh')#  THIS DID NOT WORK FOR REMOVING THE INVISIBLE VALUES
        ReplaceErroredData2=OpenCSVFile.fillna(value='¬†', inplace=True)#  THIS DID NOT WORK FOR REMOVING THE INVISIBLE VALUES
        print('')
        print('Replacing error data - attempt 1')
        print(ReplaceErroredData1.head(10))
        print('')
        print('Replacing error data - attempt 2')
        print(ReplaceErroredData2)
        print('')
        print('Replacing the problem value is not working using the methods implemented')
        print(OpenCSVFile.loc[4])   #Print out one individual row to see if it would give something else out, it doesn't
        print('-----------------------------------------')
        print('')
        print('')

    #COMMENT THIS OUT AFTER THE ASSIGNMENT - THIS IS AGAIN EXPLICIT TO THE ASSIGNMENT
    DealingWithTheNulls(OpenCSVFile)




    print('*****************************************')
    print('*****************************************')
    print('*****************************************')
    print('')
    print('')
    print('')


#THIS CLASS NEEDS EDITED IF REUSED
class BasicBargraphPlotting:
    print('-----------------------------------------')
    print('-----------------------------------------')
    print('-----------------------------------------')
    print('STEP 3 - Bar graph plotting')
    print('')
    MostCommon = 5  # int(DatSetCompanyNames.__len__()/30)

    # THIS IS ONLY USEFUL FOR THE ASSIGNMENT, NEED TO COMMENT OUT THE DECLARATION IF REUSED
    def BeanType(self, MostCommon):
        # USED TO COUNT THE OCCURENCES OF EACH bean
        DatSetCompanyNames = OpenCSVFile['Bean\nType'].value_counts()
        # MostCommon=int(DatSetCompanyNames.__len__())
        figure1 = plt.figure(figsize=(10, 5))
        plot1 = figure1.add_subplot(111)
        DatSetCompanyNames[0:MostCommon].plot(kind='bar')
        plt.title('Bean type', fontsize=20)
        plt.xlabel('Bean Name', fontsize=20)
        plt.ylabel('Number of occurences', fontsize=20)
        plt.tight_layout()
        #plt.savefig('Cocoa bean.pdf')  # Plotting each year, percentage change of volume and market capital first 30 days ' + str(dflist[placer][0]) + '.pdf')
        # plt.show()
        plt.close()
        print('Plotted graph 3: Bean type vs number of occurences')
        print('')

    #THIS IS ONLY USEFUL FOR THE ASSIGNMENT, NEED TO COMMENT OUT THE DECLARATION IF REUSED
    def CocoaPercentage(self,MostCommon):
        # USED TO COUNT THE OCCURENCES OF cocoa percentage
        DatSetCompanyNames = OpenCSVFile['Cocoa\nPercent'].value_counts()
        # MostCommon=int(DatSetCompanyNames.__len__())
        figure1 = plt.figure(figsize=(10, 5))
        plot1 = figure1.add_subplot(111)
        DatSetCompanyNames[0:MostCommon].plot(kind='bar')
        plt.title('Chocolate produced based on cocoa percentage', fontsize=20)
        plt.xlabel('Cocoa percentage', fontsize=20)
        plt.ylabel('Number of occurences', fontsize=20)
        plt.tight_layout()
        #plt.savefig('Cocoa percentage.pdf')  # Plotting each year, percentage change of volume and market capital first 30 days ' + str(dflist[placer][0]) + '.pdf')
        # plt.show()
        plt.close()
        print('Plotted graph 2: Cocoa percentage VS number of occurences')
        print('')

    #THIS IS ONLY USEFUL FOR THE ASSIGNMENT, NEED TO COMMENT OUT THE DECLARATION IF REUSED
    def CompanyOccurences(self,MostCommon):
        # USED TO COUNT THE OCCURENCES OF EACH COMPANY
        DatSetCompanyNames = OpenCSVFile['Company \n(Maker-if known)'].value_counts()
        NormalisedDatSetCompanyNames=(DatSetCompanyNames)
        #print(OpenCSVFile.__len__)
        print('')
        figure1 = plt.figure(figsize=(10, 5))
        plot1 = figure1.add_subplot(111)
        DatSetCompanyNames[0:MostCommon].plot(kind='bar')
        plt.title('Companies with the most amount of products on shelfs', fontsize=20)
        plt.xlabel('Company names', fontsize=20)
        plt.ylabel('Number of occurences', fontsize=20)
        plt.tight_layout()
        #plt.savefig('Plot companies.pdf')  # Plotting each year, percentage change of volume and market capital first 30 days ' + str(dflist[placer][0]) + '.pdf')
        #plt.show()
        plt.close()
        print('Plotted graph 1: Companies VS number of occurences')
        print('')






    #THIS IS ONLY USEFUL FOR THE ASSIGNMENT, NEED TO COMMENT OUT THE DECLARATION IF REUSED
    def BeanOriginRecurrence(self,MostCommon):
        # USED TO COUNT THE OCCURENCES OF EACH bean
        DatSetCompanyNames=OpenCSVFile['Broad Bean\nOrigin'].value_counts()
        #MostCommon=int(DatSetCompanyNames.__len__())
        figure1=plt.figure(figsize=(10,5))
        plot1=figure1.add_subplot(111)
        DatSetCompanyNames[0:MostCommon].plot(kind='bar')
        plt.title('Bean origins', fontsize=20)
        plt.xlabel('Origin location', fontsize=20)
        plt.ylabel('Number of occurences', fontsize=20)
        plt.tight_layout()
        plt.savefig('Bean origins.pdf')  # Plotting each year, percentage change of volume and market capital first 30 days ' + str(dflist[placer][0]) + '.pdf')
        #plt.show()
        plt.close()
        print('Plotted graph 4: Bean origins VS number of occurences')
        print('')

    #THIS IS ONLY USEFUL FOR THE ASSIGNMENT, NEED TO COMMENT OUT THE DECLARATION IF REUSED
    def ChocolateCompanyLocation(self,MostCommon):
        # USED TO COUNT THE OCCURENCES OF EACH bean
        DatSetCompanyNames = OpenCSVFile['Company\nLocation'].value_counts()
        #MostCommon = int(DatSetCompanyNames.__len__())
        figure1 = plt.figure(figsize=(10, 5))
        plot1 = figure1.add_subplot(111)
        DatSetCompanyNames[0:MostCommon].plot(kind='bar')
        plt.title('Chocolate company hotspots', fontsize=20)
        plt.xlabel('Company location', fontsize=20)
        plt.ylabel('Number of occurrences', fontsize=20)
        plt.tight_layout()
        plt.savefig('Chocolate company hotspots.pdf')  # Plotting each year, percentage change of volume and market capital first 30 days ' + str(dflist[placer][0]) + '.pdf')
        #plt.show()
        plt.close()
        print('Plotted graph 5: Chocolate company location VS number of occurrences')
        print('')
        print('-----------------------------------------')


    #THIS IS ONLY USEFUL FOR THE ASSIGNMENT, THESE ARE THE COMMENTS THAT NEED DELETED WHEN USING ON ANOTHER DATATSET
    CompanyOccurences(OpenCSVFile, MostCommon)
    CocoaPercentage(OpenCSVFile,MostCommon)
    BeanType(OpenCSVFile, MostCommon)
    BeanOriginRecurrence(OpenCSVFile,MostCommon)
    ChocolateCompanyLocation(OpenCSVFile, MostCommon)


    print('*****************************************')
    print('*****************************************')
    print('*****************************************')
    print('')
    print('')
    print('')


class ClusteringRatingCocoaPercentage:

    print('-----------------------------------------')
    print('-----------------------------------------')
    print('-----------------------------------------')
    print('Step 4 - Machine learning methods')
    print('')
    print('Rating head')
    Rating=OpenCSVFile['Rating']
    print(Rating.head())
    print('')
    print('Percentage head')
    Percentage=OpenCSVFile['Cocoa\nPercent']
    print(Percentage.head())


    plt.scatter(Percentage, Rating)
#    plt.show()
    plt.close()
    DataframeForTraining=OpenCSVFile[['Company \n(Maker-if known)','Cocoa\nPercent', 'Company\nLocation', 'Broad Bean\nOrigin']]
    DataframeForXaxis=OpenCSVFile[['Cocoa\nPercent']]
    #DataframeForTraining=DataframeForTraining.convert_objects(convert_numeric=True)        Does not work.
    print(DataframeForTraining.head())

    #DataframeForTraining.to_numeric()
    print('')

    DataframeToPredict=OpenCSVFile[['Rating']]
    print(DataframeToPredict.head())
    print('')


    #creating the numerical format for my data
    print('')
    print('-----------------------------------------')
    print('Label Encoding')
    print('-----------------------------------------')
    Labelencoder1=preprocessing.LabelEncoder()
    Labelencoder2=preprocessing.LabelEncoder()
    Labelencoder3=preprocessing.LabelEncoder()
    Labelencoder4=preprocessing.LabelEncoder()
    Labelencoder5=preprocessing.LabelEncoder()
    Labelencoder6=preprocessing.LabelEncoder()


    #DataframeForTrainingMain=Labelencoder1.fit_transform(DataframeForTraining[['Company \n(Maker-if known)','Cocoa\nPercent','Company\nLocation','Broad Bean\nOrigin']])#.apply(Labelencoder1.fit_transform)
    #DataframeForTraining1=DataframeForTraining[['Company \n(Maker-if known)']].apply(Labelencoder1.fit_transform)
    #DataframeForTraining2=DataframeForTraining[['Cocoa\nPercent']].apply(Labelencoder2.fit_transform)
    #DataframeForTraining3=DataframeForTraining[['Company\nLocation']].apply(Labelencoder3.fit_transform)
    #DataframeForTraining4=DataframeForTraining[['Broad Bean\nOrigin']].apply(Labelencoder4.fit_transform)
    print('')
    print('Main dataframe')
    #DataframeForTrainingMain=pd.concat([DataframeForTraining1,DataframeForTraining2,DataframeForTraining3,DataframeForTraining4],axis=1)
    print('')
    #DataframeForTraining2['Cocoa\nPercent']=DataframeForTraining['Cocoa\nPercent'].to_numeric()
    #print(DataframeForTraining1.head())


    DataframeForTrainingMain=DataframeForTraining.apply(Labelencoder1.fit_transform)
    print('Machine learning, train and prediction part')
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(DataframeForTrainingMain, DataframeToPredict)



    print(DataframeForTrainingMain.head())
    print('')
    print('')

    print('X_train')
    print(X_train.head())
    print('')
    print('')
    print('X_test')
    print(X_test.head())
    print('')
    print('')
    print('y_train')
    print(y_train.head())
    print('')
    print('')
    print('y_test')
    print(y_test.head())
    print('')

    print('')
    print('Used to train on a model')
    print('')
    print('X_train shape')
    print(X_train.head())

    print('y_train shape')
    print(y_train.shape)
    print('')
    print('Used to test the developed model')
    print('X_test shape')
    print(X_test.head())


    X_test_tolist=X_test.index.tolist()
    X_test_tolist_toCocoaPercentage=DataframeForTraining['Cocoa\nPercent'].loc[X_test_tolist]
    print(X_test_tolist_toCocoaPercentage.head())

    print('y_test shape')
    print(y_test.shape)
    print('')




    LinReg = LinearRegression()
    training = LinReg.fit(X_train, y_train)

    print('Making a prediction')
    PredictRating = training.predict(X_test)
    PredictedRating=pd.DataFrame(PredictRating, columns=['Predictions'])
    print(PredictedRating.head())
    print('')


    print("X_test")
    print(X_test.head())

    print('')
    print('Predicted rating')
    print(len(PredictedRating))
    print(PredictedRating.size)
    print(PredictedRating.head())
    print('')
    print('Test data')
    print(len(y_test))
    print(y_test.size)
    print(y_test.head())
    print('')
    #r=Labelencoder1.inverse_transform(X_test)
    #p=pd.DataFrame(r)
    #print(p.head())

    X_test_tolist_toCocoaPercentageSeries=pd.Series(X_test_tolist_toCocoaPercentage)
    PredictedRating.reset_index(drop=True,inplace=True)
    y_test.reset_index(drop=True,inplace=True)
    X_test_tolist_toCocoaPercentageSeries.reset_index(drop=True,inplace=True)

    TestPredictedCocoa=pd.concat([y_test,PredictedRating, X_test_tolist_toCocoaPercentageSeries],axis=1)
    TestPredictedCocoaSorted=TestPredictedCocoa.sort_values(['Cocoa\nPercent'])


    print(TestPredictedCocoaSorted.head())
    #plotting predictions
    yaxis=list(range(0,len(PredictedRating)))

    plt.close()
    figure1 = plt.figure(figsize=(10,5))
    ax1=figure1.subplots()
    #ax1.scatter(,y_test)

    ax1.plot(yaxis,TestPredictedCocoaSorted['Rating'], 'b.')
    ax1.set_ylabel('Actual', color='b')
    ax1.tick_params('y', colors='b')

    ax2=ax1.twinx()
    ax2.plot(yaxis,TestPredictedCocoaSorted['Predictions'],'y.')
    ax2.set_ylabel('Predicted', color='y')
    ax2.tick_params('y', colors='y')

    NumberOfTicks=20
    print('List length '+str(len(TestPredictedCocoaSorted))+' length of list divided '+str(len(TestPredictedCocoaSorted)/NumberOfTicks))
    values=TestPredictedCocoaSorted['Cocoa\nPercent'].tolist()
    Values=values[0::NumberOfTicks]
    print(values[0::NumberOfTicks])
    plt.xticks(yaxis, Values)

    ax1.xaxis.set_major_locator(plt.MaxNLocator(len(TestPredictedCocoaSorted)/NumberOfTicks))

    ax2.xaxis.set_major_locator(plt.MaxNLocator(len(TestPredictedCocoaSorted)/NumberOfTicks))
    plt.xticks(rotation=45)
    #plt.savefig('MachineLearningGraph.pdf')  # Plotting each year, percentage change of volume and market capital first 30 days ' + str(dflist[placer][0]) + '.pdf')




    figure2 = plt.figure(figsize=(10, 5))
    ax3 = figure2.subplots()
    # ax1.scatter(,y_test)

    ax3.plot(yaxis, TestPredictedCocoaSorted['Rating'], 'b.')
    ax3.set_ylabel('Actual', color='b')
    ax3.tick_params('y', colors='b')

    ax4 = ax3.twinx()
    ax4.plot(yaxis, TestPredictedCocoaSorted['Predictions'], 'y.')
    ax4.set_ylabel('Predicted', color='y')
    ax4.tick_params('y', colors='y')

    NumberOfTicks = 20
    print('List length ' + str(len(TestPredictedCocoaSorted)) + ' length of list divided ' + str(
        len(TestPredictedCocoaSorted) / NumberOfTicks))
    values = TestPredictedCocoaSorted['Cocoa\nPercent'].tolist()
    Values = values[0::NumberOfTicks]
    print(values[0::NumberOfTicks])
    plt.xticks(yaxis, Values)

    ax3.xaxis.set_major_locator(plt.MaxNLocator(len(TestPredictedCocoaSorted) / NumberOfTicks))
    ax4.set_ylim([1, 5])
    ax3.set_ylim([1, 5])

    ax4.xaxis.set_major_locator(plt.MaxNLocator(len(TestPredictedCocoaSorted) / NumberOfTicks))
    plt.xticks(rotation=45)
    #plt.savefig('MachineLearningGraphWithYlims.pdf')  # Plotting each year, percentage change of volume and market capital first 30 days ' + str(dflist[placer][0]) + '.pdf')


    print('')

    print('*****************************************')
    print('*****************************************')
    print('*****************************************')
    print('')
    print('')
    print('')








class MultiVariableAnalysis:
    print('-----------------------------------------')
    print('-----------------------------------------')
    print('-----------------------------------------')
    print('STEP 5 - Cross checking variables')
    print('')
    company_name = []
    company_location = []
    company_occurence=[]
    company_uncertainty = []
    company_rating=[]
    PlayDataframe = OpenCSVFile[['Company \n(Maker-if known)','Company\nLocation','Broad Bean\nOrigin']]
    OpenCSVFile.sort_values(['Company \n(Maker-if known)'], ascending=True)
    DataSetLength = OpenCSVFile.__len__()
    counter=0
    Companyname=""


    for CompanyColumnIndex, CompanyRowEntry in OpenCSVFile.iterrows():
        if CompanyColumnIndex==DataSetLength-1:
            pass
        else:
            if OpenCSVFile['Company \n(Maker-if known)'].iloc[CompanyColumnIndex] !=OpenCSVFile['Company \n(Maker-if known)'].iloc[CompanyColumnIndex + 1]:
                counter =0

            if OpenCSVFile['Company \n(Maker-if known)'].iloc[CompanyColumnIndex]==OpenCSVFile['Company \n(Maker-if known)'].iloc[CompanyColumnIndex+1]:
                counter+=1
                Companyname=OpenCSVFile['Company \n(Maker-if known)'].iloc[CompanyColumnIndex]
                Companyuncertainty=OpenCSVFile['Broad Bean\nOrigin'].iloc[CompanyColumnIndex]
                Companylocation=OpenCSVFile['Company\nLocation'].iloc[CompanyColumnIndex]
                Companyrating=OpenCSVFile['Rating'].iloc[CompanyColumnIndex]


        company_uncertainty.append(Companyuncertainty)
        company_location.append(Companylocation)
        company_name.append(Companyname)
        company_occurence.append(counter)
        company_rating.append(Companyrating)

    print('Company name array length:' + str(company_name.__len__()))
    #print(company_name)
    print('')
    print('Company occurence array length: '+str(company_occurence.__len__()))
    #print(company_occurence)
    print('')
    print('Company location array length: '+ str(company_location.__len__()))
    #print(company_location)
    print('')
    print('Company uncertainty array length: '+ str(company_uncertainty.__len__()))
    #print(company_uncertainty)
    print('')
    print('Company rating array length: '+str(company_rating.__len__()))
    #print(company_rating)
    print('')
    print('The point to this section is that everything has been sorted by order of the company. This gives a high degree of freedom for exploring the dataset further.')

    compressedCompanyName=[]
    compressedCompanyRating=[]

    #Can't remember what's going on in here!!!!!
    for i in range(0,len(company_name)-1):
        companynameold=company_name[i]

        index_current=i
        index_firstseen=0
        if company_name[i]!=company_name[i+1]:
            compressedCompanyName.append(companynameold)
            companynameold=company_name[i]
            index_current=2
        if company_name[i]==company_name[i+1]:
            pass
    #print(compressedCompanyName)




    print('*****************************************')
    print('*****************************************')
    print('*****************************************')
    print('')
    print('')
    print('')


#class Global:
    #print('Step 5 - Failed Geolocation visualisation')
    #print('')
    # LocationGeopy=geopy.Bing()
    # Creating a pandas series to hold the latitude and longitudinal information
    # OpenCSVFile['CompanyLocationGeopy']=OpenCSVFile['Company\nLocation'].apply(Location.geocode,timeout=None) #the timeout is required so that it actually goes through the dataset and not stop when doing the conversion.
    #print('')
    #print('sajdhasjkdh')
    #print('')
    #Location = []
    # counter =0
    #LengthofDataframe = len(OpenCSVFile)
    #OpenCSVFile2 = OpenCSVFile.ix[:0]  # (LengthofDataframe/4)]
    #print('Length of dataframe now ' + str(len(OpenCSVFile2)))
    #value = 60

    # OpenCSVFile2['CompanyLocationGeopy']=OpenCSVFile2['Company\nLocation'].apply(LocationGeopy.geocode,timeout=10) #the timeout is required so that it actually goes through the dataset and not stop when doing the conversion.

    # for i in OpenCSVFile['Company\nLocation']:
    #    if counter<value:
    #        print(i)
    ##        t=LocationGeopy.geocode(i, timeout=1)#LocationGeopy.geocode(i, timeout=10)
    #        Location.append(t)
    #        counter+=1
    # Location.append(t)
    # Location.append((OpenCSVFile['Company\nLocation'][i]))
    # print(Location)
    # print(Location)

    # print('Company location after given to geopy')
    # print(Seddd.address)
    # print('')
    # print('Company location given by geopy, latitude and longitude')
    # print((Seddd.latitude, Seddd.longitude))
    # SedddXla=Seddd.latitude
    # SedddYlo=Seddd.longitude

    # ax = plt.axes(projection=ccrs.Mollweide())
    # ax.figure(figsize=(5.91496607667, 3))
    # ax.stock_img()
    # plt.plot(SedddXla,SedddYlo,color='blue', marker='o', transform=ccrs.Geodetic())
    # ax.coastlines(resolution='110m')
    # ax.gridlines()
    # plt.show()
    # plt.close()






#Step1=InitialInvestigation()
#Step2=DataErrors()
#Step3=BasicBargraphPlotting()
#Step4=MultiVariableAnalysis()