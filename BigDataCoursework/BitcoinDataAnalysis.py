#USING PYTHON 3.5 WITH PYCHARM CE!!!!

import random as rm
import pandas as pd
import numpy as np
import pickle
from matplotlib import pyplot as plt
from itertools import *
from datetime import datetime
import calendar
from dateutil.parser import parse
from matplotlib import gridspec

from sklearn.linear_model import LinearRegression
import warnings
import math, datetime, time

from sklearn import preprocessing, cross_validation, svm,metrics
from sklearn.cluster import MeanShift

import seaborn as sns

import scipy
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cophenet
from scipy.spatial.distance import pdist
from pylab import rcParams
import seaborn as sb
from sklearn.cluster import AgglomerativeClustering, KMeans
import sklearn.metrics as snms
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import scale


print('Loaded data set input:')
print(' ')
DataSetIn=pd.DataFrame.from_csv('bitcoin_price.csv', index_col=None)
DataSetIn1=open('bitcoin_price.csv','r')
#DataSetIn=pd.DataFrame.from_csv('bitcoin_price.csv', index_col=None)
#DataSetIn1=open('bitcoin_price.csv','r')
print(DataSetIn.head())
print('----------------------------------------------------------------------------------------')
print('')
print(' ')

################# SORTING THE DATA IN TERMS OF NUMBER OF PEOPLE STILL LEFT SPEAKING THE LANGUAGE #####################

print('Describing the data: ')
print('')
print(DataSetIn.describe())
print('')
print('')

print('Getting information on the data:')
print('')
print(DataSetIn.info())
print('')
print('')
print('----------------------------------------------------------------------------------------')

########################################################################################################


########################### MAKING A DATE LIST #######################

# PUT THE DATE INTO PROPER FORMAT

FixedDataSet=pd.DataFrame(DataSetIn)

print('')
print('Removing commas from the time column')
print('')
FixedDataSet['Date']=FixedDataSet['Date'].str.replace(',','')
#FirstDateEntry=DataSetIn['Date'].iloc[-1]
#FDE=parse(FirstDateEntry)                                   #https://stackoverflow.com/questions/2265357/parse-date-string-and-change-format
#print(FDE)
print(FixedDataSet.head())
print('')
print(' ')

print('Converting the time into numerical format')
print('')
FixedDataSet['Date']=pd.to_datetime(FixedDataSet['Date'])
#FixedDataSet['Date']=FixedDataSet['Date'].astype(datetime)
print(FixedDataSet.head())
print('')
print('')


print('Removing the commas from the Market Cap column')
print('')
# PUTTING THE MARKET CAPITAL INTO THE DATA FORMAT
FixedDataSet['Market Cap']=FixedDataSet['Market Cap'].str.replace(',','')
print(FixedDataSet.head())
print(' ')
print('')

#PUTTING THE VOLUME INTO THE RIGHT FORMAT
print('Removing the commas from the Volume column')
print('')
FixedDataSet['Volume']=FixedDataSet['Volume'].str.replace(',','')
print(FixedDataSet.tail())
print('')


print('Describing the input data: ')
print('')
print(FixedDataSet.describe())
print('')
print('')
print('Getting information on the data:')
print('')
print(FixedDataSet.info())
print('')
print('')
print('')
print('----------------------------------------------------------------------------------------')
########################################################################################################


########################### SORTING VALUES CHANGE 2 #######################
print('Converting Market Cap into an integer')
print('')
# PUTTING THE MARKET CAPITAL INTO THE DATA FORMAT
FixedDataSet['Market Cap']=pd.to_numeric(FixedDataSet['Market Cap']).astype(np.float64)
print(FixedDataSet.head())
print(' ')
print('')

print('Describing the input data: ')
print('')
print(FixedDataSet.describe())
print('')
print('')

print('Getting information on the data:')
print('')
print(FixedDataSet.info())
print('')
print('')
print('----------------------------------------------------------------------------------------')

########################################################################################################



########################### MAKING A DATE LIST #######################


#PUTTING THE VOLUME INTO THE RIGHT FORMAT
print('Changing the Volume into integer type:')
print('')
FixedDataSet['Volume']=FixedDataSet['Volume'].str.replace(',','')
FixedDataSet['Volume']=FixedDataSet['Volume'].str.replace('-','0').astype(np.float64)
print(FixedDataSet.tail())
print('')

print('Describing the data types: ')
print('')
print(FixedDataSet.describe())
print('')
print('')

print('Getting information on the data:')
print('')
print(FixedDataSet.info())
print('')
print('')

########################################################################################################


print('----------------------------------------------------------------------------------------')
print('----------------------------------------------------------------------------------------')
print('----------------------------------------------------------------------------------------')
print('--------------------------------- ONTO VISULAISATION -----------------------------------')
print('----------------------------------------------------------------------------------------')
print('----------------------------------------------------------------------------------------')
print('----------------------------------------------------------------------------------------')



############# CREATING THE FIRST GRAPH ##########################################################


#################################################################################################
#################################################################################################
######################################## FIGURE 1 ###############################################
#################################################################################################
#################################################################################################




print(' ')

figure1=plt.figure()
DateColumn=pd.Series(FixedDataSet['Date'])
MarketCapitalColumn=pd.Series(FixedDataSet['Market Cap'])
plt.subplot(221)
MarketCapilatalGraph1=plt.plot(DateColumn,MarketCapitalColumn)
plt.title('Market Capital')
plt.subplot(222)
HighLowGraph=plt.plot(DateColumn, FixedDataSet['High'],DateColumn, FixedDataSet['Low'])
plt.title('Highest VS Lowest')
plt.legend(['Highest','Lowest'],loc='upper left')
plt.subplot(224)
OpenCloseGraph=plt.plot(DateColumn, FixedDataSet['Open'],DateColumn, FixedDataSet['Close'])
plt.title('Open VS Close')
plt.legend(['Open','Close'],loc='upper left')
plt.subplot(223)
VolumeGraph=plt.plot(DateColumn, FixedDataSet['Volume'])
plt.title('Volume')
plt.close()
########################################################################################################




#################################################################################################
#################################################################################################
######################################## FIGURE 2 ###############################################
#################################################################################################
#################################################################################################


dflist=[]

print('Explicitly stating the day, month and year for each row')
FixedDataSet['Day']=FixedDataSet['Date'].dt.day #https://stackoverflow.com/questions/28990256/python-pandas-time-series-year-extraction
FixedDataSet['Month']=FixedDataSet['Date'].dt.month
FixedDataSet['Year']=FixedDataSet['Date'].dt.year
print(FixedDataSet.head())
print('')

print('Printing separate years data')
for yeargroup in FixedDataSet.groupby(FixedDataSet['Year']):
    dflist.append(yeargroup)

Length=(len(dflist))
for placer in range(0,Length):
    figure1 = plt.figure(figsize=(15,10))
    DateColumn=(dflist[placer][1]['Date'])
    MarketCapitalColumn= (dflist[placer][1]['Market Cap'])
    VolumeGraph =(dflist[placer][1]['Volume'])
    plot1=figure1.add_subplot(111)                  #http://kitchingroup.cheme.cmu.edu/blog/2013/09/13/Plotting-two-datasets-with-very-different-scales/
    plot1.plot(DateColumn,MarketCapitalColumn,'m',label='Market Capital')
    plot1.legend(loc='upper left',fontsize=16)
    plot1.set_title('Year group: '+ str(dflist[placer][0]),fontsize=20)
    plot1.set_ylabel('Market Capital',fontsize=16)
    plot1.set_xlabel('Date',fontsize=16)

    plot2=plot1.twinx()
    plot2.plot(DateColumn,VolumeGraph,'y',label='Volume')
    plot2.legend(loc='upper right',fontsize=16)
    plot2.set_ylabel('Volume',fontsize=16)
    plot1.tick_params(labelsize=16)
    plot2.tick_params(labelsize=16)
    plt.close()
plt.close()
    #plt.savefig('Plotting each year, Volume and market capital '+str(dflist[placer][0])+'.pdf')



############ PLOTTING THE CLOSE AND START VALUES TOGETHER #############


#################################################################################################
#################################################################################################
######################################## FIGURE 3 ###############################################
#################################################################################################
#################################################################################################



for placer in range(0,Length):
    figure1 = plt.figure(figsize=(15,10))
    DateColumn=(dflist[placer][1]['Date'])
    MarketCapitalColumn= (dflist[placer][1]['Open'])
    VolumeGraph =(dflist[placer][1]['Close'])
    plot1=figure1.add_subplot(111)                  #http://kitchingroup.cheme.cmu.edu/blog/2013/09/13/Plotting-two-datasets-with-very-different-scales/
    plot1.plot(DateColumn,MarketCapitalColumn,'m',label='Open')
    plot1.legend(loc='upper left',fontsize=16)
    plot1.set_title('Year group: '+ str(dflist[placer][0]),fontsize=20)
    plot1.set_ylabel('Open',fontsize=16)
    #plot1.set_ylabel(MarketCapitalColumn)
    plot1.set_xlabel('Date',fontsize=16)
    #plot1.legend('Market Capital')

    plot2=plot1.twinx()
    plot2.plot(DateColumn,VolumeGraph,'y',label='Close')
    plot2.legend(loc='upper right',fontsize=16)
    plot2.set_ylabel('Close',fontsize=16)
    plot1.tick_params(labelsize=16)
    plot2.tick_params(labelsize=16)
    plt.close()
plt.close()
    #plt.legend('Market Capital','Volume')
    #plt.savefig('Plotting each year, open and close '+str(dflist[placer][0])+'.pdf')
#plt.show()
#plt.close()

#print(FixedDataSet.tail())
print('')


############# FIGURING OUT THE RELATIONSHIP BETWEEN MARKET CAPITAL AND VOLUME #######



#################################################################################################
#################################################################################################
######################################## FIGURE 4 ###############################################
#################################################################################################
#################################################################################################



for placer in range(1,Length):
    FluctuationInMC2 = []
    FluctuationInV2 = []

    FluctuationInMC3 = []
    FluctuationInV3 = []
    figure1 = plt.figure(figsize=(15,10))
    yeargrouplength=len(dflist[placer][1])-1
    DateColumn=(dflist[placer][1]['Date'])
    FluctuationInMC1=(dflist[placer][1]['Market Cap'])
    FluctuationInV1=(dflist[placer][1]['Volume'])
    for i in range(0, 10):#len(FluctuationInMC1)-1):

        FluctuationInMC2.append((FluctuationInMC1.iloc[i+1]-FluctuationInMC1.iloc[i])/(FluctuationInMC1.iloc[i])*100)            #the list has gone backwards in time, so the first element is actually the oldest date, and the last is the newest date so index+1 - index
        FluctuationInV2.append((FluctuationInV1.iloc[i+1]-FluctuationInV1.iloc[i])/(FluctuationInV1.iloc[i])*100)

    for t in reversed(FluctuationInV2):
        FluctuationInV3.append(t)
    for z in reversed(FluctuationInMC2):
        FluctuationInMC3.append(z)

    plot1 = plt.subplot(211)  # http://kitchingroup.cheme.cmu.edu/blog/2013/09/13/Plotting-two-datasets-with-very-different-scales/
    plot1.plot(FluctuationInMC3, 'm', label='Change in Market Capital')
    plot1.set_title('Year group: ' + str(dflist[placer][0]), fontsize=20)
    plot1.set_ylabel('Percentage change in Market Capital', fontsize=16)
    plot1.set_xlabel('Day', fontsize=16)

    plot2=plt.subplot(212)
    plot2.plot(FluctuationInV3, 'y', label='Change in Volume')
    plot2.set_ylabel('Percentage change in Volume', fontsize=16)
    plot1.tick_params(labelsize=16)
    plot2.tick_params(labelsize=16)
    plot2.set_xlabel('Day', fontsize=16)
    #plt.gca().invert_xaxis()
    #plt.savefig('Plotting each year, percentage change of volume and market capital first 30 days ' + str(dflist[placer][0]) + '.pdf')
    plt.close()
plt.close()




#################################################################################################
#################################################################################################
######################################## FIGURE 5 ###############################################
#################################################################################################
#################################################################################################





for placer in range(1,Length):
    FluctuationInMC2 = []
    FluctuationInV2 = []

    FluctuationInMC3 = []
    FluctuationInV3 = []


    figure1 = plt.figure(figsize=(15,10))
    yeargrouplength=len(dflist[placer][1])-1

    DateColumn=(dflist[placer][1]['Date'])
    FluctuationInMC1=(dflist[placer][1]['Market Cap'])
    FluctuationInV1=(dflist[placer][1]['Volume'])
    FluctuationInMonth=(dflist[placer][1]['Month'])
    lengthofmonth=len(FluctuationInMonth)


yearlydata=[]
yearreturnmc = []
yearreturnv = []
for indexer in range (0,len(dflist)):
    month1 = []
    month2 = []
    month3 = []
    month4 = []
    month5 = []
    month6 = []
    month7 = []
    month8 = []
    month9 = []
    month10 = []
    month11 = []
    month12 = []



    month1 = []
    month2 = []
    month3 = []
    month4 = []
    month5 = []
    month6 = []
    month7 = []
    month8 = []
    month9 = []
    month10 = []
    month11 = []
    month12 = []


    holder=pd.Series(dflist[indexer][1]['Month'])
    holder2=pd.Series(dflist[indexer][1]['Market Cap'])
    holder4=pd.Series(dflist[indexer][1]['Volume'])
    holder3=pd.concat([holder,holder2,holder4],axis=1)

    yeargroupMC=[]
    yeargroupV=[]
    yeargroup=[]
    yeargroupMC2=[]
    yeargroupV2=[]
    daysinlist=[]
    daysin1=0
    daysin2=0
    y1=pd.DataFrame()

    for i in range (0, len(holder3)):
        if holder3.iloc[i]['Month']==1:
            month1.append(holder3.iloc[i])
            month1 = pd.DataFrame(month1)
            yeargroupMC.append(month1['Market Cap'])  #this returns just the market cap of each day in january, may have to make one for each month
            yeargroupV.append(month1['Volume'])   #this returns the volume for each day in january

        if holder3.iloc[i]['Month'] == 2:
            month2.append(holder3.iloc[i])
            month2 = pd.DataFrame(month2)#.sum()
            yeargroupMC.append(month2['Market Cap'])  #this returns just the market cap of each day in january
            yeargroupV.append(month2['Volume'])   #this returns the volume for each day in january

        if holder3.iloc[i]['Month'] == 3:
            month3.append(holder3.iloc[i])
            month3 = pd.DataFrame(month3)#.sum()
            yeargroupMC.append(month3['Market Cap'])  #this returns just the market cap of each day in january
            yeargroupV.append(month3['Volume'])   #this returns the volume for each day in january

        if holder3.iloc[i]['Month'] == 4:
            month4.append(holder3.iloc[i])
            month4 = pd.DataFrame(month4)#.sum()
            yeargroupMC.append(month4['Market Cap'])  #this returns just the market cap of each day in january
            yeargroupV.append(month4['Volume'])   #this returns the volume for each day in january

        if holder3.iloc[i]['Month'] == 5:
            month5.append(holder3.iloc[i])
            month5 = pd.DataFrame(month5)#.sum()
            yeargroupMC.append(month5['Market Cap'])  #this returns just the market cap of each day in january
            yeargroupV.append(month5['Volume'])   #this returns the volume for each day in january

        if holder3.iloc[i]['Month'] == 6:
            month6.append(holder3.iloc[i])
            month6 = pd.DataFrame(month6)#.sum()
            yeargroupMC.append(month6['Market Cap'])  #this returns just the market cap of each day in january
            yeargroupV.append(month6['Volume'])   #this returns the volume for each day in january

        if holder3.iloc[i]['Month'] == 7:
            month7.append(holder3.iloc[i])
            month7 = pd.DataFrame(month7)#.sum()
            yeargroupMC.append(month7['Market Cap'])  #this returns just the market cap of each day in january
            yeargroupV.append(month7['Volume'])   #this returns the volume for each day in january

        if holder3.iloc[i]['Month'] == 8:
            month8.append(holder3.iloc[i])
            month8 = pd.DataFrame(month8)#.sum()
            yeargroupMC.append(month8['Market Cap'])  #this returns just the market cap of each day in january
            yeargroupV.append(month8['Volume'])   #this returns the volume for each day in january

        if holder3.iloc[i]['Month'] == 9:
            month9.append(holder3.iloc[i])
            month9 = pd.DataFrame(month9)#.sum()
            yeargroupMC.append(month9['Market Cap'])  #this returns just the market cap of each day in january
            yeargroupV.append(month9['Volume'])   #this returns the volume for each day in january

        if holder3.iloc[i]['Month'] == 10:
            month10.append(holder3.iloc[i])
            month10 = pd.DataFrame(month10)#.sum()
            yeargroupMC.append(month10['Market Cap'])  #this returns just the market cap of each day in january
            yeargroupV.append(month10['Volume'])   #this returns the volume for each day in january


        if holder3.iloc[i]['Month'] == 11:
            month11.append(holder3.iloc[i])
            month11 = pd.DataFrame(month11)#.sum()
            yeargroupMC.append(month11['Market Cap'])  #this returns just the market cap of each day in january
            yeargroupV.append(month11['Volume'])   #this returns the volume for each day in january


        if holder3.iloc[i]['Month'] == 12:
            month12.append(holder3.iloc[i])
            month12 = pd.DataFrame(month12)#.sum()
            yeargroupMC.append(month12['Market Cap'])  # this returns just the market cap of each day in january
            yeargroupV.append(month12['Volume'])  # this returns the volume for each day in january
            #print(month12)

    ####### it does everything in the if statement before coming here!!!!! ############
    y1mc=pd.DataFrame(yeargroupMC).sum()
    y1v=pd.DataFrame(yeargroupV).sum()



    yearreturnmc.append(y1mc)
    yearreturnv.append(y1v)

###### plotting the bar charts ###############
months=['Jan','Feb','Mar','Apr','May','Jun','Jul', 'Aug','Sep','Oct','Nov','Dec']


for t in range(0,len(yearreturnv)):
    year = ['2013', '2014', '2015', '2016', '2017']
    figure1=plt.figure(figsize=(15,10))
    plot1 = figure1.add_subplot(211)  # http://kitchingroup.cheme.cmu.edu/blog/2013/09/13/Plotting-two-datasets-with-very-different-scales/
    v = np.arange(len(yearreturnv[t]))
    yearreturnv[t] = yearreturnv[t][::-1]
    plot1.bar(v, yearreturnv[t], width=0.35)
    plot1.set_title('Year group: '+ str(year[t]),fontsize=20)
    plot1.set_ylabel('Accumulative volume exchanged', fontsize=14)
    plot1.set_xlabel('Month', fontsize=16)
    #plt.xticks(['Jan','Feb','Mar','Apr','May','Jun','Jul', 'Aug','Sep','Oct','Nov','Dec'])

    plot2=figure1.add_subplot(212)
    v = np.arange(len(yearreturnmc[t]))
    yearreturnmc[t] = yearreturnmc[t][::-1]
    plot2.bar(v, yearreturnmc[t], width=0.35)
    plot2.set_ylabel('Accumulative market capital exchanged', fontsize=14)
    plot2.set_xlabel('Month', fontsize=16)
    #plt.savefig('Barcahrts' + str(year[t]) + '.pdf')  # Plotting each year, percentage change of volume and market capital first 30 days ' + str(dflist[placer][0]) + '.pdf')
    plt.close()
#plt.show()
plt.close()         # JUST MAKING SURE ALL GRAPHS FROM ABOVE ARE CLOSED
plt.close()
plt.close()
plt.close()





#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
################################## MACHINE LEARNING #############################################
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################


###### ONTO THE supervised MACHINE LEARNING pART NOW #######        ATTEMPT 1

print('Now at machine learning section')




################################################################################################################
################################################################################################################
############################################ IMPORTANT #########################################################
##### followed https://www.youtube.com/watch?v=JcI5Vnw0b2c&list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v&index=2 #####
############################# https://www.youtube.com/watch?v=3ZWuPVWq7p4 ######################################
########################## https://www.youtube.com/watch?v=SSu00IRRraY&t=302s ##################################
#################################### TO IMPLEMENT THIS SECTION #################################################
################################################################################################################
################################################################################################################


###### JUST SETTING UP MY DATAFRAME FOR MACHINE LEARNING STUFF

MainDataFrame1=pd.DataFrame(dflist[3][1])
MainDataFrame1=MainDataFrame1[['Date','Open','High','Low','Close','Volume','Market Cap']]     ### these are my features.

DateColumn=MainDataFrame1['Date']

MainDataFrame1=MainDataFrame1.sort_values('Date', ascending=True)
MainDataFrame1=MainDataFrame1.set_index('Date')
interstingcolumns=['Open','High','Low','Close','Volume']
print(MainDataFrame1.head())


MarketCaplist=MainDataFrame1['Market Cap']

print('')
Volumelist=MainDataFrame1[interstingcolumns]
print(MainDataFrame1.head())
print(len(MarketCaplist))
print('')
print(Volumelist.head())
print(len(Volumelist))

#Volumelist=preprocessing(Volumelist)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(Volumelist, MarketCaplist)

print('Used to develop/test on a model')
print(X_train.shape)
print(y_train.shape)
print('Used to test the developed model')
print(X_test.shape)
print(y_test.shape)
print('')


LinReg=LinearRegression()
training=LinReg.fit(X_train,y_train)
print('')
print('Onto modelling predict')
predcitcap=training.predict(X_test)
PredictedCap=pd.Series(predcitcap)
print(PredictedCap.head())
print('')

Datelist1=MarketCaplist.index[len(X_train):].tolist()
Datelist1=pd.Series(Datelist1)
Combined=pd.concat([PredictedCap,Datelist1],axis=1)
Combined=Combined.set_index(1)
print(len(Combined))

actualdata=MarketCaplist.iloc[len(X_train):]

figure2=plt.figure(figsize=(15,10))

plot2=figure2.add_subplot(311)
plt.plot(actualdata,label='Actual')
plt.legend()

print('')
print('Maket cap')
plot2=figure2.add_subplot(312)
plt.plot(Combined,label='Predicted')
plt.legend()
print(Combined.head())


plot2=figure2.add_subplot(313)
plt.plot(actualdata,label='Actual')
plt.plot(Combined,label='Predicted')
plt.legend()

#plt.savefig('Predictions, linear supervised : ' + str(2014) + '.pdf')  # Plotting each year, percentage change of volume and market capital first 30 days ' + str(dflist[placer][0]) + '.pdf')
plt.close()
plt.close()
plt.close()
print(actualdata.head())



############### root mean squared ############### to measure the accuracy of the model
print('RMSE section')
RMSE1=np.sqrt((Combined[0]-actualdata[0])**2/len(Combined))     ### individual RMSE
RMSE2=np.sqrt(metrics.mean_squared_error(actualdata,Combined))  ### overall RMSE
print(RMSE2)

sns.pairplot(MainDataFrame1, x_vars=['Open','High','Low','Close','Volume'], y_vars='Market Cap',size=7,aspect=0.5)
#plt.savefig('Predictionsweefliihuweddlfkhjwdesf, linear supervised : ' + str(2016) + '.pdf')  # Plotting each year, percentage change of volume and market capital first 30 days ' + str(dflist[placer][0]) + '.pdf')

plt.close()
plt.close()
plt.close()

################## UNSUPERVISED MACHINE LEARNING #########
### used for model development
###### https://www.youtube.com/watch?v=ikt0sny_ImY ##########


print('Unsupervised Learning')
MnSt=MeanShift()


MeanShiftTraining=MnSt.fit(X_train,y_train)
print(MeanShiftTraining)
print('')

labels=MnSt.labels_
central_clusters=MnSt.cluster_centers_
print(central_clusters)
print('')

NumberOfClusters=len(np.unique(labels))
print(NumberOfClusters)
print('')

columnnum1=4
columnnum2=5
Columnname='Volume'

XX=MainDataFrame1.iloc[:,[columnnum1,columnnum2]].values
print(MainDataFrame1.head())

NumberOfClusters=5
KMeansClusteringModel2=KMeans(n_clusters=NumberOfClusters, init='k-means++')
y=KMeansClusteringModel2.fit_predict(XX)
print(len(y))
figure3=plt.figure(figsize=(15,10))

figure3.add_subplot(211)
plt.scatter(XX[y==0,0],XX[y==0,1],label='Cluster 1')
plt.scatter(XX[y==1,0],XX[y==1,1],label='Cluster 2')
plt.scatter(XX[y==2,0],XX[y==2,1],label='Cluster 3')
plt.scatter(XX[y==3,0],XX[y==3,1],label='Cluster 4')
plt.scatter(XX[y==4,0],XX[y==4,1],label='Cluster 5')

plt.scatter(KMeansClusteringModel2.cluster_centers_[:,0],KMeansClusteringModel2.cluster_centers_[:,1],label='Centroids')
plt.legend()
plt.ylabel('Market Cap')
plt.title('Year group: 2016, '+str(Columnname)+ ' VS Market Capital')

figure3.add_subplot(212)
plt.scatter(MainDataFrame1[Columnname],MainDataFrame1['Market Cap'])
plt.ylabel('Market Cap')
plt.xlabel(Columnname)

#plt.savefig('Clustering'+str(Columnname)+', year group : ' + str(2016) + '.pdf')  # Plotting each year, percentage change of volume and market capital first 30 days ' + str(dflist[placer][0]) + '.pdf')

plt.close()
plt.close()
plt.close()