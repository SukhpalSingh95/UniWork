import random as rm
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
import numpy as np



datasetin= pd.read_csv('/Users/sukhpalsingh/PycharmProjects/ASEWorks/Assessment2/Attempt2/TestMaterials/smallfaultmatrix.txt',header=None,sep= ',',index_col=0, comment='#') #https://stackoverflow.com/questions/21546739/load-data-from-txt-with-pandas + https://stackoverflow.com/questions/28200404/pandas-read-table-use-first-column-as-index
playdata=pd.DataFrame(data=datasetin)
print(playdata[0:10])
print(' ')
#target =['1','1','1','1','1','1','1','1','1']
#target=np.full((0, len(datasetin.columns)), 0,dtype=int)    #https://stackoverflow.com/questions/5891410/numpy-array-initialization-fill-with-identical-values
target=np.empty(len(datasetin.columns)); target.fill(1)
maxvalue=len(target)
print(maxvalue)



playdata.columns=['C1','C2','C3','C4','C5','C6','C7','C8','C9']
print(playdata[0:10])

#for index in dataset.iterrows():
#    if dataset['1'] ==target:
#        print('fdskjhf')

#TO GET EACH ELEMENT WITHIN A DATA FRAME USE [][][]
#print(playdata([1][1]))

#goes through each column
#print(' ')
#for i in playdata:
#    print('row data'+str(i))
#print(' ')


v=[0]*len(playdata)

index=enumerate(playdata)
columns1=['Scores']
#columns=len(playdata)
#create an empty Dataframe for later
Dataframe1=pd.DataFrame(index=v, columns=columns1)
Dataframe1=Dataframe1.fillna(0)

#print(Dataframe1)

print(' ')
class detailingdataframedetail:
    def individuals(self):
        self.score=score
        score=0
        return self

    def comparisontotarget(self):
        return self

#empty Dataframe
emptyDF=pd.DataFrame()
#print(emptyDF)

if emptyDF.empty:
    emptyDF=playdata
else:
    emptyDF=emptyDF.join(playdata)

print(emptyDF.head())

#emptyDF.plot()
#plt.show()
print(' ')

#randomly select a place in the search space
numberran=rm.randint(0,215)
v=playdata.ix[numberran]
print(v)
print(' ')

#playdata['Score']=playdata.ix[numberran].eq(target)
#print(playdata)
numnum=0

#for d in enumerate(Dataframe1):
#    for i in Dataframe1:
##        if d == target:
 #           print('okkkk')
 #           print(i)
 #           print(d)
 #           print(target)
 #           break
 #       else:
 #           print('nokkkkkkkkkk')
 #           numnum+=1

#print(numnum)

#searches through each row
#for index, row in playdata.iterrows():
#    if index == target:
#        # print (playdata.ix[row])
##        print('holalsdfkjhdfkjhsdfjkhdskjhfkdhjfhkjahfkhsakhfs')
 #       break
 #   else:
 #       print('afdsfhgaskhjkshdjafsd')
 #       #print(row)
 #       numnum += 1
#for index, row in playdata.iterrows():
#    for i in index:
#        if i == target:
#        # print (playdata.ix[row])
#            print('holalsdfkjhdfkjhsdfjkhdskjhfkdhjfhkjahfkhsakhfs')
#            break
#        else:
#            print('afdsfhgaskhjkshdjafsd')
        #print(row)
#            numnum += 1

#print(numnum)
#print(index)
#print(row)
print('')
sdf=playdata.isin(target)
print(sdf)
e=playdata.sum(axis=1)                #https://stackoverflow.com/questions/42886354/pandas-count-values-by-condition-for-row
#print(e)
t=pd.Series(e)
df3=playdata.assign(Score=e)            #https://chrisalbon.com/python/pandas_assign_new_column_dataframe.html
print(df3)


#Scoresearch=df4.set_index("Score")
#Scoresearch=df4.loc[columns[9]]
#print(Scoresearch)


###RETURNS JUST THE SCORE LINE
#df5=df4.loc[:,'Score']
#print(df5.head())

#t=max(df5)
#print('max value in score: '+str(t))

#z=df5.mean(0)
#print('Average value in score: '+str(int(z)))

#y=min(df5)
#print('Min value in Score: '+str(y))
#print(' ')
#numnum=0

##ADDS TERMS TOGETHER, PURELY BASED ON SCORE THOUGH, DOES NOT TRACK WHATS BEEN ADDED
#for i in df5:   #for each row in df5
#    if i<z:             #adds values of i that are smaller than the average to the average, increases its score
#        v=int(i+z)
#        print('Less than average, but now: '+str(v) )
#        numnum+=1
#    else:
#        s=int(i+z)  #if equal to average or larger than average, adds that row to average
#        print('Larger than average, but now: '+str(s))
#        numnum+=1
#print(numnum)



#Scoresearch=df4.set_index("Score")
#Scoresearch=df4.loc[columns[9]]
#print(Scoresearch)


###RETURNS JUST THE SCORE LINE
#df5=df4.loc[:,'Score']
#print(df5.head())

#t=max(df5)
#print('max value in score: '+str(t))

#z=df4.mean(0)
#print('Average value in score: '+str(int(z)))

#y=min(df5)
#print('Min value in Score: '+str(y))
#print(' ')
#numnum=0

##ADDS TERMS TOGETHER, PURELY BASED ON SCORE THOUGH, DOES NOT TRACK WHATS BEEN ADDED
#for i in df4.ix[9]:   #for each row in df5
#    if i<z:             #adds values of i that are smaller than the average to the average, increases its score
#        v=int(i+z)
#        print('Less than average, but now: '+str(v) )
#        numnum+=1
#    else:
#        s=int(i+z)  #if equal to average or larger than average, adds that row to average
#        print('Larger than average, but now: '+str(s))
#        numnum+=1
#print(numnum)
#for i in enumerate(df4):
#    for row in i:
#        print(df4.iloc[[row]])

#df4tomat=df4.as_matrix()
#print(df4tomat)