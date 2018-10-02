
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np


# In[8]:


df_links=pd.read_csv('C:\\Users\\p860n111\\Desktop\\data science\\Hw 3\\links.csv')


# In[9]:


df_movies=pd.read_csv('C:\\Users\\p860n111\\Desktop\\data science\\Hw 3\\movies.csv')


# In[10]:


df_ratings=pd.read_csv('C:\\Users\\p860n111\\Desktop\\data science\\Hw 3\\ratings.csv')


# In[11]:



df_tags=pd.read_csv('C:\\Users\\p860n111\\Desktop\\data science\\Hw 3\\tags.csv')


# In[21]:


df_ratings.groupby(['userId']).count()


# In[23]:


#addtiomnal info1: the userId that has rated the maximum movies
df_ratings.groupby(['userId']).count()['movieId']


# In[231]:


df_ratings_max_userid=df_ratings.groupby(['userId']).count()['movieId']


# In[232]:


df_ratings_max_userid= df_ratings_max_userid.to_frame()
df_ratings_max_userid


# In[233]:


df_ratings_max_userid=df_ratings_max_userid.sort_values('movieId',ascending=False)


# In[234]:


# userid with 414 gace maximum times(movies rating) i.e. gave 2698 movies rating
df_ratings_max_userid


# In[235]:


df_ratings_max_userid = df_ratings_max_userid.rename(columns={'movieId': 'Number of movies rated'})


# In[236]:


df_ratings_max_userid


# In[44]:


df_ratings_max_userid.to_csv('AdditionalInfo#1_userID_Max_rated_Movies.csv', sep=',')


# In[45]:


df_tags.groupby(['userId']).count()


# In[48]:


#addtiomnal info2: the userId that has tagged the maximum movies
df_tags.groupby(['userId']).count()['movieId']


# In[49]:


df_tag_max_userid=df_tags.groupby(['userId']).count()['movieId']
df_tag_max_userid= df_tag_max_userid.to_frame()
df_tag_max_userid


# In[54]:


df_tag_max_userid = df_tag_max_userid.rename(columns={'movieId': 'Number of movies tagged'})
df_tag_max_userid=df_tag_max_userid.sort_values('Number of movies tagged',ascending=False)


# In[55]:


df_tag_max_userid


# In[56]:


df_tag_max_userid.to_csv('AdditionalInfo#2_userID_Max_tagged_Movies.csv', sep=',')


# In[57]:


df_tag_max_userid


# In[59]:


#additional information 3: to merge the number of movies and tagged created by each user id

df_Merged= pd.merge(df_ratings_max_userid,df_tag_max_userid, on="userId", how="outer")


# In[60]:


df_Merged


# In[61]:


df_Merged.to_csv('AdditionalInfo#3_userID_max_tagged_and_max_rated.csv', sep=',')


# In[62]:


#Additiomnal info4: to convert into time and then deduce info
df_ratings_timeStamp=pd.to_datetime(df_ratings['timestamp'], unit='s')


# In[63]:


df_ratings_timeStamp


# In[64]:


# to create a new dataframe with converted time
df_ratings_timeStamp_new= df_ratings.drop('timestamp', axis=1)

df_ratings_timeStamp_merged_new=pd.concat([df_ratings_timeStamp_new,df_ratings_timeStamp], axis=1) 


# In[65]:


df_ratings_timeStamp_merged_new


# In[67]:


#created 3 new column for year month and date
df_ratings_timeStamp_merged_new['year'] = pd.DatetimeIndex(df_ratings_timeStamp_merged_new['timestamp']).year
df_ratings_timeStamp_merged_new['month'] = pd.DatetimeIndex(df_ratings_timeStamp_merged_new['timestamp']).month


# In[68]:


df_ratings_timeStamp_merged_new


# In[71]:


df_ratings_timeStamp_merged_new['date'] = pd.DatetimeIndex(df_ratings_timeStamp_merged_new['timestamp']).day


# In[72]:


df_ratings_timeStamp_merged_new


# In[76]:


#addtiomnal info4: the userId that has rated the maximum movies year wise
df_ratings_timeStamp_merged_newYEAR=df_ratings_timeStamp_merged_new.groupby(['year']).count()['userId']


# In[77]:


df_ratings_timeStamp_merged_newYEAR= df_ratings_timeStamp_merged_newYEAR.to_frame()


# In[78]:


df_ratings_timeStamp_merged_newYEAR


# In[79]:


df_ratings_timeStamp_merged_newYEAR = df_ratings_timeStamp_merged_newYEAR.rename(columns={'userId': 'Number of users who rated'})


# In[80]:


df_ratings_timeStamp_merged_newYEAR


# In[81]:


df_ratings_timeStamp_merged_newYEAR=df_ratings_timeStamp_merged_newYEAR.sort_values('Number of users who rated',ascending=False)


# In[82]:


df_ratings_timeStamp_merged_newYEAR


# In[83]:



df_ratings_timeStamp_merged_newYEAR.to_csv('AdditionalInfo#4_yearWiseUserRatingCount.csv', sep=',')


# In[84]:


#addtiomnal info5: the userId that has rated the maximum movies month wise
df_ratings_timeStamp_merged_newMONTH=df_ratings_timeStamp_merged_new.groupby(['month']).count()['userId']


# In[85]:


df_ratings_timeStamp_merged_newMONTH


# In[86]:


df_ratings_timeStamp_merged_newMONTH= df_ratings_timeStamp_merged_newMONTH.to_frame()


# In[87]:


df_ratings_timeStamp_merged_newMONTH


# In[88]:


df_ratings_timeStamp_merged_newMONTH=df_ratings_timeStamp_merged_newMONTH.sort_values('userId',ascending=False)


# In[90]:


df_ratings_timeStamp_merged_newMONTH


# In[91]:




df_ratings_timeStamp_merged_newMONTH = df_ratings_timeStamp_merged_newMONTH.rename(columns={'userId': 'Number of users rated'})

df_ratings_timeStamp_merged_newMONTH.to_csv('AdditionalInfo#5_userID_Max_rated_MoviesMonthWise.csv', sep=',')


# In[97]:


df_Merged_tags_ratings= pd.merge(df_ratings,df_tags, on="movieId", how="inner")


# In[98]:


df_Merged_tags_ratings


# In[96]:


display(df_Merged_tags_ratings)


# In[107]:


df_tags.drop('timestamp' ,axis=1,inplace=True)


# In[112]:


df_tags


# In[113]:


df_ratings.drop('timestamp' ,axis=1,inplace=True)


# In[114]:


df_ratings


# In[179]:


df_Merged_tags_ratings= pd.merge(df_ratings,df_tags, on=["movieId","userId"], how ="outer")


# In[180]:


df_Merged_tags_ratings


# In[181]:


df_Merged_tags_ratings = df_Merged_tags_ratings.dropna()
print(df_Merged_tags_ratings.shape)
print(list(df_Merged_tags_ratings.columns))


# In[182]:


df_Merged_tags_ratings


# In[ ]:





# In[135]:


from sklearn.neighbors import KNeighborsClassifier


# In[183]:


df_Merged_tags_ratings.dtypes


# In[184]:


df_Merged_tags_ratings["tag"] = df_Merged_tags_ratings["tag"].astype('category')
df_Merged_tags_ratings.dtypes
df_Merged_tags_ratings["tag_cat"] = df_Merged_tags_ratings["tag"].cat.codes
df_Merged_tags_ratings.head()


# In[170]:


print(df_Merged_tags_ratings.shape)
print(list(df_Merged_tags_ratings.columns))


# In[171]:


df_Merged_tags_ratings


# In[187]:



df_Merged_tags_ratings_new.shape


# In[186]:


df_Merged_tags_ratings_new = df_Merged_tags_ratings.filter(['userId','movieId','rating','tag_cat'], axis=1)


# In[188]:


df_Merged_tags_ratings_new.shape


# In[191]:


from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
Y=df_Merged_tags_ratings_new.movieId #output label
X=df_Merged_tags_ratings_new.drop('movieId',axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)
print(X_train)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
#print("Accuracy:",metrics.accuracy_score(y_test, y_train))
predictions = knn.predict(X_test)
score = knn.score(X_test, y_test)
print(score)


# In[192]:


df_Merged_tags_ratings_new_withTitle= pd.merge(df_Merged_tags_ratings_new,df_movies, on=["movieId"], how ="outer")


# In[194]:


df_Merged_tags_ratings_new_withTitle.dropna()


# In[195]:


df_Merged_tags_ratings_new_withTitle.dtypes


# In[196]:


df_Merged_tags_ratings_new_withTitle["title"] = df_Merged_tags_ratings_new_withTitle["title"].astype('category')
df_Merged_tags_ratings_new_withTitle.dtypes
df_Merged_tags_ratings_new_withTitle["title_cat"] = df_Merged_tags_ratings_new_withTitle["title"].cat.codes
df_Merged_tags_ratings_new_withTitle.head()

df_Merged_tags_ratings_new_withTitle["genres"] = df_Merged_tags_ratings_new_withTitle["genres"].astype('category')
df_Merged_tags_ratings_new_withTitle.dtypes
df_Merged_tags_ratings_new_withTitle["genres_cat"] = df_Merged_tags_ratings_new_withTitle["genres"].cat.codes
df_Merged_tags_ratings_new_withTitle.head()


# In[201]:


df_Merged_tags_ratings_new_withTitle.shape


# In[198]:


df_Merged_tags_ratings_new_withTitleFinal = df_Merged_tags_ratings_new_withTitle.filter(['userId','movieId','rating','tag_cat','title_cat','genres_cat'], axis=1)


# In[203]:


df_Merged_tags_ratings_new_withTitleFinal.dropna()


# In[215]:


from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
Y=df_Merged_tags_ratings_new_withTitleFinal.title_cat #output label
X=df_Merged_tags_ratings_new_withTitleFinal.drop('title_cat',axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)
print(X_train)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
#print("Accuracy:",metrics.accuracy_score(y_test, y_train))
predictions = knn.predict(X_test)
score = knn.score(X_test, y_test)
print(score)


# In[209]:


df_Merged_tags_ratings_new_withTitleFinal.shape


# In[208]:


df_Merged_tags_ratings_new_withTitleFinal=df_Merged_tags_ratings_new_withTitleFinal.dropna()


# In[216]:


import seaborn as sns

Var_Corr = df_Merged_tags_ratings_new_withTitleFinal.corr()
# plot the heatmap and annotation on it
sns.heatmap(Var_Corr, xticklabels=Var_Corr.columns, yticklabels=Var_Corr.columns, annot=True)


# In[217]:


corr = df_Merged_tags_ratings_new_withTitleFinal.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# In[218]:


corr.style.background_gradient().set_precision(2)


# In[219]:


from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
Y=df_Merged_tags_ratings_new_withTitleFinal.title_cat #output label
X=df_Merged_tags_ratings_new_withTitleFinal.drop('title_cat',axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)
print(X_train)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
#print("Accuracy:",metrics.accuracy_score(y_test, y_train))
predictions = knn.predict(X_test)
score = knn.score(X_test, y_test)
print(score)


# In[237]:


df_ratings_max_userid['userId'] = df_ratings_max_userid.index.tolist()

df_ratings_max_userid.index = np.arange(0,len(df_ratings_max_userid))


# In[240]:


df_ratings_max_userid['userId'] = df_ratings_max_userid.index.tolist()

df_ratings_max_userid.index = np.arange(0,len(df_ratings_max_userid))


# In[241]:


df_ratings_max_userid


# In[244]:


import matplotlib.pyplot as plt
plt.figure(figsize=(15,15))
ax= sns.barplot(x='Number of movies rated',y='userId',data=df_ratings_max_userid, order = df_ratings_max_userid['userId'])
ax.set(xlabel='Number of movies rated', ylabel='userId')
plt.show()


# In[243]:


df_ratings_max_userid.dtypes


# In[245]:


df_tag_max_userid['userId'] = df_tag_max_userid.index.tolist()

df_tag_max_userid.index = np.arange(0,len(df_tag_max_userid))


# In[246]:


df_tag_max_userid


# In[248]:


import matplotlib.pyplot as plt
plt.figure(figsize=(20,20))
ax= sns.barplot(x='Number of movies tagged',y='userId',data=df_tag_max_userid, order = df_tag_max_userid['userId'])
ax.set(xlabel='Number of movies tagged', ylabel='userId')
plt.show()


# In[249]:


df_Merged['userId'] = df_Merged.index.tolist()

df_Merged.index = np.arange(0,len(df_Merged))


# In[250]:


import matplotlib.pyplot as plt
plt.figure(figsize=(20,20))
ax= sns.barplot(x='Number of movies tagged',y='Numberof movies rated',data=df_Merged, order = df_Merged['Numberof movies rated'])
ax.set(xlabel='Number of movies tagged', ylabel='Numberof movies rated')
plt.show()


# In[257]:


from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
Y=df_Merged_tags_ratings_new_withTitleFinal.title_cat #output label
X=df_Merged_tags_ratings_new_withTitleFinal.drop('title_cat',axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)
print(X_train)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
#print("Accuracy:",metrics.accuracy_score(y_test, y_train))
predictions = knn.predict(X_test)
score = knn.score(X_test, y_test)
print(score)

