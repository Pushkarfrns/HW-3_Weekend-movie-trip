
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


# In[35]:


df_ratings_max_userid=df_ratings.groupby(['userId']).count()['movieId']


# In[36]:


df_ratings_max_userid= df_ratings_max_userid.to_frame()
df_ratings_max_userid


# In[37]:


df_ratings_max_userid=df_ratings_max_userid.sort_values('movieId',ascending=False)


# In[38]:


# userid with 414 gace maximum times(movies rating) i.e. gave 2698 movies rating
df_ratings_max_userid


# In[41]:


df_ratings_max_userid = df_ratings_max_userid.rename(columns={'movieId': 'Number of movies rated'})


# In[42]:


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


# In[128]:


df_Merged_tags_ratings= pd.merge(df_ratings,df_tags, on=["movieId","userId"], how ="outer")


# In[129]:


df_Merged_tags_ratings


# In[130]:


df_Merged_tags_ratings.dropna()

