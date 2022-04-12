#!/usr/bin/env python
# coding: utf-8

# In[37]:


import findspark
findspark.init()


# In[2]:


findspark.init()


# In[3]:


import pyspark 
from pyspark.sql import SparkSession
from pyspark.sql.types import * 
from pyspark.sql.functions import *


# In[ ]:





# In[ ]:





# In[4]:


import findspark
findspark.init()


import pyspark 
from pyspark.sql import SparkSession


spark = SparkSession.builder.getOrCreate()

df = spark.sql("select 'spark' as hello")

import os
os.getcwd()



# In[5]:


df_load = spark.read.csv('Practice Data2.csv', header = True)


# In[6]:


#use backslash for multiple lines 


# In[7]:


df_load


# In[8]:


#to see the first five rows 

df_load.take(5)


# In[9]:


#To Drop Columns

dropped_columns = ['Oldpeak'] #only dropping 1 in this case
df_load = df_load.drop(*dropped_columns) 

#will be the same but with the dropped columns
df_load.show(5)


# In[10]:


#OldPeak sucessfully removed. What if I want to extract a date though?


# In[11]:


#Creating Year field 

df_load = df_load.withColumn('Year', year(to_timestamp('Date ', 'MM/dd/yyyy')))


# In[12]:


#lets check out the new column! 
df_load.show(5)


# In[13]:


#looking good! Now lets extract the month. 

df_load = df_load.withColumn('Month', month(to_timestamp('Date ', 'MM/dd/yyyy')))


# In[14]:


df_load.show(10)


# In[15]:


#looks good. Now lets do a group by method to see the number of records by year


# In[16]:


df_counts = df_load.groupBy('Year').count()


# In[17]:


#lets see the general scheme 


df_counts.printSchema() #ends up being what we expected


# In[ ]:





# In[18]:


# now lets test out some machine learning functionality 

df_load.show(10)


# In[19]:


df_train = df_load.drop('Date ')
df_train.show(5)


# In[ ]:





# In[20]:


Xtrain, Xtest = df_train.randomSplit([0.7, 0.3], seed = 2018)


# In[21]:


y_train = Xtrain['HeartDisease']
y_test = Xtest['HeartDisease']


# In[22]:


Xtrain = Xtrain.drop('HeartDisease')
Xtest = Xtrain.drop('HeartDisease')


# In[28]:


# stay tuned for machine learning 


# In[35]:


from pyspark.sql.functions import col, pandas_udf


# In[41]:


type(df_load)


# In[ ]:




