#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8
import findspark



findspark.init()

import pyspark 
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

df = spark.sql("select 'spark' as hello")
df.show()
import os
os.getcwd()



# In[2]:


df = spark.read.csv('OilData.csv', header = True)

##lets do some simple data manipulation 


# In[3]:


df.show(5)


# In[4]:


df2 = df[(df['Month'] == 1)]


# In[5]:


df2.show(10)


# In[6]:


df_cut = df.toPandas().iloc[0:200]


# In[7]:


df_cut = spark.createDataFrame(df_cut)


# In[8]:


#trying out some machine learning\
#lets create the training and testing sets
#first we need to convert our datatypes to integers rather than floats. 
from pyspark.sql.types import IntegerType
from pyspark.sql.types import FloatType
df = df.withColumn("Year ", df["Year " ].cast(IntegerType()))
df = df.withColumn("Gas Price", df["Gas Price" ].cast(FloatType()))


# In[9]:


#splitting into training and testing 

trainDF, testDF = df.randomSplit([.8, .2], seed=42)


# In[ ]:





# In[10]:


df.printSchema()


# In[11]:


#algorithms in Spark requires that all the input features 
#are contained within a single vector in your DataFrame

# lets use a transformer 
#VectorAssembler takes a list of input columns and creates a 
#new DataFrame with an additional column, which we will call features


from pyspark.ml.feature import VectorAssembler
vecAssembler = VectorAssembler(inputCols=['Year '], outputCol="features")
vecTrainDF = vecAssembler.transform(trainDF)
vecTrainDF.select("Year ", "Day ", "Month").show(10) #only getting these columns


# In[12]:


# doing a simple linear regression


# In[13]:


from pyspark.ml.regression import LinearRegression
lr = LinearRegression(featuresCol="features", labelCol="Gas Price")
lrModel = lr.fit(vecTrainDF)


# In[14]:


m = round(lrModel.coefficients[0], 2)
b = round(lrModel.intercept, 2)
print(f"""The formula for the linear regression line is
price = {m}*bedrooms + {b}""")


# In[15]:


#looking good!


# In[16]:


#lets make a pipeline to steamline things a bit 

from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[vecAssembler, lr])
pipelineModel = pipeline.fit(trainDF)


# In[17]:


predDF = pipelineModel.transform(testDF)
predDF.select("Month", "features", "Gas Price", "prediction").show(10)


# In[18]:


#now lets see it with a different ML model. 


# In[19]:


#to evaluate our model with RMSE 

from pyspark.ml.evaluation import RegressionEvaluator
regressionEvaluator = RegressionEvaluator(
 predictionCol="prediction",
 labelCol="Gas Price",
 metricName="rmse")
rmse = regressionEvaluator.evaluate(predDF)
print(f"RMSE is {rmse:.1f}")


# In[21]:


#Not Bad!


# In[ ]:




