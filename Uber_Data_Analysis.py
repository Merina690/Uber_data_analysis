#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv('C:\\Users\\MERINA ANGEL\\Downloads\\uber.csv')


# Geography: USA, Sri Lanka and Pakistan
# 
# Time period: January - December 2016
# 
# Unit of analysis: Drives
# 
# Total Drives: 1,155
# 
# Total Miles: 12,204
# 
# Dataset: The dataset contains Start Date, End Date, Start Location, End Location, Miles Driven and Purpose of drive (Business, Personal, Meals, Errands, Meetings, Customer Support etc.)

# # UNDERSTANDING THE DATA

# 
# Need to understand the data to clean and build the model on it. To understand the data we use pandas, numpy and matplotlib to visulaize the data in an efficient way

# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.columns


# In[6]:


df.isnull().sum()


# In[7]:


#we can see there are some null values in the data


# In[8]:


df.tail() #here we can see that last row doesn't have any detailed information so we can remove it


# In[9]:


# Remove uncessary data either it is column or row
uber_df = df[:-1] 


# In[10]:


uber_df


# In[11]:


df.dtypes 


# As we can see that date columns are in object dtype. so we need convert them to datetime format

# In[12]:


import datetime


# In[13]:


uber_df[['START_DATE*','END_DATE*']] = uber_df[['START_DATE*','END_DATE*']].apply(pd.to_datetime)


# In[14]:


uber_df.dtypes


# In[15]:


uber_df.info()


# # VISUALIZATION

# visualization helps the people to understand the problem within less time and get insights in an easier way through graphs.
# We use matplotlib and searborn to do visualization

# In[16]:


fig = plt.figure(figsize = (10, 5))
 
uber_df['CATEGORY*'].value_counts().plot(kind='bar') #visualizing the category using bar graph


# In[17]:


# Here we can see that business category are of more than 1000 and personal is of 50


# In[18]:


x = uber_df['PURPOSE*'].value_counts() #visualization of purpose using bar graph
x.plot(kind='bar',figsize=(10,5),color='magenta')


# In[19]:


#Here we can see than most of the purpose of drive is Meeting(185),Meal/Entertain(152) and least purpose is of commute(2)


# In[20]:


hours = uber_df['START_DATE*'].dt.hour.value_counts()
hours.plot(kind='bar',color='black',figsize=(10,5)) #visualizing the hours in start date with the help of bar graph
plt.xlabel('Hours')
plt.ylabel('Frequency')
plt.title('Number of trips per hour')


# In[21]:


#Here we visualized the number of trips per hour. Most of the drivers spent 15 hours to make 100 trips in a day.


# In[22]:


month = uber_df['START_DATE*'].dt.month.value_counts() #month in start date visualization
month.plot(kind='bar',color='red',figsize=(10,5)) #using bargraph to visualize the month data (1-12) with the help of bars
plt.xlabel('Month')
plt.ylabel('Frequency')
plt.title('Number of trips per MONTH')


# In[23]:


#We visualized the month data in a start date to know number of trips made by drivers in a month. Here we can see that for 12 months they made 145 trips.


# In[24]:


uber_df['MILES*'].plot(kind='hist',color='Blue',figsize=(10,5)) #visualizing the miles travelled by drivers.
#we use histogram as this column is of numeric datatype


# In[25]:


#we visualized the miles column to know number of miles travelled by each driver


# As we need to know how much a driver travel a day and the speed he use while travelling. This is because the uber have large many customers and employees, they need to maintain the customer satisfaction and their wellbeing.

# In[26]:


minutes=[]
uber_df['Duration_Minutes'] = uber_df['END_DATE*'] - uber_df['START_DATE*'] #to know the exact minutes travelled by the driver
uber_df['Duration_Minutes'] #printing the duration minutes column
for i in uber_df['Duration_Minutes']:
    minutes.append(i.seconds / 60) 

uber_df['Duration_Minutes'] = minutes #changing the name of a variable to minutes


# In[27]:


#calculation of speed of a trip by drivers
uber_df['Duration_hours'] = uber_df['Duration_Minutes'] / 60 #calculation of hours as per the minutes travelled
uber_df['Speed_KM'] = uber_df['MILES*'] / uber_df['Duration_hours'] #number of miles travelled per hour
uber_df['Speed_KM'] #speed of a driver while travelling


# In[28]:


uber_df.isnull().sum() #checking null values


# Here we can see null values in purpose. But we can manupulate those null values as per the expert domain suggestion. We can't replace or delete them directly.

# ### What is the average length of the trip?

# In[29]:


uber_df['MILES*'].mean() # average miles travelled


# ### Average number of rides per week or per month?

# In[30]:


month.mean()  #average rides per month


# In[31]:


week=month/4  #this is to calculate for week
week #trips per week
week.mean() #average rides per week


# ### Percentage of business  vs personal ?

# In[32]:


(uber_df['CATEGORY*'].value_counts() / len(uber_df))*100


# In[33]:


uber_df  #printing uber_df file


# Now the dataset is ready for feature scaling, engineering and model building.

# In[ ]:





# In[ ]:




