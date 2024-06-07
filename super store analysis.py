#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import plotly.colors as colors
pio.templates.default = "plotly_white"


# In[3]:


data = pd.read_csv("Sample - Superstore.csv", encoding='latin-1')
data
# When you specify encoding='latin-1', it tells Pandas to use the Latin-1
#encoding to interpret the text in the CSV file. This encoding is capable of 
#representing a large number of characters from various languages, making it suitable for 
#handling text data that includes special characters, diacritics, 
#and extended character sets found in Western European languages.


# # Pre-Processing

# In[4]:


data.head()


# In[5]:


data.tail()


# In[6]:


data.columns


# In[7]:


print(data.describe())


# In[8]:


data.describe(include = 'all')


# In[9]:


data.info()


# In[10]:


data['Order Date'] = pd.to_datetime(data['Order Date'])
data['Ship Date'] = pd.to_datetime(data['Ship Date'])
data['Sales'] = data['Sales'].astype(float)
data['Discount'] = data['Discount'].astype(float)


# In[11]:


# Create a sample DataFrame
df = pd.DataFrame({'date': ['2022-01-01', '2022-02-01', '2022-03-01']})

#Define a custom function to remove the hyphen from the date column
def remove_hyphen(column):
    return column.str.replace('-', '')

# Apply the custom function to the date column
data['Order ID'] = remove_hyphen(data['Order ID'])

data


# In[12]:


null_values = data.isna().sum()
null_values


# In[13]:


data.shape


# In[14]:


data.ndim   # two dimensional data


# # Model Building

# In[15]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[16]:


df =data.copy()
df.head(5)


# In[17]:


data.columns


# In[18]:


df['Row ID'].nunique() 


# In[19]:


df.drop(columns = 'Row ID', inplace = True)


# In[20]:


df['Order ID'].nunique() 


# In[21]:


df.drop(columns = 'Order ID', inplace = True)


# In[22]:


df.drop(columns = 'Order Date', inplace = True)
df.drop(columns = 'Ship Date', inplace = True)


# In[23]:


df['Ship Mode'].nunique() 


# In[24]:


df['Customer ID'].nunique() 


# In[25]:


df.drop(columns = 'Customer ID', inplace = True)


# In[26]:


df['Customer Name'].nunique() 


# In[27]:


df.drop(columns = 'Customer Name', inplace = True)#,


# In[28]:


df['Country'].nunique() 


# In[29]:


df['City'].nunique() 


# In[30]:


df['State'].nunique() 


# In[31]:


df['Postal Code'].nunique() 


# In[32]:


df.drop(columns = 'Postal Code', inplace = True)


# In[33]:


df['Region'].nunique() 


# In[34]:


df['Product ID'].nunique() 


# In[35]:


df.drop(columns = 'Product ID', inplace = True)


# In[36]:


df['Category'].nunique() 


# In[37]:


df['Sub-Category'].nunique() 


# In[38]:


df['Product Name'].nunique() 


# In[39]:


df.drop(columns = 'Product Name', inplace = True)


# In[40]:


df['Quantity'].nunique() 


# In[41]:


df.head(5)


# In[42]:


df.shape


# In[43]:


df.columns


# In[44]:


df.info()


# # Feature Scaling

# In[45]:


# from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import StandardScaler

# Create a StandardScaler object
scaler = StandardScaler()

# Fit and transform the numeric columns
numeric_columns = ["Sales", "Quantity", "Discount", "Profit"]
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])


# In[46]:


# sc = StandardScaler()

from sklearn.preprocessing import MinMaxScaler

# Create a MinMaxScaler object
scaler = MinMaxScaler()

# Fit and transform the numeric columns
numeric_columns = ["Sales", "Quantity", "Discount", "Profit"]
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])


# # Converting Categorical value into Numeric Data

# In[47]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Ship Mode'] = le.fit_transform(df['Ship Mode'])
df['Segment'] = le.fit_transform(df['Segment'])
df['Country'] = le.fit_transform(df['Country'])
df['City'] = le.fit_transform(df['City'])
df['State'] = le.fit_transform(df['State'])
df['Region'] = le.fit_transform(df['Region'])
df['Category'] = le.fit_transform(df['Category'])
df['Sub-Category'] = le.fit_transform(df['Sub-Category'])




# In[48]:


categorical_columns = ["Ship Mode", "Segment", "Region", "Category", "Sub-Category"]
df_encoded = pd.get_dummies(df, columns=categorical_columns)


# In[49]:


df.tail(10)


# In[50]:


X = df.iloc[:,:-1].values


# In[51]:


y=df.iloc[:,-1].values


# In[52]:


#Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[53]:


print(X_train)
X_train.shape


# In[54]:


print(X_test)
X_test.shape


# In[55]:


print(y_train) 
y_train.shape


# In[56]:


print(y_test)
y_test.shape


# In[57]:


# Training the Multiple Linear Regression model on the Training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression ( )
regressor.fit(X_train, y_train)

# Make Pradections
y_pred = regressor.predict(X_test)


# In[58]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R-squared:", r2)


# # Visualization

# In[59]:


data['Order Date'] = pd.to_datetime(data['Order Date'])
data['Ship Date'] = pd.to_datetime(data['Ship Date']) 

data['Order Month'] = data['Order Date'].dt.month 
data['Order Year'] = data['Order Date'].dt.year
data['Order Day of Week'] = data['Order Date'].dt.dayofweek


# In[60]:


sales_by_month = data.groupby('Order Month')['Sales'].sum().reset_index()
fig = px.line(sales_by_month, 
              x='Order Month', 
              y='Sales', 
              title='Monthly Sales Analysis')
fig.show()


# In[61]:


sales_by_category = data.groupby('Category')['Sales'].sum().reset_index()


fig = px.pie(sales_by_category, 
             values='Sales', 
             names='Category', 
             hole=0.5, 
             color_discrete_sequence=px.colors.qualitative.Pastel)

fig.update_traces(textposition='inside', textinfo='percent+label')
fig.update_layout(title_text='Sales Analysis by Category', title_font=dict(size=24))

fig.show()


# In[62]:


sales_by_subcategory = data.groupby('Sub-Category')['Sales'].sum().reset_index()
fig = px.bar(sales_by_subcategory, 
             x='Sub-Category', 
             y='Sales', 
             title='Sales Analysis by Sub-Category')
fig.show()


# In[63]:


profit_by_month = data.groupby('Order Month')['Profit'].sum().reset_index()
fig = px.line(profit_by_month, 
              x='Order Month', 
              y='Profit', 
              title='Monthly Profit Analysis')
fig.show()


# In[64]:


profit_by_category = data.groupby('Category')['Profit'].sum().reset_index()

fig = px.pie(profit_by_category, 
             values='Profit', 
             names='Category', 
             hole=0.5, 
             color_discrete_sequence=px.colors.qualitative.Pastel)

fig.update_traces(textposition='inside', textinfo='percent+label')
fig.update_layout(title_text='Profit Analysis by Category', title_font=dict(size=24))

fig.show()


# In[65]:


profit_by_subcategory = data.groupby('Sub-Category')['Profit'].sum().reset_index()
fig = px.bar(profit_by_subcategory, x='Sub-Category', 
             y='Profit', 
             title='Profit Analysis by Sub-Category')
fig.show()


# In[66]:


sales_profit_by_segment = data.groupby('Segment').agg({'Sales': 'sum', 'Profit': 'sum'}).reset_index()

color_palette = colors.qualitative.Pastel

fig = go.Figure()
fig.add_trace(go.Bar(x=sales_profit_by_segment['Segment'], 
                     y=sales_profit_by_segment['Sales'], 
                     name='Sales',
                     marker_color=color_palette[0]))
fig.add_trace(go.Bar(x=sales_profit_by_segment['Segment'], 
                     y=sales_profit_by_segment['Profit'], 
                     name='Profit',
                     marker_color=color_palette[1]))

fig.update_layout(title='Sales and Profit Analysis by Customer Segment',
                  xaxis_title='Customer Segment', yaxis_title='Amount')

fig.show()


# In[67]:


sales_profit_by_segment = data.groupby('Segment').agg({'Sales': 'sum', 'Profit': 'sum'}).reset_index()
sales_profit_by_segment['Sales_to_Profit_Ratio'] = sales_profit_by_segment['Sales'] / sales_profit_by_segment['Profit']
print(sales_profit_by_segment[['Segment', 'Sales_to_Profit_Ratio']])

