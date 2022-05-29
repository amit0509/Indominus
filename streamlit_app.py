# Importing Libraries
from cProfile import label
from enum import auto
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="Indominus",
    page_icon="logo3.png",
    layout="wide",
    initial_sidebar_state="collapsed"
  
)

st.sidebar.image('logo1.png')
st.sidebar.markdown('This project **Indominus** is developed by team **Coders Consortium**')
st.sidebar.header('**Team Members:**')
st.sidebar.markdown('- **Amit Singh Kushwaha**')
st.sidebar.markdown('- **Ankit Sonkar**')
st.sidebar.markdown('- **Deepak Yadav**')
st.sidebar.markdown('- **Shikhar Mishra**')

st.sidebar.header('You can also Check our WebApps:')
st.sidebar.subheader('[Currency Convertor](https://amit0509.github.io/currencyconvertor/)')
st.sidebar.subheader('[Crypto Tracker](https://amit0509.github.io/cryptotracker.github.io/)')


end = datetime.now()
start = datetime(end.year-2, end.month, end.day)

st.image("logo2.png",width=100)
st.title('Stock Market Predictor')
input_symbol= st.text_input('Enter Stock Symbol', 'TSLA')



df = yf.download(input_symbol, start=start, end=end)
df = df.reset_index()
df = df.drop(['Date', 'Adj Close'], axis=1)

st.subheader('Original Chart')
fig=plt.figure(figsize=(16,10))
plt.plot(df.Close)
st.pyplot(fig)

ma100 = df.Close.rolling(100).mean()
# ma100 
# plt.figure(figsize=(12,6))
# plt.plot(df.Close)
# plt.plot(ma100, 'r')
ma200 = df.Close.rolling(200).mean()
# plt.figure(figsize=(12,6))
# plt.plot(df.Close)
# plt.plot(ma100, 'r')
# plt.plot(ma200, 'g')


# Training and Testing
data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)

# Splitting data into x_train And y_train
# x_train = []
# y_train = []
# for i in range(100, data_training.shape[0]):
#     x_train.append(data_training_array[i-100:i])
#     y_train.append(data_training_array[i,0])
# x_train, y_train = np.array(x_train), np.array(y_train)

# from keras.layers import Dense, Dropout, LSTM
# from keras.models import Sequential
# model = Sequential()

# model.add(LSTM(units =50, activation='relu', return_sequences= True,
#                input_shape= (x_train.shape[1],1)))
# model.add(Dropout(0.2))
 
 
# model.add(LSTM(units =60, activation='relu', return_sequences= True,))
# model.add(Dropout(0.3))
 
 
# model.add(LSTM(units =80, activation='relu', return_sequences= True,))
# model.add(Dropout(0.4))
 
 
# model.add(LSTM(units =120, activation='relu'))
# model.add(Dropout(0.5))
 
# model.add(Dense(units=1))

# model.compile(optimizer='adam', loss='mean_squared_error')
# model.fit(x_train, y_train, epochs=50)

# model.save('keras_model.h5')



# To save time we already trained model and load it
model= load_model('trained_model.h5')

# Tsting
past_100_days = data_training.tail(100)
final_df= past_100_days.append(data_testing, ignore_index=True)

input_data = scaler.fit_transform(final_df)
# input_data
x_test=[]
y_test=[]
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
    
x_test, y_test=np.array(x_test), np.array(y_test)

# Prediction
y_predicted=model.predict(x_test)

scaler=scaler.scale_
scale_factor= 1/(scaler[0])
y_predicted= y_predicted * scale_factor
y_test =y_test * scale_factor

st.subheader('Predicted Graph')
fig=plt.figure(figsize=(16,10))
plt.plot(y_test, label= 'Original Price')
plt.plot(y_predicted, label= 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

