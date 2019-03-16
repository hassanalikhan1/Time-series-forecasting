import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

##............................DATA PREPROCESSING and CLEANING.................................##

def parseDate(x):
	return datetime.strptime(x,"%Y-%m-%d")

# convert series to supervised learning
# convert time series into supervised learning problem
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# transform series into train and test sets for supervised learning
def prepare_data(series, n_test, n_lag, n_seq):
	# extract raw values
	raw_values = series.values
	# raw_values = raw_values.reshape(len(raw_values), 1)
	# transform into supervised learning problem X, y
	supervised = series_to_supervised(raw_values, n_lag, n_seq)
	print(supervised.info())
	supervised_values = supervised.values
	# split into train and test sets
	train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
	return train, test, supervised

# fit an LSTM network to training data
def fit_lstm(train, test, n_lag, n_seq, n_batch, nb_epoch, n_neurons):
	# reshape training into [samples, timesteps, features]
	train_X, train_Y = train[:, 0:n_lag*n_seq], train[:, n_lag*n_seq:]
	test_X, test_Y = test[:, 0:n_lag*n_seq], test[:, n_lag*n_seq:]

	print(train_X.shape,train_Y.shape,test_X.shape,test_Y.shape)

	train_X = train_X.reshape(train_X.shape[0], 1, train_X.shape[1])
	test_X = test_X.reshape(test_X.shape[0], 1, test_X.shape[1])

	# design network
	model = Sequential()
	model.add(LSTM(150, input_shape=(train_X.shape[1], train_X.shape[2])))
	model.add(Dense(train_Y.shape[1]))
	model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
	# fit network
	history = model.fit(train_X, train_Y, epochs=100, validation_data=(test_X, test_Y),batch_size=50, verbose=2, shuffle=False)
	# plot history
	# plt.plot(history.history['loss'], label='train')
	# plt.plot(history.history['val_loss'], label='test')
	# plt.legend()
	# plt.show()
	return model

##function to forecast values of target and features for n number of weeks
def make_forecast(data,weeks,shift_by,n_lag,n_seq):
	
	predictions=[]
	temp=data[len(data)-1,:]

	for i in range(weeks):
		testVec=np.roll(temp,-shift_by)
		test=testVec[0:n_lag*n_seq]
		test=test.reshape(1,1,test.size)
		prediction=model.predict(test)

		testVec[n_lag*n_seq:]=prediction;
		predictions.append(testVec)
		temp=testVec

	return predictions



df_timeseries = pd.read_csv("sample_data_arw.csv",low_memory=False,parse_dates=['date'])

#sorting the values firstly, on component vlaues and then on their dates
df_timeseries=df_timeseries.sort_values(by=['part','date'],ascending=[1,1])

#Handlig the missing and zero values
df_timeseries=df_timeseries.fillna(method='ffill')
df_timeseries=df_timeseries.fillna(method='bfill')

df_timeseries=df_timeseries.replace(to_replace=0,method='ffill')
df_timeseries=df_timeseries.replace(to_replace=0,method='bfill')

df_timeseries=df_timeseries.reset_index()

df_timeseries=df_timeseries.drop('featureG_avg',axis=1)
df_timeseries=df_timeseries.drop('Unnamed: 0',axis=1)
df_timeseries=df_timeseries.drop('index',axis=1)

df_timeseries=df_timeseries.drop(['part_category_1','part_category_2','part_category_3'],axis=1)
df_timeseries=df_timeseries.drop(['featureA_max','featureB_max','featureC_max','featureD_max','featureE_max','featureF_max'],axis=1)



cols=df_timeseries.columns
nums=[0,1,2,3,4,5,6,7,8]
##interger encode direction
encoder=preprocessing.LabelEncoder()
vals=df_timeseries.values
#vals[:,0]=encoder.fit_transform(vals[:,0])

selector=[x for x in range(vals.shape[1]) if x!= 0 and x!=1]


## SCALING THE FEATURES AND TARGET ONLY
scaler=preprocessing.MinMaxScaler(feature_range=(0,1))
vals_wo_date=scaler.fit_transform(vals[:,selector])

df_timeseries2=pd.DataFrame(vals_wo_date)
vals_temp=vals[:,[0,1]]
df_temp=pd.DataFrame(vals_temp)

df_timeseries=pd.concat([df_temp,df_timeseries2],axis=1)
df_timeseries.columns=cols

print(df_timeseries.info())


###creating separate matrixes for each part(component)
grouped=df_timeseries.groupby('part')
l_grouped=list(grouped)


n_lag = 4
n_seq = 1
n_test = 10
n_features=7
n_weeks=24

joined_df=pd.DataFrame()
for i in range(20):
	# prepare data
	print(l_grouped[i][1].describe())
	parts=l_grouped[i][1]['part']
	l_grouped[i][1].drop(['part','date'],axis=1,inplace=True)
	#l_grouped[0][1]=l_grouped[0][1].reindex(sorted(l_grouped[0][1].columns),axis=1)
	train, test,supervised = prepare_data(l_grouped[i][1], n_test, n_lag, n_seq)
	# fit model
	model = fit_lstm(train,test, n_features, n_lag, 1, 10, 1)
	forecasts=make_forecast(test,n_weeks,n_features,n_lag,n_features)

	##append in to our original supervised timeseries
	predictFrame=pd.DataFrame(forecasts)
	predictFrame.columns=supervised.columns

	predictFrame=pd.concat([supervised,predictFrame])

	predictVals=predictFrame.values

	predictVals=scaler.inverse_transform(predictVals[:,n_lag*n_features:])

	predictFrame=pd.DataFrame(predictVals)

	predictFrame.columns=l_grouped[i][1].columns

	##to csv
	csvVals=predictFrame.values[-n_weeks:,:]
	csv_df=pd.DataFrame(csvVals[-n_weeks:,:])
	csv_df.columns=predictFrame.columns

	csv_df=csv_df.reset_index(drop=True)

	new_df=pd.DataFrame(parts[:n_weeks],columns=['part'])
	new_df=new_df.reset_index(drop=True)
	csv_df=pd.concat([csv_df,new_df],axis=1)

	csv_df=csv_df.reset_index(drop=True)
	joined_df=joined_df.reset_index(drop=True)

	joined_df=pd.concat([joined_df,csv_df],axis=0)

joined_df.to_csv('forecast.csv')

	##plot the line graph to show change in target
	# fig, ax = plt.subplots()
	# plt.plot(predictFrame['target'][:-n_weeks],color='black')
	# plt.plot(predictFrame['target'][-n_weeks:],color='red')

	# plt.legend()
	# plt.show()




print (predictFrame)


