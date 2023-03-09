import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

df=pd.read_csv(r'C:\Users\prana\Downloads\Python data science\Refactored_Py_DS_ML_Bootcamp-master\22-Deep Learning\TensorFlow_FILES\DATA\cancer_classification.csv')

df.head()

df.describe().transpose()

sns.countplot(x='benign_0__mal_1',data=df)

df.corr()['benign_0__mal_1'].sort_values(ascending=True)

df.corr()['benign_0__mal_1'].sort_values().plot(kind='bar')

plt.figure(figsize=(12,8))
sns.heatmap(df.corr())

X=df.drop('benign_0__mal_1',axis=1).values
y=df['benign_0__mal_1'].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)

from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()

X_train=scaler.fit_transform(X_train)

X_test=scaler.transform(X_test)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout

X_train.shape

model = Sequential()

# https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw

model.add(Dense(units=30,activation='relu'))

model.add(Dense(units=15,activation='relu'))


model.add(Dense(units=1,activation='sigmoid'))

# For a binary classification problem
model.compile(loss='binary_crossentropy', optimizer='adam')

model.fit(x=X_train, 
          y=y_train, 
          epochs=600,
          validation_data=(X_test, y_test), verbose=1
          )

loses=pd.DataFrame(model.history.history)

loses.plot()   ### this indicating overfitting data means we are using too much epochs

loses

model = Sequential()

# https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw

model.add(Dense(units=30,activation='relu'))

model.add(Dense(units=15,activation='relu'))


model.add(Dense(units=1,activation='sigmoid'))

# For a binary classification problem
model.compile(loss='binary_crossentropy', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping

help(EarlyStopping)

early_stop=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=25)

model.fit(x=X_train, 
          y=y_train, 
          epochs=600,
          validation_data=(X_test, y_test), verbose=1,
          callbacks=[early_stop])

model_loss=pd.DataFrame(model.history.history)

model_loss.plot()

from tensorflow.keras.layers import Dropout

model = Sequential()

# https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw

model.add(Dense(units=30,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=15,activation='relu'))
model.add(Dropout(0.5))


model.add(Dense(units=1,activation='sigmoid'))

# For a binary classification problem
model.compile(loss='binary_crossentropy', optimizer='adam')

model.fit(x=X_train, 
          y=y_train, 
          epochs=600,
          validation_data=(X_test, y_test), verbose=1,
          callbacks=[early_stop])

model_loss=pd.DataFrame(model.history.history)

model_loss.plot()

predict_x=model.predict(X_test) 
classes=np.argmax(predict_x,axis=1)

classes

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,classes))

print(classification_report(y_test,classes))
