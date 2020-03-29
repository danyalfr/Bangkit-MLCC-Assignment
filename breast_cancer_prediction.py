import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix

# DATA PRE-PROCESSING
# ----------------------------------------------------------------------------------------------------

df = pd.read_csv("Breast_cancer_data.csv")

corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20, 20))
# plot heat map
g = sns.heatmap(df[top_corr_features].corr(), annot=True, cmap="RdYlGn")

X = df[["mean_perimeter", "mean_texture"]].values
y = df['diagnosis'].values

# Split dataset into training, validation, and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)

# Scaling the data
scaler = MinMaxScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
# ----------------------------------------------------------------------------------------------------

# Create the model

# ----------------------------------------------------------------------------------------------------
model = Sequential()  # NN Model

model.add(Dense(units=2, activation='relu'))  # Input Layer
model.add(Dense(units=2, activation='relu'))  # Hidden Layer
model.add(Dense(units=1, activation='sigmoid'))  # Output Layer

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# ----------------------------------------------------------------------------------------------------

# Training the model
# ----------------------------------------------------------------------------------------------------

early_stop = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10)
model.fit(x=X_train, y=y_train, epochs=500, validation_test=(X_val, y_val), verbose=1, callbacks=[early_stop])
loss = pd.DataFrame(model.history.history)
loss.plot()
# ----------------------------------------------------------------------------------------------------

# Model Evaluation
# ----------------------------------------------------------------------------------------------------

predictions = (model.predict(X_test) > 0.5).astype("int32")
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
# ----------------------------------------------------------------------------------------------------
