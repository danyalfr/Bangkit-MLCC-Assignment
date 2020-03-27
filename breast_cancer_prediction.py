import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
from tensorflow import feature_column
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# DATA PRE-PROCESSING
# ----------------------------------------------------------------------------------------------------
df = pd.read_csv("Breast_cancer_data.csv")

X = df.drop('diagnosis', axis=1).values  # Column X consist of features except for diagnosis (label)
y = df['diagnosis'].values  # data label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3,
                                                    random_state=42)  # Split dataset into training and testing
# Scaling the data
scaler = MinMaxScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# ----------------------------------------------------------------------------------------------------

