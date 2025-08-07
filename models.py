def random_forest_reg():
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor()
    return rf



def linear_regression():
    
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
 
    return lr



def svc():
    from sklearn.svm import SVC
    svm =SVC(C=100,gamma=0.001,kernel='rbf')

    return svm

def random_forest_clf():
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=500,criterion='entropy')

    return rf



##################----------- NNs-----------##################
from scikeras.wrappers import KerasClassifier, KerasRegressor


   
def keras_mlp_regressor(input_shape):
    '''
    paizei kala mono gia scaled !!!!!!!!!!!!!!!!!!!!!
    
    '''
    import tensorflow as tf
    from tensorflow import keras
    from keras.models import Sequential
    from keras.layers import Flatten,Dense

    mlp = Sequential()
    mlp.add(Dense(256, activation='sigmoid', input_shape=input_shape))
    # Dense layer 2 (128 neurons)
    mlp.add(Dense(128, activation='sigmoid'))
    mlp.add(Dense(64, activation='sigmoid'))
    #mlp.add(Dense(32, activation='sigmoid'))
    # Output layer (10 classes)
    mlp.add(Dense(10, activation='sigmoid'))
    mlp.add(Dense(1, activation='linear'))

    mlp.compile(loss="mean_absolute_error", optimizer="adam")
    
    return mlp

def keras_mlp_classifier(input_shape):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense

    model = Sequential()
    model.add(Dense(256, activation='sigmoid', input_shape=input_shape))
    model.add(Dense(128, activation='sigmoid'))
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dense(4))  # No activation for logits
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    
    return model


def keras_cnn_regressor(input_shape):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dropout, Dense

    model = Sequential([
        Conv1D(16, 3, padding='same', activation='relu', input_shape=input_shape),
        MaxPooling1D(),
        Conv1D(32, 3, padding='same', activation='relu'),
        MaxPooling1D(),
        Conv1D(64, 3, padding='same', activation='relu'),
        MaxPooling1D(),
        Flatten(),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss="mean_absolute_error")
    
    return model


def keras_cnn_classifier(input_shape):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dropout, Dense

    model = Sequential([
        Conv1D(16, 3, padding='same', activation='relu', input_shape=input_shape),
        MaxPooling1D(),
        Conv1D(32, 3, padding='same', activation='relu'),
        MaxPooling1D(),
        Conv1D(64, 3, padding='same', activation='relu'),
        MaxPooling1D(),
        Flatten(),
        Dropout(0.2),
        Dense(4)
    ])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])        
    
    return model


def keras_lstm_regressor(input_shape):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dropout, Dense, LSTM
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(50),
        Dropout(0.3),
        Dense(50, activation="relu"),
        Dense(1)
    ])
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    
    return model


def keras_lstm_classifier(input_shape):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dropout, Dense, LSTM
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(50),
        Dropout(0.3),
        Dense(50, activation="relu"),
        Dense(4, activation="softmax")
    ])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
    
    return model