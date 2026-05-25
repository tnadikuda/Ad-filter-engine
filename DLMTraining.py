import pandas as pd
import numpy as np
import re, os
from string import printable
from sklearn import model_selection

import tensorflow as tf
from keras.models import Sequential, Model, model_from_json, load_model
from keras import regularizers
from keras.layers import Dense, Dropout, Activation, Lambda, Flatten
from keras.layers import Input, ELU, LSTM, Embedding, Convolution2D, MaxPooling2D, \
    BatchNormalization, Convolution1D, MaxPooling1D, concatenate
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import SGD, Adam, RMSprop
from keras import backend as K
from keras.regularizers import l2
from keras.layers import SpatialDropout1D, SpatialDropout2D
from keras.layers import Bidirectional
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

## Load data
df = pd.read_csv('data/url_data_DL.csv')
print(len(df))

# Data Preparation
url_int_tokens = [[printable.index(x) + 1 for x in url if x in printable] for url in df.url]
max_len = 75
X = pad_sequences(url_int_tokens, maxlen=max_len)
target = np.array(df.isMalicious)

print('Matrix dimensions of X: ', X.shape, 'Vector dimension of target: ', target.shape)

X_train, X_test, target_train, target_test = model_selection.train_test_split(X, target, test_size=0.25, random_state=33)

def print_layers_dims(model):
    l_layers = model.layers
    for i in range(len(l_layers)):
        print(l_layers[i])
        print('Input Shape: ', l_layers[i].input_shape, 'Output Shape: ', l_layers[i].output_shape)

# Creating holders to store the model performance results
ML_Model = []
accuracy_list = []
f1_score_list = []
recall_list = []
precision_list = []

def storeResults(model, a, b, c, d):
    ML_Model.append(model)
    accuracy_list.append(round(a, 3))
    f1_score_list.append(round(b, 3))
    recall_list.append(round(c, 3))
    precision_list.append(round(d, 3))

## Architecture A - Simple LSTM
def simple_lstm(max_len=75, emb_dim=32, max_vocab_len=100, lstm_output_size=32, W_reg=regularizers.l2(1e-4)):
    main_input = Input(shape=(max_len,), dtype='int32', name='main_input')
    emb = Embedding(input_dim=max_vocab_len, output_dim=emb_dim, input_length=max_len,
                    embeddings_regularizer=l2(1e-4))(main_input)
    emb = SpatialDropout1D(0.2)(emb)
    lstm = LSTM(lstm_output_size)(emb)
    lstm = Dropout(0.5)(lstm)
    output = Dense(1, activation='sigmoid', name='output')(lstm)
    model = Model(inputs=[main_input], outputs=[output])
    adam = Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    return model

epochs = 5
batch_size = 64
model = simple_lstm()
model.fit(X_train, target_train, epochs=epochs, batch_size=batch_size)
loss, accuracy = model.evaluate(X_test, target_test, verbose=1)
accuracy_var = accuracy
print('\nFinal Cross-Validation Accuracy', accuracy, '\n')

from sklearn.metrics import precision_score, recall_score, f1_score
raw_predictions_1 = model.predict(X_test)
fpr_1, tpr_1, thresholds_1 = roc_curve(target_test, raw_predictions_1)
roc_auc_1 = auc(fpr_1, tpr_1)
print("AUC Score1:", roc_auc_1)

predictions = model.predict(X_test)
binary_predictions = (predictions > 0.5).astype(int)
f1_var = f1_score(target_test, binary_predictions)
recall_var = recall_score(target_test, binary_predictions)
precision_var = precision_score(target_test, binary_predictions)
storeResults('Simple LSTM', accuracy_var, f1_var, recall_var, precision_var)
model.save('models/simple_LSTM.h5')

## Architecture B - 1D Convolution + LSTM
def lstm_conv(max_len=75, emb_dim=32, max_vocab_len=100, lstm_output_size=32, W_reg=regularizers.l2(1e-4)):
    main_input = Input(shape=(max_len,), dtype='int32', name='main_input')
    emb = Embedding(input_dim=max_vocab_len, output_dim=emb_dim, input_length=max_len,
                    embeddings_regularizer=l2(1e-4))(main_input)
    emb = SpatialDropout1D(0.25)(emb)
    conv = Convolution1D(kernel_size=8, filters=256, padding='same')(emb)
    conv = ELU()(conv)
    conv = MaxPooling1D(pool_size=4)(conv)
    conv = Dropout(0.5)(conv)
    lstm = LSTM(lstm_output_size)(conv)
    lstm = Dropout(0.5)(lstm)
    output = Dense(1, activation='sigmoid', name='output')(lstm)
    model = Model(inputs=[main_input], outputs=[output])
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    return model

epochs = 5
batch_size = 32
model = lstm_conv()
model.fit(X_train, target_train, epochs=epochs, batch_size=batch_size)
loss, accuracy = model.evaluate(X_test, target_test, verbose=1)
accuracy_var = accuracy

raw_predictions_2 = model.predict(X_test)
fpr_2, tpr_2, thresholds_2 = roc_curve(target_test, raw_predictions_2)
roc_auc_2 = auc(fpr_2, tpr_2)
print("AUC Score2:", roc_auc_2)

binary_predictions = (model.predict(X_test) > 0.5).astype(int)
f1_var = f1_score(target_test, binary_predictions)
recall_var = recall_score(target_test, binary_predictions)
precision_var = precision_score(target_test, binary_predictions)
storeResults('1D Convolution and LSTM', accuracy_var, f1_var, recall_var, precision_var)
model.save('models/LSTMwith1DConv.h5')

## Architecture C - 1D Convolutions + Fully Connected
def conv_fully(max_len=75, emb_dim=32, max_vocab_len=100, W_reg=regularizers.l2(1e-4)):
    main_input = Input(shape=(max_len,), dtype='int32', name='main_input')
    emb = Embedding(input_dim=max_vocab_len, output_dim=emb_dim, input_length=max_len,
                    embeddings_regularizer=l2(1e-4))(main_input)
    emb = SpatialDropout1D(0.25)(emb)

    def sum_1d(X):
        return K.sum(X, axis=1)

    def get_conv_layer(emb, kernel_size=5, filters=256):
        conv = Convolution1D(kernel_size=kernel_size, filters=filters, padding='same')(emb)
        conv = ELU()(conv)
        conv = Lambda(sum_1d, output_shape=(filters,))(conv)
        conv = Dropout(0.5)(conv)
        return conv

    conv1 = get_conv_layer(emb, kernel_size=2, filters=256)
    conv2 = get_conv_layer(emb, kernel_size=3, filters=256)
    conv3 = get_conv_layer(emb, kernel_size=4, filters=256)
    conv4 = get_conv_layer(emb, kernel_size=5, filters=256)
    merged = concatenate([conv1, conv2, conv3, conv4], axis=1)
    hidden1 = Dense(1024)(merged)
    hidden1 = ELU()(hidden1)
    hidden1 = BatchNormalization(axis=-1)(hidden1)
    hidden1 = Dropout(0.5)(hidden1)
    hidden2 = Dense(1024)(hidden1)
    hidden2 = ELU()(hidden2)
    hidden2 = BatchNormalization(axis=-1)(hidden2)
    hidden2 = Dropout(0.5)(hidden2)
    output = Dense(1, activation='sigmoid', name='output')(hidden2)
    model = Model(inputs=[main_input], outputs=[output])
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    return model

epochs = 5
batch_size = 32
model = conv_fully()
model.fit(X_train, target_train, epochs=epochs, batch_size=batch_size)
loss, accuracy = model.evaluate(X_test, target_test, verbose=1)
accuracy_var = accuracy

raw_predictions_3 = model.predict(X_test)
fpr_3, tpr_3, thresholds_3 = roc_curve(target_test, raw_predictions_3)
roc_auc_3 = auc(fpr_3, tpr_3)
print("AUC Score3:", roc_auc_3)

binary_predictions = (model.predict(X_test) > 0.5).astype(int)
f1_var = f1_score(target_test, binary_predictions)
recall_var = recall_score(target_test, binary_predictions)
precision_var = precision_score(target_test, binary_predictions)
storeResults('1D Convolutions and Fully Connected Layers', accuracy_var, f1_var, recall_var, precision_var)
model.save('models/FullyConnectedLayerswith1DConv.h5')

## Architecture D - 1D Conv + BiLSTM (Best Model)
def bilstm_conv(max_len=75, emb_dim=32, max_vocab_len=100, lstm_output_size=32, W_reg=regularizers.l2(1e-4)):
    main_input = Input(shape=(max_len,), dtype='int32', name='main_input')
    emb = Embedding(input_dim=max_vocab_len, output_dim=emb_dim, input_length=max_len,
                    embeddings_regularizer=l2(1e-4))(main_input)
    emb = SpatialDropout1D(0.25)(emb)
    conv = Convolution1D(kernel_size=8, filters=256, padding='same')(emb)
    conv = ELU()(conv)
    conv = MaxPooling1D(pool_size=4)(conv)
    conv = Dropout(0.5)(conv)
    lstm = Bidirectional(LSTM(lstm_output_size))(conv)
    lstm = Dropout(0.5)(lstm)
    output = Dense(1, activation='sigmoid', name='output')(lstm)
    model = Model(inputs=[main_input], outputs=[output])
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    return model

epochs = 5
batch_size = 32
model = bilstm_conv()
model.fit(X_train, target_train, epochs=epochs, batch_size=batch_size)
loss, accuracy = model.evaluate(X_test, target_test, verbose=1)
accuracy_var = accuracy
print('\nFinal Cross-Validation Accuracy', accuracy, '\n')
model.save('models/1DConvWiithBiLSTM.h5')

raw_predictions_4 = model.predict(X_test)
fpr_4, tpr_4, thresholds_4 = roc_curve(target_test, raw_predictions_4)
roc_auc_4 = auc(fpr_4, tpr_4)
print("AUC Score4:", roc_auc_4)

# Results comparison
result = pd.DataFrame({
    'ML Model': ML_Model,
    'Accuracy': accuracy_list,
    'f1_score': f1_score_list,
    'Recall': recall_list,
    'Precision': precision_list,
})
sorted_result = result.sort_values(by=['Accuracy', 'f1_score'], ascending=False).reset_index(drop=True)
print(sorted_result)

# ROC Curve plot
plt.figure()
lw = 2
plt.plot(fpr_4, tpr_4, color='darkorange', lw=lw, label='ROC curve CNN-BiLSTM (area = %0.2f)' % roc_auc_4)
plt.plot(fpr_3, tpr_3, color='skyblue', lw=lw, label='ROC curve CNN-FullyConnected (area = %0.2f)' % roc_auc_3)
plt.plot(fpr_2, tpr_2, color='green', lw=lw, label='ROC curve CNN-LSTM (area = %0.2f)' % roc_auc_2)
plt.plot(fpr_1, tpr_1, color='red', lw=lw, label='ROC curve Simple LSTM (area = %0.2f)' % roc_auc_1)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
plt.show()
