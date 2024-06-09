import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, ConfusionMatrixDisplay,\
     cohen_kappa_score, matthews_corrcoef
from ElectraLocalLoad import Pre_train_texts
from imblearn.over_sampling import ADASYN
from scipy.stats import spearmanr
import keras.backend as K
import time
import warnings

warnings.filterwarnings("ignore")

seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

data = pd.read_csv('the path of Dataset', encoding='UTF-8')
Pre_train_ModelPath = "the path of Electra model"

sentence_vector = Pre_train_texts(data, Pre_train_ModelPath)
sentence_vector = [tensor.tolist() for tensor in sentence_vector]
x1 = pd.DataFrame(sentence_vector)

x2, y = data.iloc[:, 4:-1], data.iloc[:, -1]    # class_level from 4：-1   method_level from 5：-1

combine_data = pd.concat([x1, x2, y], axis=1)
# combine_data.to_csv("the new path of dataset", index=False)  # save or not
# print(combine_data.shape)  

combine_data.columns = combine_data.columns.astype(str)
x, y = combine_data.iloc[:, 0:-1], combine_data.iloc[:, -1]

# Data Balance for class_lever dataset
# syn = ADASYN(sampling_strategy='minority', random_state=42)  
# x, y = syn.fit_resample(x, y) 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

x_train = x_train.values.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.values.reshape((x_test.shape[0], x_test.shape[1], 1))

x1_train = x_train[:, 0:1024]
x2_train = x_train[:, 1024:]
x1_test = x_test[:, 0:1024]
x2_test = x_test[:, 1024:]

one_hot = OneHotEncoder(sparse=False)
y_train = one_hot.fit_transform(y_train.values.reshape(len(y_train), 1))
y_test = one_hot.fit_transform(y_test.values.reshape(len(y_test), 1))

num_classes = 3 
input_x1 = Input(shape=(x1_train.shape[1:]))
input_x2 = Input(shape=(x2_train.shape[1:]))
print(input_x1.shape)
print(input_x2.shape)

# BiLSTM
l1 = Bidirectional(LSTM(32, activation='tanh', return_sequences=True))(input_x1)
l1 = Bidirectional(LSTM(64, activation='tanh', return_sequences=True))(l1)
l1 = Flatten()(l1)
l1 = Dense(256, activation='relu')(l1)
attention = Dense(1, activation='tanh')(l1)
attention = Flatten()(attention)
attention = Activation('softmax')(attention)
attention = RepeatVector(256)(attention)
attention = Permute([2, 1])(attention)
attention = Multiply()([l1, attention]) 
attention = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(256, ))(attention)
l1 = Flatten()(attention)

#  CNN
l2 = Conv1D(filters=32, kernel_size=6, strides=1, padding='same', activation='tanh')(input_x2)
l2 = BatchNormalization()(l2)
l2 = Conv1D(filters=64, kernel_size=6, strides=1, padding='same', activation='relu')(l2)
l2 = Conv1D(filters=64, kernel_size=6, strides=1, padding='same', activation='relu')(l2)
l2 = LeakyReLU(alpha=0.33)(l2)
l2 = Dropout(0.5)(l2)
l2 = BatchNormalization()(l2)
l2 = Conv1D(filters=128, kernel_size=6, strides=1, padding='same', activation='relu')(l2)
l2 = Conv1D(filters=128, kernel_size=6, strides=1, padding='same', activation='relu')(l2)
l2 = LeakyReLU(alpha=0.33)(l2)
l2 = Dropout(0.5)(l2)
l2 = BatchNormalization()(l2)
l2 = Conv1D(filters=256, kernel_size=6, strides=1, padding='same', activation='relu')(l2)
l2 = Conv1D(filters=256, kernel_size=6, strides=1, padding='same', activation='relu')(l2)
l2 = LeakyReLU(alpha=0.33)(l2)
l2 = Dropout(0.5)(l2)
l2 = BatchNormalization()(l2)
l2 = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(l2)
l2 = Flatten()(l2)

# Concatenate
model_together = concatenate([l1, l2], axis=1)
output = Dense(128, activation='relu')(model_together)
output = LeakyReLU(alpha=0.33)(output)
attention = Dense(1, activation='tanh')(output)
attention = Flatten()(attention)
attention = Activation('softmax')(attention)
attention = RepeatVector(128)(attention)
attention = Permute([2, 1])(attention)
attention = Multiply()([output, attention])
attention = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(128, ))(attention)
output = Dense(num_classes, kernel_regularizer=regularizers.l2(0.01), activation='softmax')(attention)
model = Model(inputs=[input_x1, input_x2], outputs=output)

model.summary()
model.compile(optimizer=Adam(0.001),
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])
lr_reduce = tf.keras.callbacks.ReduceLROnPlateau('val_loss', patience=3, factor=0.5, min_lr=0.000001)

start = time.time()
history = model.fit([x1_train, x2_train],
                    y_train,
                    epochs=60,
                    batch_size=16,
                    validation_data=([x1_test, x2_test], y_test),
                    callbacks=[lr_reduce],
                    shuffle=True)
end = time.time()
total_time = end - start

score = model.evaluate([x1_train, x2_train], y_train, verbose=0)
print('train loss:', score[0])
print('train accuracy:', score[1])

print(model.evaluate([x1_test, x2_test], y_test, verbose=0))
print("train_time: ", total_time, "seconds")

y_test = np.argmax(y_test, axis=1)  
predict = model.predict([x1_test, x2_test], verbose=0) 
y_pred = np.argmax(predict, axis=1) 
Confusion_matrix = confusion_matrix(y_test, y_pred)
print(Confusion_matrix)
print(classification_report(y_test, y_pred, digits=5))

P_m = precision_score(y_test, y_pred, average='macro')
R_m = recall_score(y_test, y_pred, average='macro')
F1_m = f1_score(y_test, y_pred, average='macro')
print('precision-macro : ', P_m, '\n', 'recall-macro :  ', R_m, '\n', 'F1-macro  : ', F1_m, '\n')

P_w = precision_score(y_test, y_pred, average='weighted')
R_w = recall_score(y_test, y_pred, average='weighted')
F1_w = f1_score(y_test, y_pred, average='weighted')
print('precision-weighted : ', P_w, '\n', 'recall-weighted :  ', R_w, '\n', 'F1-weighted  : ', F1_w, '\n')

corr, p_value = spearmanr(y_test, y_pred)
print("spearman：", corr)
print('P_value：', p_value)

Confusion_matrix_display = ConfusionMatrixDisplay(Confusion_matrix)
Confusion_matrix_display.plot()
plt.show()
plt.plot(history.epoch, history.history.get('categorical_accuracy'), label='categorical_accuracy')
plt.plot(history.epoch, history.history.get('val_categorical_accuracy'), label='val_categorical_accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()
plt.plot(history.epoch, history.history.get('loss'), label='loss')
plt.plot(history.epoch, history.history.get('val_loss'), label='val_loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# save model
# model_path = "the path of trained model"
# model.save(model_path)
