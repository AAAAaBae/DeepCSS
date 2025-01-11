import time
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, ConfusionMatrixDisplay,\
     cohen_kappa_score, matthews_corrcoef, matthews_corrcoef, precision_score, recall_score, f1_score, precision_recall_curve, auc
from ElectraLocalLoad import Pre_train_texts
from imblearn.over_sampling import ADASYN
import keras.backend as K
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

x, y = data.iloc[:, 0:-1], data.iloc[:, -1]
print(len(x), len(y), len(data))

syn = ADASYN(sampling_strategy='minority', random_state=42)
x, y = syn.fit_resample(x, y)  
print(len(y[y == 0]))
print(len(y[y == 1]))  
print(len(y[y == 2]))

x = x.values.reshape((x.shape[0], x.shape[1], 1))
print(x.shape)

x1 = x[:, 0:1024]
x2 = x[:, 1024:]

one_hot = OneHotEncoder(sparse=False)
y = one_hot.fit_transform(y.values.reshape(len(y), 1))
print(y.shape)

num_classes = 3 
k_fold = 5
kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)
cv_accuracy = []  
cv_losses = []  
cv_recall = []  
cv_precision = []
cv_f1_score = []
cv_MCC = []

fold = 0
best_fold = 0
best_fold_f1 = 0

os.environ["CUDA_VISIBLE_DEVICES"] = '0' 
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.9  
config.gpu_options.allow_growth = True  
sess = tf.compat.v1.Session(config=config)

start = time.time()
for train_index, test_index in kf.split(x):
    x1_train, x1_test = x1[train_index], x1[test_index]
    x2_train, x2_test = x2[train_index], x2[test_index]
    y_train, y_test = y[train_index], y[test_index]

    input_x1 = Input(shape=(x1_train.shape[1:]))
    input_x2 = Input(shape=(x2_train.shape[1:]))

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

    history = model.fit([x1_train, x2_train],
                        y_train,
                        epochs=60,  
                        batch_size=8,
                        validation_data=([x1_test, x2_test], y_test),
                        callbacks=[lr_reduce],
                        shuffle=True)
     
    scores = model.evaluate([x1_test, x2_test], y_test, verbose=0)
    cv_losses.append(scores[0]) 
    cv_accuracy.append(scores[1])  

    y_test = np.argmax(y_test, axis=1)  
    predict = model.predict([x1_test, x2_test], verbose=0)  
    y_pred = np.argmax(predict, axis=1)  
    Confusion_matrix = confusion_matrix(y_test, y_pred)
    print(Confusion_matrix)
    print(classification_report(y_test, y_pred, digits=5))

    P_m = precision_score(y_test, y_pred, average='macro')
    cv_precision.append(P_m)

    R_m = recall_score(y_test, y_pred, average='macro')
    cv_recall.append(R_m)

    F1_m = f1_score(y_test, y_pred, average='macro')
    cv_f1_score.append(F1_m)

    mcc = matthews_corrcoef(y_test, y_pred)
    cv_MCC.append(mcc)

    if F1_m > best_fold_f1:
        best_fold_f1 = F1_m
        best_fold = fold
        model.save('best_fold_model.h5') 
    fold += 1
end = time.time()
total_time = end - start
print("Total time：", total_time)
print("Accuracy of 5-fold：", cv_accuracy)
print("Loss of 5-fold：", cv_losses)
print("Precision of 5-fold：", cv_precision)
print("Recall of 5-fold：", cv_recall)
print('F1 of 5-fold：', cv_f1_score)
print("MCC of 5-fold：", cv_MCC)

train_idx, test_idx = list(kf.split(x))[best_fold]
x1_test = x1[test_idx]
x2_test = x2[test_idx]
y_test = y[test_idx]
loaded_model = load_model('best_fold_model.h5')

y_test_l = np.argmax(y_test, axis=1)  
predict_l = loaded_model.predict([x1_test, x2_test], verbose=0)  
y_pred_l = np.argmax(predict_l, axis=1)  
Confusion_matrix = confusion_matrix(y_test_l, y_pred_l)
print(Confusion_matrix)
print(classification_report(y_test_l, y_pred_l, digits=5))

pr_auc = dict()
precision_dict = dict()
recall_dict = dict()

for i in range(num_classes):
    precision_dict[i], recall_dict[i], _ = precision_recall_curve(y_test[:, i], predict_l[:, i])
    pr_auc[i] = auc(recall_dict[i], precision_dict[i])
    plt.plot(recall_dict[i], precision_dict[i], label='class {0} (PR - AUC = {1:0.4f})'.format(i, pr_auc[i]))
print("Each severity class PR - AUC:", pr_auc)

precision_dict["macro"], recall_dict["macro"], _ = precision_recall_curve(y_test.ravel(), predict_l.ravel())
pr_auc["macro"] = auc(recall_dict["macro"], precision_dict["macro"])
plt.plot(recall_dict['macro'], precision_dict['macro'], label='macro-averaged PR-AUC curve (area = {0:0.4f})'.format(pr_auc["macro"]))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision - Recall Curves')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.legend(loc="lower left")
plt.savefig('PR_curve.jpg', dpi=500)
