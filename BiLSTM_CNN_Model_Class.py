# Functional model模型搭建风格
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

# # 读取数据集和预训练模型路径
# data = pd.read_csv('E:/Deeplearning/MyTrial/DataSet/FeatureEnvy/FeatureEnvy_1.csv', encoding='UTF-8')
# Pre_train_ModelPath = "E:/Deeplearning/MyTrial/Preprocessing/ElectraPathPytorch"
# # 获取文本向量数据 x1
# sentence_vector = Pre_train_texts(data, Pre_train_ModelPath)
# sentence_vector = [tensor.tolist() for tensor in sentence_vector]
# x1 = pd.DataFrame(sentence_vector)
# # 获取度量数据和标签 x2 和 y
# x2, y = data.iloc[:, 4:-1], data.iloc[:, -1]    # 类级从4：-1   方法级从5：-1
# # 最后组合成为DataFrame格式
# combine_data = pd.concat([x1, x2, y], axis=1)
# combine_data.to_csv("E:/Deeplearning/MyTrial/DataSet/FeatureEnvy/FeatureEnvy_2.csv", index=False)  # 可以选择保存下来
# print(combine_data.shape)  # 查看合并后的数据形状
# # 获取数据（x为特征矩阵，y为标签）
# combine_data.columns = combine_data.columns.astype(str)
# x, y = combine_data.iloc[:, 0:-1], combine_data.iloc[:, -1]

# 实验测试快速读取数据，直接从保存的数据文件中读取
data = pd.read_csv('E:/Deeplearning/MyTrial/DataSet/LongMethod/LongMethod_2.csv', encoding='UTF-8')
x, y = data.iloc[:, 0:-1], data.iloc[:, -1]
print(len(x), len(y), len(data))

# # 数据样本平衡
syn = ADASYN(sampling_strategy='minority', random_state=42)
x, y = syn.fit_resample(x, y)   # x, y仍是数据框格式
print(len(y[y == 0]))
print(len(y[y == 1]))  # 查看平衡后的各类别样本数量
print(len(y[y == 2]))

# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#  转换下特征x矩阵x输入shape
x_train = x_train.values.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.values.reshape((x_test.shape[0], x_test.shape[1], 1))

x1_train = x_train[:, 0:1024]
x2_train = x_train[:, 1024:]
x1_test = x_test[:, 0:1024]
x2_test = x_test[:, 1024:]

# 转换标签y为独热编码和输入形状
one_hot = OneHotEncoder(sparse=False)
y_train = one_hot.fit_transform(y_train.values.reshape(len(y_train), 1))
y_test = one_hot.fit_transform(y_test.values.reshape(len(y_test), 1))

num_classes = 3  # 标签类别数
input_x1 = Input(shape=(x1_train.shape[1:]))  # 定义分支网络1输入形状shape
input_x2 = Input(shape=(x2_train.shape[1:]))  # 定义分支网络2输入形状shape
print(input_x1.shape)
print(input_x2.shape)

# 创建BiLSTM模型
l1 = Bidirectional(LSTM(32, activation='tanh', return_sequences=True))(input_x1)
l1 = Bidirectional(LSTM(64, activation='tanh', return_sequences=True))(l1)
l1 = Flatten()(l1)
l1 = Dense(256, activation='relu')(l1)
attention = Dense(1, activation='tanh')(l1)
attention = Flatten()(attention)
attention = Activation('softmax')(attention)
attention = RepeatVector(256)(attention)
attention = Permute([2, 1])(attention)
attention = Multiply()([l1, attention])    # 简单的点积注意力机制
attention = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(256, ))(attention)
l1 = Flatten()(attention)

# l1 = Bidirectional(LSTM(32, activation='tanh', return_sequences=True))(input_x1)
# l1 = Bidirectional(LSTM(64, activation='tanh', return_sequences=True))(l1)
# attention = Dense(1, activation='tanh')(l1)
# attention = Flatten()(attention)
# attention = Activation('softmax')(attention)
# attention = RepeatVector(128)(attention)
# attention = Permute([2, 1])(attention)
# attention = Multiply()([l1, attention])    # 简单的点积注意力机制
# attention = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(128, ))(attention)
# l1 = Dense(256, activation='relu')(attention)
# l1 = Flatten()(l1)

#  创建CNN模型
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

# 拼接两个模型,并加入注意力机制
model_together = concatenate([l1, l2], axis=1)
output = Dense(128, activation='relu')(model_together)
output = LeakyReLU(alpha=0.33)(output)
attention = Dense(1, activation='tanh')(output)
attention = Flatten()(attention)
attention = Activation('softmax')(attention)
attention = RepeatVector(128)(attention)
attention = Permute([2, 1])(attention)
attention = Multiply()([output, attention])    # 简单的点积注意力机制
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
                    epochs=50,
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
print("训练总时长: ", total_time, "seconds")

# 打印出混淆矩阵和分类报告
y_test = np.argmax(y_test, axis=1)  # 将独热编码转换为类别标签值，从0开始
predict = model.predict([x1_test, x2_test], verbose=0)  # 模型预测验证
y_pred = np.argmax(predict, axis=1)  # 将概率转化为类别标签值，从0，1，2开始
Confusion_matrix = confusion_matrix(y_test, y_pred)
print(Confusion_matrix)
print(classification_report(y_test, y_pred, digits=5))

# 单独打印出评估指标
P_m = precision_score(y_test, y_pred, average='macro')
R_m = recall_score(y_test, y_pred, average='macro')
F1_m = f1_score(y_test, y_pred, average='macro')
print('precision-macro : ', P_m, '\n', 'recall-macro :  ', R_m, '\n', 'F1-macro  : ', F1_m, '\n')

P_w = precision_score(y_test, y_pred, average='weighted')
R_w = recall_score(y_test, y_pred, average='weighted')
F1_w = f1_score(y_test, y_pred, average='weighted')
print('precision-weighted : ', P_w, '\n', 'recall-weighted :  ', R_w, '\n', 'F1-weighted  : ', F1_w, '\n')

# 其他常用的多分类评估指标
corr, p_value = spearmanr(y_test, y_pred)
print("spearman相关性系数：", corr)
print('P值：', p_value)

# 可视化混淆矩阵
Confusion_matrix_display = ConfusionMatrixDisplay(Confusion_matrix)
Confusion_matrix_display.plot()
plt.show()
# 可视化准确度
plt.plot(history.epoch, history.history.get('categorical_accuracy'), label='categorical_accuracy')
plt.plot(history.epoch, history.history.get('val_categorical_accuracy'), label='val_categorical_accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()
# 可视化损失率
plt.plot(history.epoch, history.history.get('loss'), label='loss')
plt.plot(history.epoch, history.history.get('val_loss'), label='val_loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# 保存训练好的最佳模型，以供泛化测试使用，进行对比分析。
# model_path1 = "E:/Deeplearning/MyTrial/Trained_Best_Model/DataClass/my_model双重注意力机制.h5"
# model_path2 = "E:/Deeplearning/MyTrial/Trained_Best_Model/LargeClass/my_model双重注意力机制.h5"
# model_path3 = "E:/Deeplearning/MyTrial/Trained_Best_Model/GodClass/my_model双重注意力机制2.h5"
# model.save(model_path1)
