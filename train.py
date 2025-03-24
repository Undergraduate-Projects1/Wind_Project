# -*- coding: utf-8 -*-

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import tensorflow as tf
from keras import Sequential, layers, utils

def predict_next(model, sample, epoch=20):
    temp1 = list(sample[:,0])
    for i in range(epoch):
        sample = sample.reshape(1, x_Seq_len, 1)
        pred = model.predict(sample)
        value = pred.tolist()[0][0]
        temp1.append(value)
        sample = np.array(temp1[i+1 : i+x_Seq_len+1])

    return temp1
def create_new_dataset(dataset, seq_len=12):
    '''基于原始数据集构造新的序列特征数据集
    Params:
        dataset : 原始数据集
        seq_len : 序列长度（时间跨度）
    Returns:
        X, y
    '''
    X = []  # 初始特征数据集为空列表
    y = []  # 初始标签数据集为空列表,y标签为样本的下一个点，即预测点

    start = 0  # 初始位置
    end = dataset.shape[0] - seq_len  # 截止位置,dataset.shape[0]就是有多少条

    for i in range(start, end):  # for循环构造特征数据集
        sample = dataset[i: i + seq_len]  # 基于时间跨度seq_len创建样本
        label = dataset[i + seq_len]  # 创建sample对应的标签
        X.append(sample)  # 保存sample
        y.append(label)  # 保存label
    # 返回特征数据集和标签集
    return np.array(X), np.array(y)
def split_dataset(X, y, train_ratio=0.8):
    '''基于X和y，切分为train和test
    Params:
        X : 特征数据集
        y : 标签数据集
        train_ratio : 训练集占X的比例

    Returns:git init

        X_train, X_test, y_train, y_test
    '''
    X_len = len(X)  # 特征数据集X的样本数量
    train_data_len = int(X_len * train_ratio)  # 训练集的样本数量

    X_train = X[:train_data_len]  # 训练集
    y_train = y[:train_data_len]  # 训练标签集

    X_test = X[train_data_len:]  # 测试集
    y_test = y[train_data_len:]  # 测试集标签集

    # 返回值
    return X_train, X_test, y_train, y_test

# 功能函数：基于新的X_train, X_test, y_train, y_test创建批数据(batch dataset)

def create_batch_data(X, y, batch_size=32, data_type=1):
    '''基于训练集和测试集，创建批数据
    Params:
        X : 特征数据集
        y : 标签数据集
        batch_size : batch的大小，即一个数据块里面有几个样本
        data_type : 数据集类型（测试集表示1，训练集表示2）

    Returns:
        train_batch_data 或 test_batch_data
    '''
    if data_type == 1:  # 测试集
        dataset = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))  # 封装X和y，成为tensor类型
        test_batch_data = dataset.batch(batch_size)  # 构造批数据
        # 返回
        return test_batch_data
    else:  # 训练集
        dataset = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))  # 封装X和y，成为tensor类型
        train_batch_data = dataset.cache().shuffle(1000).batch(batch_size)  # 构造批数据
        # 返回
        return train_batch_data

if __name__ == '__main__':
        """数据获取"""
        #获取数据集 订正数据格式
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
        plt.rcParams['axes.unicode_minus'] = False#解决负数问题
        x_Seq_len=16
        #修改csv文件路径
        dataset = pd.read_excel("D:/pythoncode/WindProject/风力发电功率数据集.xlsx")
        dataset['数据时间']=pd.to_datetime(dataset['数据时间'],format="%Y-%m-%d %H:%M:%S")
        dataset.index=dataset.数据时间#将其索引变为时间
        dataset.drop(columns='数据时间',axis=1,inplace=True)
        plt.figure()
        plt.plot(dataset)
        plt.show()
        """
        数据清洗
        """
        #缺失值处理
        #查看是否有缺失值
        print(dataset.info())#无缺失值
        # print(dataset[dataset.isnull()==False])#无
        # dataset['总有功功率（kw）']=dataset['总有功功率（kw）'].fillna(0) 对缺失值填值处理
        # dataset1=dataset[dataset['总有功功率（kw）'].notnull()] 剔除存在缺失值的数据，自己选择一直缺失值处理的方法
        #异常值处理
        """
        箱型图查看 可视化数据分布
        """
        f, ax = plt.subplots()
        sns.boxplot(y='总有功功率（kw）', data=dataset, ax=ax)
        plt.show()
        s = dataset.describe()
        # 基本统计量，存在异常值的将其筛选出来进行处理，可以用中位数填值或者众数填值，方法任选，这里没有异常值就没有处理
        q1 = s.loc['25%']
        q3 = s.loc['75%']
        iqr = q3 - q1#分位差
        mi = q1 - 1.5 * iqr#下限，低于这个为异常值
        ma = q3 + 1.5 * iqr#上限，高于这个为异常值
        #无异常值
        """
          数据归一化处理，均值为0，方差为1
        """
        scaler = MinMaxScaler()
        dataset['总有功功率（kw）'] = scaler.fit_transform(dataset['总有功功率（kw）'].values.reshape(-1, 1))
        #将归一化的数据保持
        with open('data.csv','w',encoding='utf-8',newline='')as f:
            w=csv.writer(f)
            w.writerow(dataset['总有功功率（kw）'])
        #归一化后的绘图
        dataset['总有功功率（kw）'].plot()
        plt.show()
        """
        特征提取（特征工程）
        """
        dataset_new = dataset
        # X为特征数据集，y为标签数据集
        X, y = create_new_dataset(dataset_new.values, seq_len=x_Seq_len)
        # X_train为数据训练集，X_test为数据测试集,y_train为标签训练集,y_test为标签测试集合
        X_train, X_test, y_train, y_test = split_dataset(X, y)
        # 基于新的X_train, X_test, y_train, y_test创建批数据(batch dataset)
        # 测试批数据
        test_batch_dataset = create_batch_data(X_test, y_test, batch_size=24, data_type=1)
        # 训练批数据
        train_batch_dataset = create_batch_data(X_train, y_train, batch_size=24, data_type=2)
        """
        构建模型
        """
        model = Sequential([
            layers.LSTM(8, input_shape=(x_Seq_len, 1)),
            layers.Dense(1)
        ])
        # 定义 checkpoint，保存权重文件
        file_path = "best_checkpoint.hdf5"#将数据加载到内存
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=file_path,
                                                                 monitor='loss',
                                                                 mode='min',
                                                                 save_best_only=True,
                                                                 save_weights_only=True)
        """
        编译运行，预测，指标检验
        """
        # 模型编译
        model.compile(optimizer='adam', loss="mae")
        # 模型训练（次数200）
        history = model.fit(train_batch_dataset,
                            epochs=100,
                            validation_data=test_batch_dataset,
                            callbacks=[checkpoint_callback])
        # 显示 train loss 和 val loss
        plt.figure()
        plt.plot(history.history['loss'], label='train loss')
        plt.plot(history.history['val_loss'], label='val loss')
        plt.title("LOSS")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(loc='best')
        plt.show()
        # 模型验证
        test_pred = model.predict(X_test, verbose=1)
        plt.figure()
        d1=plt.plot(y_test, label='True')
        d2=plt.plot(test_pred, label='pred')
        plt.legend([d1,d2],labels=['True','pred'])
        plt.show()
        # 计算r2
        score = r2_score(y_test, test_pred)
        print("r^2 的值： ", score)
        # 绘制test中前100个点的真值与预测值
        y_true = y_test # 真实值
        y_pred = test_pred  # 预测值

        fig, axes = plt.subplots(2, 1)
        ax0=axes[0].plot(y_true, marker='o', color='red',label='true')
        ax1=axes[1].plot(y_pred, marker='*', color='blue',label='pred')
        plt.show()

        """
        模型测试，评估
        """
        # 选择test中的最后一个样本
        sample = X_test[-1]
        sample = sample.reshape(1, sample.shape[0], 1)
        # 模型预测
        sample_pred = model.predict(sample)#predict()预测标签值
        ture_data = X_test[-1]  # 真实test的最后20个数据点
        # 预测后48个点
        preds=predict_next(model,ture_data,48)
        # 绘图
        plt.figure()
        plt.plot(preds, color='yellow', label='Prediction')
        plt.plot(ture_data, color='blue', label='Truth')
        plt.xlabel("Epochs")
        plt.ylabel("Value")
        plt.legend(loc='best')
        plt.show()
        relative_error = 0
        """模型精确度计算"""
        for i in range(len(y_pred)):
            relative_error += (abs(y_pred[i] - y_true[i]) / y_true[i]) ** 2
        acc = 1- np.sqrt(relative_error / len(y_pred))
        print(f'模型的测试准确率为：{acc}')
