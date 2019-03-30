#加载包
# 以下程序为预测离散化之后的sin函数
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

import pandas as pd  
import io  

# 加载matplotlib工具包，使用该工具包可以对预测的函数曲线进行绘图
import matplotlib as mpl
from tensorflow.contrib.learn.python.learn.estimators.estimator import SKCompat
mpl.use('Agg')
from matplotlib import pyplot as plt
learn = tf.contrib.learn

tf.logging.set_verbosity(3)

# 数据集名称，数据集要放在你的工作目录下
HIDDEN_SIZE = 30  # Lstm中隐藏节点的个数
NUM_LAYERS = 2  # LSTM的层数
TIMESTEPS = 17  # 循环神经网络的截断长度
TRAINING_STEPS = 250  # 训练轮数
BATCH_SIZE = 32  # batch大小

TRAINING_END = 1827  # 训练数据索引
TESTING_END = 2192  # 测试数据索引

PRE_TIME = 2

INPUT_COLNUM = 5
OUTPUT_COLNUM = 1

CSV_TRAIN  = "train_lijiadu.csv"
CSV_TEST = "train_lijiadu.csv"

MAX_X = 0
MAX_Y = 0
MIN_Y = 0
def generate_data(fileCSV, start, end):
    X = []
    y = []
    df_train = pd.read_csv(fileCSV)
    data = df_train.values
    data_x = data[0:TESTING_END, 0:INPUT_COLNUM]
    data_y = data[0:TESTING_END, INPUT_COLNUM]

    MAX_X = np.max(data_x)
    MAX_Y = np.max(data_y)
    MIN_Y = np.min(data_y)

    df = (data_x - 0)/(MAX_X - 0)
    df_y = (data_y - MIN_Y)/(MAX_Y - MIN_Y)
    
    for i in range(start, end - TIMESTEPS - 1):
        t = np.hstack( df[i:i+TIMESTEPS, 0:INPUT_COLNUM])   # t.shape = (400,) = （TIMESTEPS * INPUT_COLNUM，）
        #t =np.append(t, df_y[i :i + TIMESTEPS-PRE_TIME]) 
        t_y = df_y[i+TIMESTEPS]
        X.append( [t.tolist()])
        y.append( [t_y])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), MAX_X, MAX_Y,  MIN_Y

def LstmCell():
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE,state_is_tuple=True)
    return lstm_cell

# 定义lstm模型
def lstm_model(X, y):
    
    cell = tf.nn.rnn_cell.MultiRNNCell([LstmCell() for _ in range(NUM_LAYERS)])
    output, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    output = tf.reshape(output, [-1, HIDDEN_SIZE])
    # 通过无激活函数的全连接层计算线性回归，并将数据压缩成一维数组结构
    predictions = tf.contrib.layers.fully_connected(output, 1, None)
    
    # 将predictions和labels调整统一的shape
    labels = tf.reshape(y, [-1])
    predictions = tf.reshape(predictions, [-1])
    
    #loss = tf.losses.mean_squared_error(predictions, labels)
    loss = tf.losses.absolute_difference(predictions, labels)
   

    train_op = tf.contrib.layers.optimize_loss(loss, tf.contrib.framework.get_global_step(),
                                             optimizer="Adam",
                                             learning_rate=0.001)
    return predictions, loss, train_op

# 进行训练
# 封装之前定义的lstm
regressor = SKCompat(learn.Estimator( model_fn=lstm_model, model_dir="Models/model_ljdpc"))
# 生成数据
# train_X.shape = (1816, 1, 400)， TRAINING_EXAMPLES - TIMESTEPS -1
# train_y.shape = (1816, 1)
# test_X.shape = (354, 1, 400), 
# test_y.shape = (354, 1)
train_X, train_y, MAX_X, MAX_Y ,MIN_Y= generate_data(CSV_TRAIN, 0, TRAINING_END)
test_X, test_y, MAX_X, MAX_Y ,MIN_Y= generate_data(CSV_TEST, TRAINING_END-TIMESTEPS, TESTING_END)

# 拟合数据
regressor.fit( train_X, train_y, batch_size=BATCH_SIZE, steps=TRAINING_STEPS)
# 计算预测值
predicted = [[pred] for pred in regressor.predict(test_X)]

# 计算MSE
rmse = np.sqrt(((predicted - test_y) ** 2).mean(axis=0))

print("Mean Square Error is:%f" % rmse[0])


#print(predicted) [[],[],[]]
#print(test_y)  
# output xls
simi = np.squeeze ( np.array(predicted) * (MAX_Y-MIN_Y) + MIN_Y)
obsi = np.squeeze (test_y * (MAX_Y-MIN_Y) + MIN_Y )

#计算 Ens 计算纳希效率系数
Ens = 1 - np.sum((obsi - simi)** 2)/ np.sum((obsi - np.mean(obsi))**2)
print("纳希效率系数 Ens: %f" % Ens)

#计算 R2 计算确定性系数
R2_up = np.sum((obsi - np.mean(obsi))*(simi - np.mean(simi)))**2 
R2_bt = np.sum(( obsi - np.mean(obsi))**2) * np.sum((simi - np.mean(simi))**2)
R2 = R2_up / R2_bt
print("确定性系数 R2: %f" % R2)

#计算 Re 相对误差
Re = np.sum( simi - obsi)/np.sum(obsi) * 100
print("相对误差 Re:百分之 %f" % Re)

simi_column = pd.Series(simi, name='simi')
obsi_column = pd.Series(obsi, name='obsi')
predictions = pd.concat([simi_column, obsi_column], axis=1)
writer = pd.ExcelWriter('lijiadu.xls')
predictions.to_excel(writer,'sheet_name')
writer.save()

fig = plt.figure()
plot_predicted, = plt.plot(simi, label='predicted')
plot_test, = plt.plot(obsi, label='real_flow')
plt.legend([plot_predicted, plot_test],['predicted_hs:%d_t:%d'%(HIDDEN_SIZE,TIMESTEPS), 'real_flow'])
fig.savefig('flow_lijiadupc.png')
plt.show()




