import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
#from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, Dense, Activation
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
test_num=500
train_times=400
#random walk model to generate the probability tendency of task
def random_walk_model():
    fig=plt.figure(0)
    #time span
    T=500
    #drift factor飘移率
    mu=0.00005
    #volatility波动率
    sigma=0.04
    #t=0初试价
    S0=np.random.random()
    #length of steps
    dt=1
    N=round(T/dt)
    #generate 500 steps and collect it into t
    t=np.linspace(0,T,N)

    #W is standard normal list
    W=np.random.standard_normal(size=N)
    W=(np.random.poisson(lam=5,size=N)-5)/15
    # W=np.random.uniform(low=-0.5,high = 0.5,size=N)
    print("W ",W)
    #W.shape=(500,)
    #几何布朗运动过程,产生概率轨迹
    W=np.cumsum(W)*np.sqrt(dt)
    S=sigma*W #+ (mu-0.5*sigma**2)*t
    # S=S0*np.exp(X)
    # S = np.random.poisson(lam=10,size= N)
    # S = np.random.uniform(low=0,high=1,size=N)
    # plt.show()
    #save the probability tendency of picture
    fd=pd.DataFrame({'pro':S})
    fd.to_csv('pic/random_walk_model.csv',sep=',',index=False)
    plt.savefig('pic/random_data.png')
    return S
def random_test(sequence_length=5,split=0.7):

    #get the stored data by using pandas
    test_data = pd.read_csv('./pic/random_walk_model.csv', sep=',',usecols=[0])
    #print("test_data:",test_data)

    #generate new test data for 2d
    test_data = np.array(test_data).astype('float64')
    #print('test_data:',test_data.shape)
    #test_data: (500, 1)

    #70% are used to be trained, the rest is used to be tested
    split_boundary = int(test_data.shape[0] * split)
    #print('split_boundary:',split_boundary)
    #split_boundary:350

    pro_test=np.linspace(split_boundary,test_data.shape[0],test_data.shape[0]-split_boundary)
    pro_x=np.linspace(1,split_boundary,split_boundary)
    # plt.plot(pro_x,test_data[:split_boundary])
    # plt.plot(pro_test,test_data[split_boundary:],'red')
    # plt.legend(['train_data','test_data'])
    # plt.xlabel('times')
    # plt.ylabel('probability')
    # plt.show()
    #print("test_data: ",test_data,test_data.shape),test_data.shape=(600,1),array to list format
 
    #generate 3d format of data and collect it
    data = []
 
    for i in range(len(test_data) - sequence_length - 1):
        data.append(test_data[i: i + sequence_length + 1])
    #print(len(data[0][0]),len(data[1]),len(data))
    #1 6 494
    reshaped_data = np.array(data).astype('float64')
    #print("reshaped_data:",reshaped_data.shape)
   #reshaped_data: (494, 6, 1)
 
    #random the order of test_data to improve the robustness
    np.random.shuffle(reshaped_data)
    #from n to n*5 are the training data collected in x, the n*6th is the true value collected in y
    x = reshaped_data[:, :-1]
    y = reshaped_data[:, -1]
 
    #print("x ",x.shape,"\ny ",y.shape)
    #x  (494, 5, 1) y  (494, 1)
 
    #train data
    train_x = x[: split_boundary]
    train_y = y[: split_boundary]
    #test data
    test_x = x[split_boundary:]
    test_y=y[split_boundary:]
    #print("train_y:",train_x.shape,"train_y:",train_y.shape,"test_x ",test_x.shape,"test_y",test_y.shape)
    #train_y: (350, 5, 1) train_y: (350, 1) test_x  (144, 5, 1) test_y (144, 1)
    return train_x, train_y, test_x, test_y
def read_test(sequence_length=160,split=0.7):

    #get the stored data by using pandas
    test_data = pd.read_csv('../Poisson.csv', sep=',',usecols=[1,2,3,4,5,6,7,8,9,10])
    # print("test_data:",test_data)

    #generate new test data for 2d
    test_data = np.array(test_data).astype('float64')
    #print('test_data:',test_data.shape)
    #test_data: (500, 1)

    #70% are used to be trained, the rest is used to be tested
    split_boundary = int(test_data.shape[0] * split)
    #print('split_boundary:',split_boundary)
    #split_boundary:350

    pro_test=np.linspace(split_boundary,test_data.shape[0],test_data.shape[0]-split_boundary)
    pro_x=np.linspace(1,split_boundary,split_boundary)
    # plt.plot(pro_x,test_data[:split_boundary])
    # plt.plot(pro_test,test_data[split_boundary:],'red')
    # plt.legend(['train_data','test_data'])
    # plt.xlabel('times')
    # plt.ylabel('probability')
    # plt.show()
    #print("test_data: ",test_data,test_data.shape),test_data.shape=(600,1),array to list format
 
    #generate 3d format of data and collect it
    data = []
 
    for i in range(len(test_data) - sequence_length - 1):
        data.append(test_data[i: i + sequence_length])
    #print(len(data[0][0]),len(data[1]),len(data))
    #1 6 494
    reshaped_data = np.array(data).astype('float64')
    print("reshaped_data:",reshaped_data.shape)
   #reshaped_data: (494, 6, 1)
 
    #random the order of test_data to improve the robustness
    np.random.shuffle(reshaped_data)
    #from n to n*5 are the training data collected in x, the n*6th is the true value collected in y
    x = reshaped_data[:, :int(sequence_length/2),:]
    y = reshaped_data[:,int(sequence_length/2):, :]
 
    print("x ",x.shape,"\ny ",y.shape)
    #x  (494, 5, 1) y  (494, 1)
 
    #train data
    train_x = x[: split_boundary]
    train_y = y[: split_boundary]
    #test data
    test_x = x[split_boundary:]
    test_y=y[split_boundary:]
    print("train_y:",train_x.shape,"train_y:",train_y.shape,"test_x ",test_x.shape,"test_y",test_y.shape)
    #train_y: (350, 5, 1) train_y: (350, 1) test_x  (144, 5, 1) test_y (144, 1)
    return train_x, train_y, test_x, test_y

def build_model():
    # input_dim是输入的train_x的最后一个维度,相当于输入的神经只有1个——特征只有1个，train_x的维度为(n_samples, time_steps, input_dim)
    #如果return_sequences=True：返回形如（samples，timesteps，output_dim）的3D张量否则，返回形如（samples，output_dim）的2D张量
    #unit并不是输出的维度，而是门结构（forget门、update门、output门）使用的隐藏单元个数
    model = Sequential()
    #use rmsprop for optimizer
    rmsprop=keras.optimizers.RMSprop(lr=0.001, rho=0.9,epsilon=1e-08,decay=0.0)
    #build one LSTM layer

    model.add(LSTM(input_shape=( 80, 10), units=10, return_sequences=True,use_bias=True,activation='relu'))
    #model.add(LSTM(100, return_sequences=False,use_bias=True,activation='tanh'))
 
    #comiple this model
    model.compile(loss='mse', optimizer=rmsprop)#rmsprop
    return model
 
def train_model(train_x, train_y, test_x, test_y):
    #call function to build model
    model = build_model()
 
    try:
        #store this model to use its loss parameter
        history=model.fit(train_x, train_y, batch_size=200, epochs=train_times,verbose=2)
        #store the loss
        lossof_history=history.history['loss']
 
        predict = model.predict(test_x)
        predict = np.reshape(predict, (predict.size, ))
        print(predict)
        print(test_y)
        #evaluate this model by returning a loss
        loss=model.evaluate(test_x,test_y)
        print("loss is ",loss)
        model.save("../lstm.pth")
    #if there is a KeyboardInterrupt error, do the following
    except KeyboardInterrupt:
        print("error of predict ",predict)
        print("error of test_y: ",test_y)
 
    try:
        #x1 is the xlabel to print the test value, there are 500 data,30% is for testing
        x1=np.linspace(1,test_y.shape[0],test_y.shape[0])
        print(x1)
        #x1 is the xlabel to print the loss value, there are 500 data,70% is for training
        x2=np.linspace(1,train_times,train_times)
        fig = plt.figure(1)
        #print the predicted value and true value
        np.save('./real_prediction.npy', predict)
        np.save('./test.npy', test_y)
        plt.title("test with rmsprop lr=0.01_")
        plt.plot(x1,predict)
        plt.plot(x1,test_y)
        plt.legend(['predict', 'true'])
        plt.xlabel('times')
        plt.ylabel('propability')
        plt.savefig('pic/train_with_rmsprop_lr=0.01.png')
        #print the loss
        fig2=plt.figure(2)
        plt.title("loss lr=0.01")
        plt.plot(x2,lossof_history)
        plt.savefig('pic/train_with_rmsprop_lr=0.01_LOSS_.png')
        plt.show()
    #if the len(x1) is not equal to predict.shape[0] / test_y.shape[0] / len(x2) is not equal to lossof_history.shape[0],there will be an Exception
    except Exception as e:
        print("error: ",e)

if __name__ == '__main__':
    #random_walk_model() function is only used by once, because data are stored as pic/random_data.csv
    # random_walk_model()
 
    #prepare the right data format for LSTM
    train_x, train_y, test_x, test_y=read_test()
    #standard the format for LSTM input
    # test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))
    #print("main: train_x.shape ",train_x.shape)
    #main: train_x.shape  (350, 5, 1)
    train_model(train_x, train_y, test_x, test_y)