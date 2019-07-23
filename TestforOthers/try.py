# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras
import numpy as np
import csv
import matplotlib as plt
import pandas as pd  
from sklearn.utils import shuffle

df=pd.read_csv("test.csv")

print(df)
df=df.values
df=np.array(df)

x_data=df[:,:75]
y_data=df[:,75]

#define module
x=tf.placeholder(tf.float32,[1,75],name="X")
y=tf.placeholder(tf.float32,[1,1],name="Y")

with tf.name_scope("Model1"):
    w=tf.Variable(tf.random_normal([75,1],stddev=0.01,name="W"))
    b=tf.Variable(1.0,name="b")
    def model(x,w,b):
        return tf.matmul(x,w)+b
    pred=model(x,w,b)
    
#train model
train_epochs=50
learning_rate=0.01

with tf.name_scope("LossFunction"):
    loss_function=tf.reduce_mean(tf.pow(y-pred,2))
    
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)

sess=tf.Session()
init=tf.global_variables_initializer()

logdir='d:/log'
sum_loss_op=tf.summary.scalar("loss",loss_function)
merged=tf.summary.merge_all()

sess.run(init)
writer=tf.summary.FileWriter(logdir,sess.graph)

#epoch
loss_list=[]
for epoch in range(train_epochs):
    loss_sum=0.0
    for xs,ys in zip (x_data,y_data):
        
        xs=xs.reshape(1,75)
        ys=ys.reshape(1,1)
        
        _,summary_str,loss = sess.run([optimizer,sum_loss_op,loss_function],feed_dict={x:xs,y:ys})
        writer.add_summary(summary_str,epoch)
        
        loss_sum = loss_sum + loss
        loss_list.append(loss)
        
        x_data,y_data = shuffle(x_data,y_data)
        
    b0temp = b.eval(session=sess)            #训练中当前变量b值
    w0temp = w.eval(session=sess)            #训练中当前权重w值
    loss_average = loss_sum/len(y_data)      #当前训练中的平均损失
    loss_list.append(loss_average)
    
print("epoch=",epoch+1,"loss=",loss_average,"b=",b0temp,"w=",w0temp)


n = np.random.randint(40)
print(n)
x_test = x_data[n]

x_test = x_test.reshape(1,75)
predict = sess.run(pred,feed_dict={x:x_test})
print("预测值：%f"%predict)

target = y_data[n]
print("标签值：%f"%target)