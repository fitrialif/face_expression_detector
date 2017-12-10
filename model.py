import tensorflow as tf
import csv
import numpy as np
import os
import random
class Model():
    def __init__(self):
        self.X = tf.placeholder(tf.float32,[None,2304])
        self.Y = tf.placeholder(tf.int32,[None])  # 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).
        self.checkpoint_save_dir = os.path.join("checkpoint")
        self.DATA_PATH = os.path.join("data_set","fer2013.csv")

        self.keep_prob= tf.placeholder(tf.float32)
        
        
        self.X_img = tf.reshape(self.X,[-1,48,48,1])
        self.Y_one_hot = tf.one_hot(self.Y,7)

        self.weight_1 = tf.Variable(tf.random_normal([3,3,1,32],stddev=0.01))
        self.bias_1 = tf.Variable(tf.random_normal([32],stddev=0.01))

        self.L1 = tf.nn.conv2d(self.X_img,self.weight_1,strides=[1,1,1,1],padding='SAME')
        self.L1 = tf.nn.relu(self.L1)
        self.L1 = tf.nn.max_pool(self.L1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')  # now size becomes ? 24 24 32
        self.L1 = tf.nn.bias_add(self.L1, self.bias_1)

        self.L1 = tf.nn.dropout(self.L1,keep_prob=self.keep_prob)



        self.weight_2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
        self.bias_2 = tf.Variable(tf.random_normal([64], stddev=0.01))

        self.L2 = tf.nn.conv2d(self.L1, self.weight_2, strides=[1, 1, 1, 1], padding='SAME')
        self.L2 = tf.nn.relu(self.L2)
        self.L2 = tf.nn.max_pool(self.L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')  # now size becomes ? 12 12 64
        self.L2 = tf.nn.bias_add(self.L2, self.bias_2)

        self.L2 = tf.nn.dropout(self.L2, keep_prob=self.keep_prob)

        self.weight_3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
        self.bias_3 = tf.Variable(tf.random_normal([128], stddev=0.01))

        self.L3 = tf.nn.conv2d(self.L2, self.weight_3, strides=[1, 1, 1, 1], padding='SAME')
        self.L3 = tf.nn.relu(self.L3)
        self.L3 = tf.nn.max_pool(self.L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')  # now size becomes ? 6 6 128
        self.L3 = tf.nn.bias_add(self.L3, self.bias_3)

        self.L3 = tf.nn.dropout(self.L3, keep_prob=self.keep_prob)
        self.L3_flatten = tf.reshape(self.L3,[-1,6*6*128])

        # FC layer
        self.weight_4 = tf.get_variable(name="w4",shape=[6*6*128,7],initializer=tf.contrib.layers.xavier_initializer())
        self.bias_4 = tf.Variable(tf.random_normal([7]))

        self.logits = tf.matmul(self.L3_flatten,self.weight_4)+self.bias_4       #size now become ?,7

        self.cost = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y_one_hot,logits=self.logits)
        self.train = tf.train.AdamOptimizer().minimize(self.cost)

        self.prediction = tf.argmax(self.logits,1)

        self.prediction_result = tf.equal(tf.argmax(self.logits,1),tf.argmax(self.Y_one_hot,1))

        self.mean_cost = tf.reduce_mean(self.cost,0)
        self.mean_accuracy= tf.reduce_mean(tf.cast(self.prediction_result,tf.float32))


        self.file_path  = os.path.join("data_set","fer2013.csv")    # data from : https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
    def predict_result(self,checkpoint_save_dir,feed_dict):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            latest_chkpt = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_save_dir)  # old to new
            saver.restore(sess, latest_chkpt)

            return sess.run(self.prediction,feed_dict=feed_dict)
    def start_train(self,checkpoint_save_dir,epoch,batch_size,data_path,eval_freq):
        if not os.path.exists(checkpoint_save_dir):
            print("check point dir not found, making one")
            os.mkdir(checkpoint_save_dir)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver(max_to_keep=1000)
            latest_chkpt = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_save_dir)  # old to new
            if latest_chkpt==None:
                print("no checkpoint was found")
            else:
                print("checkpoint found at: %s"%latest_chkpt)
                saver.restore(sess,latest_chkpt)

            step=0
            for i in range(epoch):
                print("%s epoch"%i)
                for x_data,y_data,purpose in self.get_input(data_path,batch_size):
                    if(step % eval_freq==0):
                        print("-------------------------------------------------------------------------------------")
                        cost,accuracy = sess.run([self.mean_cost,self.mean_accuracy],feed_dict={self.X: x_data,self.Y:y_data,self.keep_prob:1})
                        print("COST: ",cost)
                        print("Accuracy: ",accuracy)
                        saver.save(sess,os.path.join(checkpoint_save_dir,"%s epoch %s step"%(i,step)))
                        print("progress saved at %s "%checkpoint_save_dir+"%s epoch %s step"%(i,step))
                        print("-------------------------------------------------------------------------------------")
                    else:
                        sess.run(self.train, feed_dict={self.X: x_data, self.Y: y_data, self.keep_prob: 0.7})
                    step += 1
    def get_input(self,file_path,batch_size,shuffle=True):
        with open(file_path) as csvfile:
            csvfile = np.array(csvfile.readlines()[1:-31])
            if shuffle:
                np.random.shuffle(csvfile)
            for i in range(len(csvfile)-batch_size):
                y = []
                x = []
                purpose = []
                txt_list = csvfile[i:i+batch_size]
                for txt in txt_list:
                    txt=txt.split(",")
                    x.append([np.uint8(x_data) for x_data in txt[1].split()])
                    y.append(txt[0])
                    purpose.append([txt[2]])
                # if shuffle:
                #     for i in range(len(x)):
                #         change_to = random.randrange(0,len(x))
                #
                #         tmp_1 = x[i]
                #         x[i] = x[change_to]
                #         x[change_to] = tmp_1
                #
                #         tmp_2 = y[i]
                #         y[i] = y[change_to]
                #         y[change_to] = tmp_2


                yield x,y,purpose
if __name__ == "__main__":
    a = Model()
    a.start_train(a.checkpoint_save_dir,15,1000,a.DATA_PATH,10)

