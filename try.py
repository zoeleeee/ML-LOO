import tensorflow as tf 
import numpy as np 

class Model(object):
    def __init__(self):
        self.x_input = tf.placeholder(tf.float32, shape=[None])
        self.y_input = tf.placeholder(tf.int64, shape=None)
        self.pre_softmax = self.forward(self.x_input)
        self.logits = self.pre_softmax
    def forward(self,x_input):
        return self.build_model(x_input)
    def build_model(self, x_input):
        inter_xs = []
        x = tf.nn.xw_plus_b(x_input, [1], [3])
        inter_xs.append(x)
        x = tf.nn.xw_plus_b(x, [2], [4])
        inter_xs.append(x)
        x = tf.nn.xw_plus_b(x, [5], [2])
        inter_xs.append(x)
        return x, inter_xs

def main():
    model = Model()
    with tf.Session() as sess:
        feed_dict = {model.x_input:[1,2,3]}
        y_pred, activations = sess.run(model.logits, feed_dict=feed_dict)
        for i in [1,2,3]:
            print(i, y_pred[i-1], activations[i-1])

if __name__ == '__main__':
    main()