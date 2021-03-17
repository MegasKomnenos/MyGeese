import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        
import tensorflow as tf
import numpy as np
import json
import base64

NUM_ACT = 4

class Residual(tf.keras.layers.Layer):
    def __init__(self, flt, **kwargs):
        super(Residual, self).__init__(**kwargs)
        
        self.conv_0 = tf.keras.layers.Conv2D(flt, 3, padding='same', use_bias=False)
        self.conv_1 = tf.keras.layers.Conv2D(flt, 3, padding='same', use_bias=False)
        self.conv_2 = tf.keras.layers.Conv2D(flt, 1, padding='same')
        self.bn_0 = tf.keras.layers.BatchNormalization()
        self.bn_1 = tf.keras.layers.BatchNormalization()
        
    def call(self, inp, training=False):
        if len(inp.shape) < 4:
            inp = tf.expand_dims(inp, 0)
            
        x = self.conv_0(inp)
        x = self.bn_0(x, training)
        x = tf.nn.relu(x)
        x = self.conv_1(x)
        x = self.bn_1(x, training)
        x = tf.nn.relu(x)
        x += self.conv_2(inp)
        
        return tf.nn.relu(x)

class Net(tf.keras.Model):
    def __init__(self, layers, out, stock):
        super(Net, self).__init__()

        self.stats = { 'layers': layers, 'out': out, 'stock': stock }
        self.stock = stock

        self.tower = []

        for l in layers:
            if l == -1:
                self.tower.append(tf.keras.layers.AveragePooling2D())
            else:
                self.tower.append(Residual(l))
        
        self.flt = tf.keras.layers.Flatten()
        self.out = tf.keras.layers.Dense(out)
        self.act = tf.keras.layers.PReLU()
    
    def call(self, inp, training=False):
        x = inp

        if len(x.shape) < 4:
            x = tf.expand_dims(x, 0)

        for block in self.tower:
            x = block(x, training=training)

        x = self.flt(x)
        x = self.out(x)
        x = self.act(x)
            
        return x
        
gen = input()

x = tf.zeros((7, 11, 4))

net = Net([32, 32, 48, 48, 48, 48, 64, 64], NUM_ACT, x)
net(net.stock)
net.load_weights(f'ddrive/{gen}c.h5')

with open('ww.txt', 'w') as f:
    f.write(json.dumps([base64.b64encode(arr.astype(np.half)).decode('ascii') for arr in net.get_weights()]))
with open('ws.txt', 'w') as f:
    f.write(json.dumps([arr.shape for arr in net.get_weights()]))