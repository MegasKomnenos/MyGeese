import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        
import tensorflow as tf
import numpy as np
import json
import base64

NUM_ACT = 4

class Block(tf.keras.layers.Layer):
    def __init__(self, flt, **kwargs):
        super(Block, self).__init__(**kwargs)
        
        self.dense = tf.keras.layers.Dense(flt, use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization()
        
    def call(self, inp, training=False):
        x = self.dense(inp)
        x = self.bn(x, training)
        x = tf.nn.relu(x)
        
        return x

class Net(tf.keras.Model):
    def __init__(self, layers, out, stock):
        super(Net, self).__init__()

        self.stats = { 'layers': layers, 'out': out, 'stock': stock }
        self.stock = stock

        self.tower = []

        for l in layers:
            self.tower.append(Block(l))
        
        self.flt = tf.keras.layers.Flatten()
        self.out = tf.keras.layers.Dense(out)
    
    def call(self, inp, training=False):
        x = inp

        if len(x.shape) < 4:
            x = tf.expand_dims(x, 0)
            
        x = self.flt(x)

        for block in self.tower:
            x = block(x, training=training)
            
        x = self.out(x)
            
        return x
        
gen = int(input())

x = tf.zeros((7, 11, 6))

net = Net([512, 128, 128, 256], NUM_ACT, x)
net(net.stock)

if gen >= 0:
    net.load_weights(f'ddrive/{gen}c.h5')

with open('ww.txt', 'w') as f:
    f.write(json.dumps([base64.b64encode(arr.astype(np.half)).decode('ascii') for arr in net.get_weights()]))
with open('ws.txt', 'w') as f:
    f.write(json.dumps([arr.shape for arr in net.get_weights()]))
