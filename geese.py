import os
 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

import tensorflow as tf
import numpy as np
import random
import itertools
import tqdm
import orjson
import base64
import socket
from ray.util.multiprocessing import Pool
from kaggle_environments.envs.hungry_geese.hungry_geese import Action
from kaggle_environments import make

tcritic = None
tgen = None

NUM_GRID = (7, 11)
NUM_CHANNEL = 6
NUM_ACT = 4
NUM_GEESE = 4

GEN_ENDED_AT = int(input())
GEN_ENDS_AT = int(input())
GAME_PER_GEN = 400

NUM_LAMBDA = 0.8
NUM_P = 0.95

STOCK_X = tf.convert_to_tensor(np.zeros((*NUM_GRID, NUM_CHANNEL)), dtype='float32')
STOCK_ACT = [Action(i + 1) for i in range(NUM_ACT)]

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
        self.act = tf.keras.layers.PReLU()
    
    def call(self, inp, training=False):
        x = inp

        if len(x.shape) < 4:
            x = tf.expand_dims(x, 0)
            
        x = self.flt(x)

        for block in self.tower:
            x = block(x, training=training)
            
        x = self.out(x)
        x = self.act(x)
            
        return x

    @classmethod
    def clone(cls, self):
        out = cls(**self.stats)
        out(self.stock)

        for i in range(len(self.tower)):
            out.tower[i].set_weights(self.tower[i].get_weights())
            
        out.out.set_weights(self.out.get_weights())
        out.act.set_weights(self.act.get_weights())

        return out

class Critic(Net):
    def train_step(self, dat):
        x, y, act = dat

        with tf.GradientTape() as tape:
            y_pred = tf.gather(self(x, training=True), act, batch_dims=1)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        
        t = self.trainable_variables
        g = tape.gradient(loss, t)

        self.optimizer.apply_gradients(zip(g, t))
        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}

class Cell:
    def __init__(self, s, a, r):
        self.s = s
        self.ss = None
        self.a = a
        self.r = r
    
    def ser(self):
        out = dict()

        out['s'] = base64.b64encode(self.s.numpy()).decode('ascii')
        out['a'] = self.a
        out['r'] = self.r

        if self.ss != None:
            out['ss'] = base64.b64encode(self.ss.numpy()).decode('ascii')

        return out

    def deser(dct):
        dct['s'] = tf.convert_to_tensor(np.frombuffer(base64.decodebytes(dct['s'].encode('utf-8')), dtype=np.float32).reshape((*NUM_GRID, NUM_CHANNEL)))
        out = Cell(dct['s'], dct['a'], dct['r'])

        if 'ss' in dct:
            out.ss = tf.convert_to_tensor(np.frombuffer(base64.decodebytes(dct['ss'].encode('utf-8')), dtype=np.float32).reshape((*NUM_GRID, NUM_CHANNEL)))
        
        return out

class CellGroup:
    def __init__(self, cs, target):
        self.xs = [c.s for c in cs]
        self.ys = [tf.constant([[c.r]]) for c in cs]
        self.acts = [tf.constant([[c.a]]) for c in cs]

        indexs = list()
        ss = list()

        for i, c in enumerate(cs):
            if c.ss != None:
                indexs.append(i)
                ss.append(c.ss)

        for b in range(len(indexs) // 8192 + 1):
            sss = tf.stack(ss[b * 8192:(b + 1) * 8192])
            y = target(sss, training=True)
            y = tf.nn.relu(tf.reduce_max(y, axis=1))
            y *= NUM_LAMBDA

            for i, ii in enumerate(indexs[b * 8192:(b + 1) * 8192]):
                self.ys[ii] += y[i]

class Goose:
    def __init__(self, critic):
        self.critic = critic

        self.cs = list()
        self.heads = None

    def __call__(self, obs, _):
        acts = [None] * len(obs['geese'])
        
        if self.heads != None:
            for i, head in enumerate(self.heads):
                if head and obs['geese'][i]:
                    r, c = pos_to_coord(head)
                    rr, cc = pos_to_coord(obs['geese'][i][0])
                    
                    if (r - 1) % NUM_GRID[0] == rr:
                        acts[i] = STOCK_ACT[0]
                    elif (c + 1) % NUM_GRID[1] == cc:
                        acts[i] = STOCK_ACT[1]
                    elif (r + 1) % NUM_GRID[0] == rr:
                        acts[i] = STOCK_ACT[2]
                    elif (c - 1) % NUM_GRID[1] == cc:
                        acts[i] = STOCK_ACT[3]
        
        self.heads = list()
        
        for goose in obs['geese']:
            if goose:
                self.heads.append(goose[0])
            else:
                self.heads.append(None)
            
        x = obs_to_x(obs, acts)

        scores = [float(score) + 0.1 for score in tf.nn.relu(self.critic(x))[0]]

        sm = 1.5 * sum(scores)

        scores = [score / sm for score in scores]
        scores[scores.index(max(scores))] += 1 / 3

        p = random.choices(population=scores, weights=scores)[0]
        i = scores.index(p)

        self.cs.append(Cell(x, i, 0.0))

        return STOCK_ACT[i].name

def run_game(gen):
    global tcritic, tgen
    
    if tcritic == None or tgen != gen:
        tcritic = Critic([512, 128, 128, 32], NUM_ACT, STOCK_X)
        tcritic(tcritic.stock)

        if gen >= 0:
            tcritic.load_weights(f'ddrive/{gen}c.h5')
            
        tgen = gen
        
    geese = [Goose(tcritic) for _ in range(NUM_GEESE)]
    env = make('hungry_geese')
    steps = env.run(geese)

    for i, step in enumerate(steps):
        if i == 0:
            continue

        obs = step[0]['observation']

        for ii, goose in enumerate(obs['geese']):
            if goose:
                geese[ii].cs[i - 1].r += 1

                if goose[0] in steps[i - 1][0]['observation']['food']:
                    geese[ii].cs[i - 1].r += 0.5
    
    mx = 0

    for goose in steps[-1][0]['observation']['geese']:
        if goose and len(goose) > mx:
            mx = len(goose)
    
    for i, goose in enumerate(steps[-1][0]['observation']['geese']):
        if goose and len(goose) == mx:
            geese[i].cs[-1].r += 9

    dat = list()

    for goose in geese:
        for i, c in enumerate(goose.cs[:-1]):
            c.ss = goose.cs[i + 1].s
        
        dat.extend(goose.cs)
    
    return dat

def pos_to_coord(pos):
    return pos // NUM_GRID[1], pos % NUM_GRID[1]

def act_to_id(act):
    if act.name == 'NORTH':
        return 0
    elif act.name == 'EAST':
        return 1
    elif act.name == 'SOUTH':
        return 2
    elif act.name == 'WEST':
        return 3

def obs_to_x(obs, acts):
    x = [[[0 for _ in range(NUM_CHANNEL)] for _ in range(NUM_GRID[1])] for _ in range(NUM_GRID[0])]
    index = obs['index']
    foods = obs['food']
    geese = obs['geese']

    rc, cc = pos_to_coord(geese[index][0])
    rc = round((NUM_GRID[0] - 1) / 2) - rc
    cc = round((NUM_GRID[1] - 1) / 2) - cc
    
    for food in foods:
        r, c = pos_to_coord(food)
        
        r = (r + rc) % NUM_GRID[0]
        c = (c + cc) % NUM_GRID[1]

        x[r][c][5] = 1

    for i, goose in enumerate(geese):
        if goose:
            r, c = pos_to_coord(goose[0])
            
            r = (r + rc) % NUM_GRID[0]
            c = (c + cc) % NUM_GRID[1]
            
            x[r][c][4] = 1
            
            if acts[i] != None:
                x[r][c][act_to_id(acts[i])] = -1
            
            for ii, block in enumerate(goose):
                if ii == 0:
                    continue
                
                r, c = pos_to_coord(block)
                rt, ct = pos_to_coord(goose[ii - 1])
                
                d = None
                
                if (r - 1) % NUM_GRID[0] == rt:
                    d = 0
                elif (c + 1) % NUM_GRID[1] == ct:
                    d = 1
                elif (r + 1) % NUM_GRID[0] == rt:
                    d = 2
                elif (c - 1) % NUM_GRID[1] == ct:
                    d = 3
                
                r = (r + rc) % NUM_GRID[0]
                c = (c + cc) % NUM_GRID[1]

                x[r][c][4] = 1
                x[r][c][d] = 1
    
    return tf.convert_to_tensor(x, dtype='float32')

class ProgCallback(tf.keras.callbacks.Callback):
    def __init__(self, pbar, **kwargs):
        super(ProgCallback, self).__init__(**kwargs)
        self.pbar = pbar
 
    def on_epoch_end(self, _, logs=None):
        self.pbar.update()

class MyDataset:
    def __init__(self, xs, ys, acts, shapes, batch_size=4096):
        self.xs = xs
        self.ys = ys
        self.acts = acts
        self.shapes = shapes
        self.batch_size = batch_size

    def _generator(self):
        indexs = np.arange(len(self.ys))
        np.random.shuffle(indexs)
    
        for x in range(len(self.ys) // self.batch_size + 1):
            index = indexs[x * self.batch_size:(x + 1) * self.batch_size]
            
            X = np.empty((len(index), *self.shapes[0]))
            y = np.empty((len(index), *self.shapes[1]))
            a = np.empty((len(index), *self.shapes[2]))
    
            for i, ii in enumerate(index):
                X[i,] = self.xs[ii]
                y[i,] = self.ys[ii]
                a[i,] = self.acts[ii]
    
            yield X, y, a

    def new(self):
        return tf.data.Dataset.from_generator(
            self._generator,
            output_types=('float32', 'float32', 'int32'), 
            output_shapes=((None, *self.shapes[0]), (None, *self.shapes[1]), (None, *self.shapes[2]))
        )

if __name__ == '__main__':
    socket.setdefaulttimeout(900)
    
    pool = Pool()
    
    critic = Critic([512, 128, 128, 32], NUM_ACT, STOCK_X)
    critic(critic.stock)
    
    if GEN_ENDED_AT >= 0:
        critic.load_weights(f'ddrive/{GEN_ENDED_AT}c.h5')
    
    cg = list()

    if GEN_ENDED_AT >= 0:
        with open(f'ddrive/{GEN_ENDED_AT}c.json') as f:
            ts = orjson.loads(f.read())
            ts = pool.map(Cell.deser, ts)
            cg.append(ts)
    if GEN_ENDED_AT >= 1:
        with open(f'ddrive/{GEN_ENDED_AT - 1}c.json') as f:
            ts = orjson.loads(f.read())
            ts = pool.map(Cell.deser, ts)
            cg.append(ts)
    if GEN_ENDED_AT >= 2:
        with open(f'ddrive/{GEN_ENDED_AT - 2}c.json') as f:
            ts = orjson.loads(f.read())
            ts = pool.map(Cell.deser, ts)
            cg.append(ts)
    if GEN_ENDED_AT >= 3:
        with open(f'ddrive/{GEN_ENDED_AT - 3}c.json') as f:
            ts = orjson.loads(f.read())
            ts = pool.map(Cell.deser, ts)
            cg.append(ts)

    for gen in range(GEN_ENDED_AT + 1, GEN_ENDS_AT + 1):
        print(f'Generation {gen}')
        print('Running Games...')
        
        cs = list()

        with tqdm.tqdm(total=GAME_PER_GEN) as pbar:
            for i, dat in enumerate(pool.imap(run_game, [gen - 1] * GAME_PER_GEN)):
                cs.extend(dat)
                pbar.update()

        print('Running Games Complete.')
        print('Processing Data...')

        cg.insert(0, cs)
        
        if len(cg) > 5:
            cg.pop()

        ts = pool.map(Cell.ser, cs)
        
        with open(f'ddrive/{gen}c.json', 'wb') as f:
            f.write(orjson.dumps(ts))

        print('Processing Data Complete.')
        print("Training...")
        
        xs = list()
        ys = list()
        acts = list()

        with tqdm.tqdm(total=len(cg)) as pbar:
            for cs in cg:
                tcg = CellGroup(cs, critic)
                
                xs.extend(tcg.xs)
                ys.extend(tcg.ys)
                acts.extend(tcg.acts)
                
                pbar.update()

        with tqdm.tqdm(total=10) as pbar:
            prog_callback = ProgCallback(pbar)

            cdat = MyDataset(xs, ys, acts, [(*NUM_GRID, NUM_CHANNEL), (1,), (1,)]).new()
            cdat = cdat.prefetch(tf.data.experimental.AUTOTUNE)

            critic.compile(optimizer=tf.keras.optimizers.SGD(0.1), loss='huber')
            hist = critic.fit(cdat, epochs=10, verbose=0, callbacks=[prog_callback])

        print(hist.history['loss'])

        print("Training Complete.")

        critic.save_weights(f'ddrive/{gen}c.h5')