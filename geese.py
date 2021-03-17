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
NUM_ACT = 4
NUM_GEESE = 4

GEN_ENDED_AT = -1
GAME_PER_GEN = 400

NUM_LAMBDA = 0.8
NUM_P = 0.95

STOCK_X = tf.convert_to_tensor(np.zeros((*NUM_GRID, 4)), dtype='float32')
STOCK_ACT = [Action(i + 1) for i in range(NUM_ACT)]

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
        dct['s'] = tf.convert_to_tensor(np.frombuffer(base64.decodebytes(dct['s'].encode('utf-8')), dtype=np.float32).reshape((*NUM_GRID, 4)))
        out = Cell(dct['s'], dct['a'], dct['r'])

        if 'ss' in dct:
            out.ss = tf.convert_to_tensor(np.frombuffer(base64.decodebytes(dct['ss'].encode('utf-8')), dtype=np.float32).reshape((*NUM_GRID, 4)))
        
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

        for b in range(len(indexs) // 2048 + 1):
            sss = tf.stack(ss[b * 2048:(b + 1) * 2048])
            y = target(sss, training=True)
            y = tf.nn.relu(tf.reduce_max(y, axis=1))
            y *= NUM_LAMBDA

            for i, ii in enumerate(indexs[b * 2048:(b + 1) * 2048]):
                self.ys[ii] += y[i]

class Goose:
    def __init__(self, critic):
        self.critic = critic

        self.cs = list()
        self.act = None

    def __call__(self, obs, _):
        x = obs_to_x(obs, self.act)

        scores = [float(score) + 0.1 for score in tf.nn.relu(self.critic(x))[0]]

        sm = 1.5 * sum(scores)

        scores = [score / sm for score in scores]
        scores[scores.index(max(scores))] += 1 / 3

        p = random.choices(population=scores, weights=scores)[0]
        i = scores.index(p)

        self.cs.append(Cell(x, i, 0.0))
        
        self.act = STOCK_ACT[i]

        return self.act.name

def run_game(gen):
    global tcritic, tgen
    
    if tcritic == None or tgen != gen:
        tcritic = Critic([32, 32, 48, 48, 48, 48, 64, 64], NUM_ACT, STOCK_X)
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
                    geese[ii].cs[i - 1].r += 1
    
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

def obs_to_x(obs, act):
    x = [[[0 for _ in range(4)] for _ in range(NUM_GRID[1])] for _ in range(NUM_GRID[0])]
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

        x[r][c][0] = 1.

    for i, goose in enumerate(geese):
        if goose:
            if i != index:
                r, c = pos_to_coord(goose[0])

                r = (r + rc) % NUM_GRID[0]
                c = (c + cc) % NUM_GRID[1]

                for tact in STOCK_ACT:
                    rt, ct = tact.to_row_col()

                    x[(r + rt) % NUM_GRID[0]][(c + ct) % NUM_GRID[1]][2] = 1.
            
            for block in goose:
                r, c = pos_to_coord(block)

                r = (r + rc) % NUM_GRID[0]
                c = (c + cc) % NUM_GRID[1]

                x[r][c][1] = 1

            r, c = pos_to_coord(goose[-1])

            r = (r + rc) % NUM_GRID[0]
            c = (c + cc) % NUM_GRID[1]

            x[r][c][3] = 1
            
    if act:
        r, c = pos_to_coord(geese[index][0])
        rt, ct = act.opposite().to_row_col()
        
        r = (r + rt + rc) % NUM_GRID[0]
        c = (c + ct + cc) % NUM_GRID[1]
        
        x[r][c][1] = 1
        x[r][c][3] = 0
    
    return tf.convert_to_tensor(x, dtype='float32')

class ProgCallback(tf.keras.callbacks.Callback):
    def __init__(self, pbar, **kwargs):
        super(ProgCallback, self).__init__(**kwargs)
        self.pbar = pbar
 
    def on_epoch_end(self, _, logs=None):
        self.pbar.update()

class MyDataset:
    def __init__(self, xs, ys, acts, shapes, batch_size=2048):
        self.xs = xs
        self.ys = ys
        self.acts = acts
        self.shapes = shapes
        self.batch_size = batch_size

    def _generator(self):
        indexs = np.arange(len(self.ys))
        np.random.shuffle(indexs)
    
        for x in range(len(self.ys) // self.batch_size):
            X = np.empty((self.batch_size, *self.shapes[0]))
            y = np.empty((self.batch_size, *self.shapes[1]))
            a = np.empty((self.batch_size, *self.shapes[2]))
    
            for i, ii in enumerate(indexs[np.arange(x * self.batch_size, (x + 1) * self.batch_size)]):
                X[i,] = self.xs[ii]
                y[i,] = self.ys[ii]
                a[i,] = self.acts[ii]
    
            yield X, y, a

        x = len(self.ys) % self.batch_size

        X = np.empty((x, *self.shapes[0]))
        y = np.empty((x, *self.shapes[1]))
        a = np.empty((x, *self.shapes[2]))

        for i, ii in enumerate(indexs[np.arange(len(self.ys) - x, len(self.ys))]):
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
    
    critic = Critic([32, 32, 48, 48, 48, 48, 64, 64], NUM_ACT, STOCK_X)
    critic(critic.stock)
    
    target = critic.clone(critic)
    
    if GEN_ENDED_AT >= 0:
        critic.load_weights(f'ddrive/{GEN_ENDED_AT}c.h5')
        target.load_weights(f'ddrive/{GEN_ENDED_AT}t.h5')
    
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

    for gen in range(GEN_ENDED_AT + 1, 100):
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
                tcg = CellGroup(cs, target)
                
                xs.extend(tcg.xs)
                ys.extend(tcg.ys)
                acts.extend(tcg.acts)
                
                pbar.update()

        with tqdm.tqdm(total=10) as pbar:
            prog_callback = ProgCallback(pbar)

            cdat = MyDataset(xs, ys, acts, [(*NUM_GRID, 4), (1,), (1,)]).new()
            cdat = cdat.prefetch(2)

            critic.compile(optimizer=tf.keras.optimizers.SGD(0.02), loss='huber')
            hist = critic.fit(cdat, epochs=10, verbose=0, callbacks=[prog_callback])

        print(hist.history['loss'])

        for i in range(len(critic.tower)):
            w0 = target.tower[i].get_weights()
            w1 = critic.tower[i].get_weights()
            target.tower[i].set_weights([w0[ii] * NUM_P + (1 - NUM_P) * w1[ii] for ii in range(len(w0))])

        w0 = target.out.get_weights()
        w1 = critic.out.get_weights()
        target.out.set_weights([w0[ii] * NUM_P + (1 - NUM_P) * w1[ii] for ii in range(len(w0))])

        w0 = target.act.get_weights()
        w1 = critic.act.get_weights()
        target.act.set_weights([w0[ii] * NUM_P + (1 - NUM_P) * w1[ii] for ii in range(len(w0))])

        print("Training Complete.")

        critic.save_weights(f'ddrive/{gen}c.h5')
        target.save_weights(f'ddrive/{gen}t.h5')