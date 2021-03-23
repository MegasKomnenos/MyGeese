import os
 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

import tensorflow as tf
import numpy as np
import random
import itertools
import tqdm
import json
import base64
import dill
import time
import math
from multiprocessing import shared_memory, resource_tracker
from pathos.multiprocessing import ProcessPool
from pathos.helpers import mp
from kaggle_environments.envs.hungry_geese.hungry_geese import Action
from kaggle_environments import make
from pathfinding.core.grid import Grid

NUM_GRID = (7, 11)
NUM_CHANNEL = 6
NUM_ACT = 4
NUM_GEESE = 4
    
GAME_PER_GEN = 300
NUM_REPLAY_BUF = 6

NUM_LAMBDA = 0.8
NUM_RAND = 2

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
    
    def call(self, inp, training=False):
        x = inp

        if len(x.shape) < 4:
            x = tf.expand_dims(x, 0)
            
        x = self.flt(x)

        for block in self.tower:
            x = block(x, training=training)
            
        x = self.out(x)
            
        return x

    @classmethod
    def clone(cls, self):
        out = cls(**self.stats)
        out(self.stock)

        for i in range(len(self.tower)):
            out.tower[i].set_weights(self.tower[i].get_weights())
            
        out.out.set_weights(self.out.get_weights())

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
    def __init__(self, create=True, cs=None):
        self.s_sm = shared_memory.SharedMemory(create=create, name='s_sm', size=2217600000)
        self.s = np.ndarray((1200000, *NUM_GRID, NUM_CHANNEL), dtype=np.float32, buffer=self.s_sm.buf)

        self.ss_sm = shared_memory.SharedMemory(create=create, name='ss_sm', size=1200000)
        self.ss = np.ndarray((1200000, 1), dtype=np.bool, buffer=self.ss_sm.buf)

        self.r_sm = shared_memory.SharedMemory(create=create, name='r_sm', size=4800000)
        self.r = np.ndarray((1200000, 1), dtype=np.float32, buffer=self.r_sm.buf)

        self.a_sm = shared_memory.SharedMemory(create=create, name='a_sm', size=4800000)
        self.a = np.ndarray((1200000, 1), dtype=np.int32, buffer=self.a_sm.buf)

        if create:
            self.cs = list()
            self.cl = -1
        else:
            self.cs = list(cs)
            self.cl = self.cs[-1]
    
    def add(self, cells):
        p = self.cl

        for cell in cells:
            p += 1

            self.s[p] = cell.s
            self.r[p] = cell.r
            self.a[p] = cell.a

            if cell.ss != None:
                self.ss[p + 1] = True
            else:
                self.ss[p + 1] = False
        
        self.cl = p
        self.cs.append(p)
    
    def pop(self, i=0):
        p = self.cs[i - 1] if i > 0 else -1
        pp = self.cs[i]
        
        self.s[p + 1:self.cl + 1 - pp + p] = self.s[pp + 1:self.cl + 1]
        self.r[p + 1:self.cl + 1 - pp + p] = self.r[pp + 1:self.cl + 1]
        self.a[p + 1:self.cl + 1 - pp + p] = self.a[pp + 1:self.cl + 1]
        self.ss[p + 1:self.cl + 1 - pp + p] = self.ss[pp + 1:self.cl + 1]

        pp -= p

        self.cs.pop(i)

        for ii in range(i, len(self.cs)):
            self.cs[ii] -= pp
        
        if self.cs:
            self.cl = self.cs[-1]
        else:
            self.cl = -1
    
    def get_y(self, target):
        ys = np.copy(self.r[:self.cl + 1])
        s = self.s[:self.cl + 1]
        ss = self.ss[:self.cl + 1]

        for b in range((self.cl + 1) // 4096 + 1):
            ts = s[b * 4096:(b + 1) * 4096]
            y = target(ts, training=True)
            y = tf.reduce_max(y, axis=1, keepdims=True)
            y = np.where(ss[b * 4096:(b + 1) * 4096], y, np.zeros(y.shape))
            
            if b == 0:
                ys[:len(y) - 1] += y[1:]
            else:
                ys[b * 4096 - 1:b * 4096 - 1 + len(y)] += y

        return ys

    def close(self, full=False):
        self.s_sm.close()
        self.ss_sm.close()
        self.r_sm.close()
        self.a_sm.close()

        if full:
            self.s_sm.unlink()
            self.ss_sm.unlink()
            self.r_sm.unlink()
            self.a_sm.unlink()

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

        scores = [float(score) for score in tf.nn.softmax(self.critic(x) / NUM_RAND)[0]]

        p = random.choices(population=scores, weights=scores)[0]
        i = scores.index(p)

        self.cs.append(Cell(x, i, 0.0))

        return STOCK_ACT[i].name

def run_game(critics):
    critic = dill.loads(critics)
        
    geese = [Goose(critic) for _ in range(NUM_GEESE)]
    env = make('hungry_geese')
    steps = env.run(geese)

    for i, step in enumerate(steps):
        if i <= 0:
            continue

        obs = step[0]['observation']
        matrix = [[1 for _ in range(NUM_GRID[0])] for _ in range(NUM_GRID[1])]

        for goose in obs['geese']:
            for block in goose:
                r, c = pos_to_coord(block)
                
                matrix[c][r] = 0

        pos = list()
        mn = 0.

        for ii, goose in enumerate(obs['geese']):
            if goose:
                grid = Grid(matrix=matrix)

                grid.set_passable_left_right_border()
                grid.set_passable_up_down_border()

                for column in grid.nodes:
                    for node in column:
                        node.g = -1
                
                lst = [grid.node(*pos_to_coord(goose[0]))]
                lst[0].g = 0
                lst[0].opened = 1

                while lst:
                    lst.sort(reverse=True, key=(lambda x: x.g))
                    
                    cur = lst.pop()

                    neighbs = [neighb for neighb in grid.neighbors(cur) if neighb.opened == 0]

                    for neighb in neighbs:
                        neighb.g = cur.g + 1
                        neighb.opened = 1
                        
                        lst.append(neighb)
                        
                sm = 0.

                for column in grid.nodes:
                    for node in column:
                        if node.g > 0:
                            sm += 1 / node.g**2
                
                pos.append(sm)
                mn += sm
            else:
                pos.append(None)

        if mn > 0.:
            mn /= len(pos) - pos.count(None)

            for ii, goose in enumerate(obs['geese']):
                if goose:
                    geese[ii].cs[i - 1].r += math.exp(3 * (pos[ii] - mn) / mn)
                    geese[ii].cs[i - 1].r += math.log(len(goose))
                elif steps[i - 1][0]['observation']['geese'][ii]:
                    geese[ii].cs[i - 1].r -= 20.
            
        if i + 1 == len(steps):
            for ii, goose in enumerate(obs['geese']):
                if goose:
                    geese[ii].cs[i - 1].r += 20.

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
    def __init__(self, ys, shapes, batch_size=1024):
        self.ys = ys
        self.shapes = shapes
        self.batch_size = batch_size

    def _generator(self):
        indexs = np.arange(len(self.ys))
        np.random.shuffle(indexs)
        
        for x in range(len(indexs) // self.batch_size + 1):
            index = indexs[x * self.batch_size:(x + 1) * self.batch_size]

            if len(index) > 0:
                s_sm = shared_memory.SharedMemory(name='s_sm')
                a_sm = shared_memory.SharedMemory(name='a_sm')

                X = np.empty((len(index), *self.shapes[0]), dtype=np.float32)
                y = np.empty((len(index), *self.shapes[1]), dtype=np.float32)
                a = np.empty((len(index), *self.shapes[2]), dtype=np.int32)

                for i, ii in enumerate(index):
                    X[i,] = np.ndarray(self.shapes[0], dtype=np.float32, buffer=s_sm.buf[NUM_GRID[0] * NUM_GRID[1] * NUM_CHANNEL * 4 * ii:NUM_GRID[0] * NUM_GRID[1] * NUM_CHANNEL * 4 * (ii + 1)])
                    y[i,] = self.ys[ii]
                    a[i,] = np.ndarray(self.shapes[2], dtype=np.int32, buffer=a_sm.buf[4 * ii:4 * (ii + 1)])

                s_sm.close()
                a_sm.close()
        
                yield X, y, a

    def new(self):
        return tf.data.Dataset.from_generator(
            self._generator,
            output_types=('float32', 'float32', 'int32'), 
            output_shapes=((None, *self.shapes[0]), (None, *self.shapes[1]), (None, *self.shapes[2]))
        )

def remove_shm_from_resource_tracker():
    """Monkey-patch multiprocessing.resource_tracker so SharedMemory won't be tracked

    More details at: https://bugs.python.org/issue38119
    """

    def fix_register(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.register(self, name, rtype)
    resource_tracker.register = fix_register

    def fix_unregister(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.unregister(self, name, rtype)
    resource_tracker.unregister = fix_unregister

    if "shared_memory" in resource_tracker._CLEANUP_FUNCS:
        del resource_tracker._CLEANUP_FUNCS["shared_memory"]

def fit_proc(gen, cs):
    remove_shm_from_resource_tracker()

    critic = Critic([512, 128, 128, 256], NUM_ACT, STOCK_X)
    critic(critic.stock)

    if gen >= 1:
        critic.load_weights(f'ddrive/{gen - 1}c.h5')

    cg = CellGroup(create=False, cs=cs)

    dat = MyDataset(cg.get_y(critic), [(*NUM_GRID, NUM_CHANNEL), (1,), (1,)]).new()
    dat = dat.prefetch(tf.data.experimental.AUTOTUNE)

    critic.compile(optimizer=tf.keras.optimizers.SGD(0.04), loss='huber')

    print('Processing Data Complete.')
    print("Training...")

    with tqdm.tqdm(total=10) as pbar:
        prog_callback = ProgCallback(pbar)

        hist = critic.fit(dat, epochs=10, verbose=0, callbacks=[prog_callback])

    print(hist.history['loss'])

    print("Training Complete.")

    critic.save_weights(f'ddrive/{gen}c.h5')

    cg.close()

if __name__ == '__main__':  
    GEN_ENDED_AT = int(input())
    GEN_ENDS_AT = int(input())

    mp.set_start_method('spawn')

    pool = ProcessPool(mp.cpu_count())

    critic = Critic([512, 128, 128, 256], NUM_ACT, STOCK_X)
    critic(critic.stock)

    cg = CellGroup()

    for gen in range(GEN_ENDED_AT + 1, GEN_ENDS_AT + 1):
        print(f'Generation {gen}')
        
        print('Running Games...')

        if gen >= 1:
            critic.load_weights(f'ddrive/{gen - 1}c.h5')

        critics = dill.dumps(critic)

        cs = list()

        with tqdm.tqdm(total=GAME_PER_GEN) as pbar:
            for i, dat in enumerate(pool.imap(run_game, itertools.repeat(critics, GAME_PER_GEN))):
                cs.extend(dat)
                pbar.update()
        
        pool.close()
        pool.join()

        print('Running Games Complete.')
        print('Processing Data...')

        cg.add(cs)

        if len(cg.cs) > NUM_REPLAY_BUF:
            cg.pop()

        os.environ['CUDA_VISIBLE_DEVICES'] = "0"

        tcs = list(cg.cs)

        p = mp.Process(target=fit_proc, args=(gen, tcs))
        p.start()
        p.join()
        p.close()

        os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

        pool.restart()

    cg.close(full=True)
