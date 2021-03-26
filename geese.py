import os
 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

if __name__ != '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

import tensorflow as tf
import numpy as np
import random
import itertools
import tqdm
import pickle
import lzma
from multiprocessing import shared_memory, resource_tracker
from pathos.multiprocessing import ProcessPool
from pathos.helpers import mp
from kaggle_environments.envs.hungry_geese.hungry_geese import Action
from kaggle_environments import make

NUM_GRID = (7, 11)
NUM_CHANNEL = 7
NUM_ACT = 4
NUM_GEESE = 4
    
GAME_PER_GEN = 400
NUM_REPLAY_BUF = 5

NUM_LAMBDA = 0.9
NUM_RAND = 0.4
NUM_SCALE = 0.1

STOCK_X = tf.convert_to_tensor(np.zeros((*NUM_GRID, NUM_CHANNEL)), dtype='int8')
STOCK_ACT = [Action(i + 1) for i in range(NUM_ACT)]

class Block(tf.keras.layers.Layer):
    def __init__(self, flt, **kwargs):
        super(Block, self).__init__(**kwargs)
        
        self.dense = tf.keras.layers.Dense(flt, use_bias=False, dtype='float16')
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
        self.out = tf.keras.layers.Dense(out, dtype='float16')
    
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
        x, xx, ss, r, a = dat

        y = self(xx, training=True)
        y = tf.reduce_max(y, axis=1, keepdims=True)
        y = tf.where(ss, y, tf.zeros_like(y))
        y *= NUM_LAMBDA
        y += r

        with tf.GradientTape() as tape:
            y_pred = tf.gather(self(x, training=True), a, batch_dims=1)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        
        t = self.trainable_variables
        g = tape.gradient(loss, t)

        self.optimizer.apply_gradients(zip(g, t))
        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}

class Cell:
    def __init__(self, s, a, r):
        self.s = s
        self.ss = False
        self.a = a
        self.r = r

class CellGroup:
    def __init__(self, create=True, cs=None):
        self.s_sm = shared_memory.SharedMemory(create=create, name='s_sm', size=(2400000 * NUM_GRID[0] * NUM_GRID[1] * NUM_CHANNEL))
        self.s = np.ndarray((2400000, *NUM_GRID, NUM_CHANNEL), dtype=np.int8, buffer=self.s_sm.buf)

        self.ss_sm = shared_memory.SharedMemory(create=create, name='ss_sm', size=2400000)
        self.ss = np.ndarray((2400000, 1), dtype=bool, buffer=self.ss_sm.buf)

        self.r_sm = shared_memory.SharedMemory(create=create, name='r_sm', size=4800000)
        self.r = np.ndarray((2400000, 1), dtype=np.float16, buffer=self.r_sm.buf)

        self.a_sm = shared_memory.SharedMemory(create=create, name='a_sm', size=9600000)
        self.a = np.ndarray((2400000, 1), dtype=np.int32, buffer=self.a_sm.buf)

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
            self.ss[p] = cell.ss
            self.r[p] = cell.r
            self.a[p] = cell.a
        
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

def run_game(weights):
    critic = Critic([512, 256, 256, 128], NUM_ACT, STOCK_X)
    critic(critic.stock)
    critic.set_weights(pickle.loads(weights))
        
    geese = [Goose(critic) for _ in range(NUM_GEESE)]
    env = make('hungry_geese')
    steps = env.run(geese)

    for i, step in enumerate(steps):
        if i <= 0:
            continue

        obs = step[0]['observation']

        for ii, goose in enumerate(obs['geese']):
            if goose:
                geese[ii].cs[i - 1].r += (len(goose) - 1) * NUM_SCALE
            elif steps[i - 1][0]['observation']['geese'][ii]:
                geese[ii].cs[i - 1].r -= (75 + 25 * len(steps[i - 1][0]['observation']['geese'][ii])) * NUM_SCALE

    dat = list()

    for goose in geese:
        for c in goose.cs[:-1]:
            c.ss = True
        
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

        x[r][c][6] = 1

    for i, goose in enumerate(geese):
        if goose:
            r, c = pos_to_coord(goose[0])
            
            r = (r + rc) % NUM_GRID[0]
            c = (c + cc) % NUM_GRID[1]
            
            x[r][c][4] = 1
            x[r][c][5] = 1
            
            if acts[i] != None:
                x[r][c][act_to_id(acts[i])] = 1
            
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
    
    return tf.convert_to_tensor(x, dtype='int8')

class ProgCallback(tf.keras.callbacks.Callback):
    def __init__(self, pbar, **kwargs):
        super(ProgCallback, self).__init__(**kwargs)
        self.pbar = pbar
 
    def on_epoch_end(self, _, logs=None):
        self.pbar.update()

class MyDataset:
    def __init__(self, total, pool, shapes, batch_size=2048):
        self.total = total
        self.pool = pool
        self.shapes = shapes
        self.batch_size = batch_size

    def _generator(self):
        indexs = np.arange(self.total)
        np.random.shuffle(indexs)

        stride = (NUM_GRID[0] * NUM_GRID[1] * NUM_CHANNEL * 1, 1, 2, 4)

        s_sm = shared_memory.SharedMemory(name='s_sm')
        ss_sm = shared_memory.SharedMemory(name='ss_sm')
        r_sm = shared_memory.SharedMemory(name='r_sm')
        a_sm = shared_memory.SharedMemory(name='a_sm')
        
        for x in range(len(indexs) // self.batch_size + 1):
            index = indexs[x * self.batch_size:(x + 1) * self.batch_size]

            if len(index) > 0:
                X = np.empty((len(index), *self.shapes[0]), dtype=np.int8)
                XX = np.empty((len(index), *self.shapes[0]), dtype=np.int8)
                ss = np.empty((len(index), *self.shapes[1]), dtype=bool)
                r = np.empty((len(index), *self.shapes[2]), dtype=np.float16)
                a = np.empty((len(index), *self.shapes[3]), dtype=np.int32)

                for i, ii in enumerate(index):
                    X[i,] = np.ndarray(self.shapes[0], dtype=np.int8, buffer=s_sm.buf[stride[0] * ii:stride[0] * (ii + 1)])
                    ss[i,] = np.ndarray(self.shapes[1], dtype=bool, buffer=ss_sm.buf[stride[1] * ii:stride[1] * (ii + 1)])
                    r[i,] = np.ndarray(self.shapes[2], dtype=np.float16, buffer=r_sm.buf[stride[2] * ii:stride[2] * (ii + 1)])
                    a[i,] = np.ndarray(self.shapes[3], dtype=np.int32, buffer=a_sm.buf[stride[3] * ii:stride[3] * (ii + 1)])

                    if ss[i][0]:
                        XX[i,] = np.ndarray(self.shapes[0], dtype=np.int8, buffer=s_sm.buf[stride[0] * (ii + 1):stride[0] * (ii + 2)])
                    else:
                        XX[i,] = np.zeros(self.shapes[0], dtype=np.int8)
        
                yield X, XX, ss, r, a

        s_sm.close()
        ss_sm.close()
        r_sm.close()
        a_sm.close()

    def new(self):
        return tf.data.Dataset.from_generator(
            self._generator,
            output_types=('int8', 'int8', 'bool', 'float16', 'int32'), 
            output_shapes=((None, *self.shapes[0]), (None, *self.shapes[0]), (None, *self.shapes[1]), (None, *self.shapes[2]), (None, *self.shapes[3]))
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

if __name__ == '__main__':  
    remove_shm_from_resource_tracker()

    GEN_ENDED_AT = int(input())
    GEN_ENDS_AT = int(input())

    mp.set_start_method('spawn')

    pool = ProcessPool(mp.cpu_count())

    critic = Critic([512, 256, 256, 128], NUM_ACT, STOCK_X)
    critic(critic.stock)

    if GEN_ENDED_AT >= 0:
        with open(f'ddrive/{GEN_ENDED_AT}.txt', 'rb') as f:
            weights = pickle.loads(lzma.decompress(f.read()))

        critic.set_weights(weights)

    critic.compile(optimizer=tf.keras.optimizers.SGD(0.1), loss=tf.keras.losses.Huber(10))

    cg = CellGroup()
    
    for gen in range(GEN_ENDED_AT + 1, GEN_ENDS_AT + 1):
        print(f'Generation {gen}')
        
        print('Running Games...')

        weights = pickle.dumps(critic.get_weights())

        cs = list()

        with tqdm.tqdm(total=GAME_PER_GEN) as pbar:
            for i, dat in enumerate(pool.imap(run_game, itertools.repeat(weights, GAME_PER_GEN))):
                cs.extend(dat)
                pbar.update()

        print('Running Games Complete.')
        print('Processing Data...')

        cg.add(cs)

        if len(cg.cs) > NUM_REPLAY_BUF:
            cg.pop()

        total = cg.cl + 1

        dat = MyDataset(total, pool, [(*NUM_GRID, NUM_CHANNEL), (1,), (1,), (1,)]).new()
        dat = dat.prefetch(tf.data.AUTOTUNE)

        print('Processing Data Complete.')
        print("Training...")

        with tqdm.tqdm(total=10) as pbar:
            prog_callback = ProgCallback(pbar)

            hist = critic.fit(dat, epochs=10, verbose=0, callbacks=[prog_callback])

        print(hist.history['loss'])

        print("Training Complete.")

        with open(f'ddrive/{gen}.txt', 'wb') as f:
            f.write(lzma.compress(pickle.dumps(critic.get_weights()), preset=9))

        pool.close()
        pool.join()
        pool.restart()

    cg.close(full=True)
