
# coding: utf-8

# In[95]:


from skimage.transform import resize
from functools import partial
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.utils import Sequence
from keras.utils.np_utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D, Dense, Input, Flatten
from keras.models import Model
from keras.optimizers import SGD
from keras.losses import categorical_crossentropy

from keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard, LearningRateScheduler
import keras.backend as K

import tensorflow as tf


# In[96]:


class LRDecay:
    def __init__(self, config):
        self.epochs_per_round = int(np.ceil(max(config.train.client_fraction * config.train.num_clients, 1))) * config.train.epochs
        self.epochs = 0
        self.rounds = 0
        
    def __call__(self, epoch, lr):
        self.epochs += 1
        if (self.epochs > 1) & (((self.epochs - 1) % self.epochs_per_round) == 0):
            self.rounds += 1
            lr = lr * config.train.decay
            return lr
        return lr
            

def simple_cnn(img_size, n_classes):
        
    inputs = Input(img_size)
    conv1 = Conv2D(kernel_size=5, filters=32)(inputs)
    pool1 = MaxPooling2D(pool_size=2, strides=2)(conv1)
    conv2 = Conv2D(kernel_size=5, filters=64)(pool1)
    pool2 = MaxPooling2D(pool_size=2, strides=2)(conv2)
    flat = Flatten()(pool2)
    dense = Dense(units=512, activation='relu')(flat)
    out = Dense(units=n_classes, activation='softmax')(dense)
    model = Model(inputs=inputs, outputs=out)
    
    return model
    

def f1(y_true, y_pred, eps=1e-10):
    y_pred = tf.one_hot(tf.argmax(y_pred, axis=-1), axis=-1, depth=tf.shape(y_pred)[-1])  # (N, K)
    tp = tf.reduce_sum(y_true * y_pred, axis=0)  # (K,)
    pred_pos = tf.reduce_sum(y_pred, axis=0)  # (K,)
    pos = tf.reduce_sum(y_true, axis=0)  # (K,)
    
    score = 2 * prec * recall / (prec + recall)
    
# Implements the following logic
#     if pos == 0:
#         if pred_pos == 0:
#             score = -1
#         else:
#             score = 0
#     elif pred_pos == 0:
#         score = 0
#     else:
#         score = 2 * prec * recall / (prec + recall)
    
    return tf.where(tf.equal(pos, 0),
               tf.where(tf.equal(pred_pos, 0), -tf.ones_like(score), tf.zeros_like(score)),
               score)
    
                        

    
    
    prec = (tp + eps)/(pred_pos + eps)
    recall = (tp + eps)/(pos + eps)
    score = 2 * prec * recall / (prec + recall)
    return tf.reduce_mean(score)
    

def get_model(config):
    model = simple_cnn(config.data.img_size, config.data.n_classes)
    model.compile(optimizer=SGD(lr=config.train.learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    return model

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, df_path, 
                 batch_size, 
                 img_size, 
                 n_classes,
                 client_colm=None,
                 num=None, 
                 shuffle=True):
        'Initialization'
        self.num = num
        self.shuffle = shuffle
        df = pd.read_csv(df_path)
        if num is not None:
            rows = df[df[client_colm]==self.num]
        else:
            rows = df
        self.batch_size = batch_size
        self.img_size = img_size
        self.n_classes = n_classes
        self.filenames = rows.filename.values
        self.labels = rows.label.values
        
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.filenames) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.filenames))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_inds):
        'Generates data containing batch_size samples' 
        # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.img_size))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, idx in enumerate(batch_inds):
            # Store sample
            X[i,] =  plt.imread(self.filenames[idx])[...,:3]

            # Store class
            y[i] = self.labels[idx]

        return X, to_categorical(y, num_classes=self.n_classes)        


class UniformDataGenerator(DataGenerator):
    def __init__(self, *args, **kwargs):
        'Initialization'
        super(UniformDataGenerator, self).__init__(*args, **kwargs)
        assert ((self.batch_size % self.n_classes) == 0)
        self.unique_labels = np.unique(self.labels)
        self.label_dict = {label: [] for label in self.unique_labels}
        for label, name in zip(self.labels, self.filenames):
            self.label_dict[label].append(name)
        self.min_examples = min(map(len, self.label_dict.values()))
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(self.min_examples * len(self.unique_labels) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate filenames for each class
        classes = np.tile(np.arange(self.n_classes), [self.batch_size // self.n_classes])
        filenames = [np.random.choice(self.label_dict[label]) for label in classes]
        # Generate data
        X, y = self.__data_generation(filenames, classes)
        
        return X, y

        
    def __data_generation(self, filenames, labels):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.img_size))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, (filename, label) in enumerate(zip(filenames, labels)):
            # Store sample
            X[i,] =  plt.imread(filename)[...,:3]

            # Store class
            y[i] = label

        return X, to_categorical(y, num_classes=self.n_classes)    
    
# In[97]:


from easydict import EasyDict
# config = EasyDict(
# {
#  'mode': 'basic',
#  'data': {
#      'train_df_path' : 'train_resized.csv',
#      'val_df_path': 'val_resized.csv',
#      'img_size': (256, 256, 3),
# #      'batch_size': 10,
#      'batch_size': 32,
#      'n_classes': 12,
# #      'client_column': 'shard_non_iid',
     
#  },
#  'train' : {
#      'learning_rate': 1e-3,
#       'epochs': 100
# #      'epochs': 5,
# #       'client_fraction':0.2,
# #      'num_clients': 10,
# #      'num_rounds': 100
     
#  },
#     'log': {
#         'path': './results/01-basic',
#         'update_freq': 5
#     },
    
# #     'resume': {
# #         'path': './results/03-fed-avg-non_iid'
# #     }
# }
# )

config = EasyDict(
{
 'mode': 'fed',
 'data': {
     'train_df_path' : 'train_resized.csv',
     'val_df_path': 'val_resized.csv',
     'img_size': (256, 256, 3),
     'batch_size': 10,
     'n_classes': 12,
     'client_column': 'shard_non_iid',
     
 },
 'train' : {
     'learning_rate': 1e-3,
     'epochs': 5,
     'client_fraction': 0.2,
     'num_clients': 10,
     'num_rounds': 1000,
     'decay': 0.99
     
 },
    'log': {
        'path': './results/02-fed-non-iid',
        'update_freq': 5
    },
    
#     'resume': {
#         'path': './results/01-fed-non-iid'
#     }
}
)


# In[98]:


def client_update(config, num, model, decay=None):
    print(num)
    print(pd.DataFrame(pd.read_csv(config.data.train_df_path).query('{}=={}'.format(
        config.data.client_column, num)).label.value_counts()).T)
    dataset = DataGenerator(df_path=config.data.train_df_path, 
                          batch_size=config.data.batch_size, 
                          img_size=config.data.img_size, 
                          n_classes=config.data.n_classes,
                          client_colm=config.data.client_column,
                          num=num)
    
    if decay is not None:
        callbacks = [LearningRateScheduler(decay, verbose=1)]
    else:
        callbacks = []
    history = model.fit_generator(dataset, 
                        callbacks = callbacks,
                        epochs=config.train.epochs, 
                                  verbose=True,
                        workers=4
                                  , use_multiprocessing=True)
    weights = model.get_weights()
    return (weights,
            len(dataset.filenames),
            history.history['loss'][-1], 
            history.history['acc'][-1])
    


# In[99]:


def average_weights(weights, n_examples):
    weight_lists = map(list, zip(*weights))
    total_examples = np.sum(n_examples)
    return [np.sum(np.stack(w, axis=-1) * n_examples, axis=-1) / total_examples for w in weight_lists]

def train_basic(config):
    if not os.path.exists(config.log.path):
        os.makedirs(config.log.path)
    
    logpath = os.path.join(config.log.path, 'csvlogs')
    if not os.path.exists(logpath):
        os.makedirs(logpath)
    model = get_model(config)
    train_data = DataGenerator(df_path=config.data.train_df_path, 
                          batch_size=config.data.batch_size, 
                          img_size=config.data.img_size, 
                          n_classes=config.data.n_classes,
                          shuffle=True)
    valid_data = DataGenerator(df_path=config.data.val_df_path, 
                          batch_size=config.data.batch_size, 
                          img_size=config.data.img_size, 
                          n_classes=config.data.n_classes,
                          shuffle=False)
    #update_freq = config.log.batch_size * config.log.update_freq
    callbacks=[ModelCheckpoint(os.path.join(config.log.path, 'ckpt'), monitor='val_acc', save_best_only=True),
                   CSVLogger(os.path.join(config.log.path, 'csvlogs', 'trainval')),
                   TensorBoard(log_dir=os.path.join(config.log.path, 'logs'), 
                               update_freq=config.log.update_freq)]
    model.fit_generator(train_data, validation_data=valid_data,
                        epochs=config.train.epochs, verbose=True,
                        workers=4, 
                        callbacks=callbacks,
                        use_multiprocessing=True)
    

def fed_averaging(config):
    if not os.path.exists(config.log.path):
        os.makedirs(config.log.path)
    
    logpath = os.path.join(config.log.path, 'csvlogs')
    if not os.path.exists(logpath):
        os.makedirs(logpath)
        
    model = get_model(config)
    valid_data = DataGenerator(df_path=config.data.val_df_path, 
                          batch_size=config.data.batch_size, 
                          img_size=config.data.img_size, 
                          n_classes=config.data.n_classes,
                          shuffle=False)
    valid_log = pd.DataFrame({'round': [], 
                        'loss': [],
                        'acc': []})
    train_log = pd.DataFrame({'round': [], 
                    'loss': [],
                    'acc': []})
    
    best_score = 0
    num_done = 0
    if 'resume' in config.keys():
        resume_ckpt = os.path.join(config.resume.path, 'ckpt')
        print('Resuming from {}'.format(resume_ckpt))
        model.load_weights(resume_ckpt)
        prev_df = pd.read_csv(os.path.join(config.resume.path, 'csvlogs', 'valid'))
        best_score = prev_df.acc.max()
        num_done = int(prev_df.loc[prev_df.acc.idxmax()]['round'])
        
    print('Best valid acc so far: {}'.format(best_score))
    
    if 'decay' in config.train:
        decay = LRDecay(config)
    else:
        decay = None
    
    
    for t in range(num_done + 1, config.train.num_rounds + 1):
        print('Round {}/{}'.format(t, config.train.num_rounds))
        print('-' * 10)
        print('Training')
        global_weights = model.get_weights()
        _global_weights = [i.copy() for i in global_weights]
        m = int(np.ceil(max(config.train.client_fraction * config.train.num_clients, 1)))
        clients = np.random.permutation(config.train.num_clients)[:m]
        local_results = []
        
        for i, client in enumerate(clients):
            model.set_weights(global_weights)
            results = client_update(config, client, model, decay)
            local_results.append(results)
                
        
        local_weights, n_examples, _tloss, _tacc = zip(*local_results)
        tloss = np.mean(_tloss)
        tacc = np.mean(_tacc)
        
        if 'decay' in config.train:
            lr = config.train.learning_rate * (config.train.decay ** t)
        
        
        model.set_weights(average_weights(local_weights, n_examples))
        print('train_loss {:.4f}, train_acc {:.4f}'.format(tloss, tacc))
        print('Validation')
        vloss, vacc = model.evaluate_generator(valid_data,
                                               verbose=True,
                                               workers=4, use_multiprocessing=True)
        
        valid_log = valid_log.append(pd.DataFrame({'round': [t], 
                                 'loss': vloss,
                                 'acc': vacc}), ignore_index=True)
        train_log = train_log.append(pd.DataFrame({'round': [t], 
                         'loss': tloss,
                         'acc': tacc}), ignore_index=True)
        
        if vacc > best_score:
            model.save_weights(os.path.join(config.log.path, 'ckpt'))
            best_score = vacc
            
        valid_log[['round', 'loss', 'acc']].to_csv(os.path.join(logpath, 'valid'), index=False)
        train_log[['round', 'loss', 'acc']].to_csv(os.path.join(logpath, 'train'), index=False)
        
        print('val_loss {:.4f}, val_acc {:.4f}'.format(vloss, vacc))
        print()
        print()
        

    model.save_weights(os.path.join(config.log.path, 'last'))


# In[ ]:


K.clear_session()
if config.mode == 'basic':
    train_basic(config)
elif config.mode == 'fed':
    fed_averaging(config)

