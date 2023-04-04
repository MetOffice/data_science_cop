import sys
import os
import time
import numpy as np
import json

os.environ.pop('TF_CONFIG', None)

WORKER_CLUSTER_PATHLIST = "./cluster_addresses.txt"
worker_index = sys.argv[1]
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import subprocess

result = subprocess.run(['ip','a'], stdout=subprocess.PIPE)
sbp_out = result.stdout.decode("utf-8")
for line in sbp_out.split('\n'):
    if 'inet' in line:
        if '/22' in line:
            worker_add = line.split('inet')[-1].split('/22')[0].strip()
            

from random import randint
def random_with_N_digits(n):
    range_start = 10**(n-1)
    range_end = (10**n)-1
    return randint(range_start, range_end)

port = random_with_N_digits(5)
string_to_write = str(worker_add) + ':' + str(port)
tmp_fname = "./slurmworkerip_"+str(worker_index)+".txt"
with open(tmp_fname, 'w') as tmp:
    tmp.write(string_to_write)

while(not os.path.exists(WORKER_CLUSTER_PATHLIST)):
      time.sleep(5)

with open(WORKER_CLUSTER_PATHLIST, 'r') as f:
    cluster_for_config = f.readlines()
for i, add in enumerate(cluster_for_config):
    cluster_for_config[i] = add.strip('\n')

tf_config = {
    'cluster': {
        'worker': cluster_for_config
    },
    'task': {'type': 'worker', 'index': worker_index}
}

import tensorflow as tf

def mnist_dataset(batch_size):
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    # The `x` arrays are in uint8 and have values in the [0, 255] range.
    # You need to convert them to float32 with values in the [0, 1] range.
    x_train = x_train / np.float32(255)
    y_train = y_train.astype(np.int64)
    train_dataset = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train)).shuffle(60000).repeat().batch(batch_size)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.FILE
    train_dataset.with_options(options)
    return train_dataset

def build_and_compile_cnn_model():
    model = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=(28, 28)),
      tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
      tf.keras.layers.Conv2D(32, 3, activation='relu'),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(10)
    ])
    model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
      metrics=['accuracy'])
    return model

cluster = tf.train.ClusterSpec(tf_config['cluster'])

cluster_resolver = tf.distribute.cluster_resolver.SimpleClusterResolver(cluster, task_type="worker",
                                           task_id=int(worker_index),
                                           num_accelerators={"CPU": 1})


strategy = tf.distribute.MultiWorkerMirroredStrategy(cluster_resolver, communication_options=tf.distribute.experimental.CommunicationOptions(implementation=tf.distribute.experimental.CommunicationImplementation.AUTO))

os.environ["TF_CONFIG"] = json.dumps(tf_config)
tf_config = json.loads(os.environ["TF_CONFIG"])
per_worker_batch_size = 64
num_workers = len(tf_config['cluster']['worker'])
global_batch_size = per_worker_batch_size * num_workers



with strategy.scope():
  # Model building/compiling need to be within `strategy.scope()`.
    multi_worker_dataset = mnist_dataset(global_batch_size)
    multi_worker_model = build_and_compile_cnn_model()
multi_worker_model.fit(multi_worker_dataset, epochs=10)
