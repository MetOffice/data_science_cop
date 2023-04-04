import os
import sys
import glob
import json
import numpy as np

file_to_write = "./cluster_addresses.txt"
try:
    os.remove(file_to_write)
except:
    ...
    
try:
    to_cleanup = glob.glob("./slurm*")
    for fileip in to_cleanup:
        os.remove(fileip)
except:
    ...
    
os.environ.pop('TF_CONFIG', None)


import subprocess
result = subprocess.run(['ip','a'], stdout=subprocess.PIPE)
sbp_out = result.stdout.decode("utf-8")
# print(sbp_out)
for line in sbp_out.split('\n'):
    if 'inet' in line:
        if '/22' in line:
            chief_address = line.split('inet')[-1].split('/22')[0].strip()
            
print(chief_address) 
chief_address = chief_address + ':12365'



submit_to_slurm_string = """#!/bin/bash -l
#SBATCH --qos=normal
#SBATCH --mem=8G
#SBATCH --ntasks=4
#SBATCH --time=298

conda activate piptf

python3 tf-dist-worker.py $ARG1

"""

tf_config_args = ["$ARG1"]

num_workers = 8

for i in range(num_workers-1):
    to_submit_cp = submit_to_slurm_string
    to_submit_cp = to_submit_cp.replace(tf_config_args[0], (str(i+1) + ' ')) # arg1 = worker index, off by one to avoid cheifindex
        
    
    tmp_fname = "./slurmsubmitworker.batch"
    with open(tmp_fname, 'w') as tmp:
        tmp.writelines(to_submit_cp)
        
    script_output = subprocess.run(['sbatch',tmp_fname])


cluster_addresses = ["NONE"]*num_workers
cluster_addresses[0] = chief_address
while "NONE" in cluster_addresses:
    for i in range(1, num_workers):
        path_to_check = f'./slurmworkerip_{i}.txt'
        if os.path.exists(path_to_check):
            with open(path_to_check, 'r') as f:
                cluster_addresses[i] = f.read().strip("\n")
                
                
                
with open(file_to_write, 'w') as f:
    for item in cluster_addresses:
        to_write = item + '\n'
        f.write(to_write)
        
        
        
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

tf_config = {
    'cluster': {
        'worker': cluster_addresses
    },
    'task': {'type': 'worker', 'index': '0'}
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
                                           task_id=0,
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
