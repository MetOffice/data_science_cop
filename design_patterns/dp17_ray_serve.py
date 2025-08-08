from typing import Dict
import numpy as np


from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import load_model

import ray




# Step 1: Create a Ray Dataset from in-memory Numpy arrays.
# You can also create a Ray Dataset from many other sources and file
# formats.
ds = ray.data.from_numpy(np.ones((1, 100)))

# Step 2: Define a Predictor class for inference.
# Use a class to initialize the model just once in `__init__`
# and re-use it for inference across multiple batches.
class TFPredictor:
    def __init__(self):
        from tensorflow import keras

        # Load a dummy neural network.
        # Set `self.model` to your pre-trained Keras model.
        input_layer = keras.Input(shape=(100,))
        output_layer = keras.layers.Dense(1, activation="sigmoid")
        self.model = keras.Sequential([input_layer, output_layer])

    # Logic for inference on 1 batch of data.
    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        # Get the predictions from the input batch.
        return {"output": self.model(batch["data"]).numpy()}

# Use 2 parallel actors for inference. Each actor predicts on a
# different partition of data.
scale = ray.data.ActorPoolStrategy(size=1)
# Step 3: Map the Predictor over the Dataset to get predictions.
predictions = ds.map_batches(TFPredictor, compute=scale)
 # Step 4: Show one prediction output.
predictions.show(limit=1)
