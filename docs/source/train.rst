DNN Training
=====

This document explains, line by line, the full workflow for training a Deep Neural Network (DNN) classifier using all NF dataset.
The same structure applies to all NF-based datasets.


1. Import Dependencies
----------------------

.. code-block:: python

   import pandas as pd
   import numpy as np

- ``pandas`` is used for loading and manipulating dataframes.
- ``numpy`` supports numerical operations.


2. Load Train, Validation, and Test Splits
------------------------------------------

.. code-block:: python

   df_train = pd.read_parquet("../Dataset/NF-ToN-IoT/NF-ToN-IoT-V2_train.parquet")
   df_valid = pd.read_parquet("../Dataset/NF-ToN-IoT/NF-ToN-IoT-V2_valid.parquet")
   df_test = pd.read_parquet("../Dataset/NF-ToN-IoT/NF-ToN-IoT-V2_test.parquet")

- Loads the preprocessed dataset splits stored in Parquet format.
- These files are generated during the data preparation stage.

Replace the filename with the correct dataset for other NF variants.


3. Separate Features and Labels
-------------------------------

.. code-block:: python

   load_X_df_train = df_train.drop(["Label"],axis=1)
   Y_df_train = df_train["Label"]

   load_X_df_valid = df_valid.drop(["Label"],axis=1)
   Y_df_valid = df_valid["Label"]

   load_X_df_test = df_test.drop(["Label"],axis=1)
   Y_df_test = df_test["Label"]

- ``drop(["Label"], axis=1)`` removes the target column to keep only features.
- ``Y_df_*`` keeps the label column for each dataset.


4. Normalize Features Using QuantileTransformer
-----------------------------------------------

.. code-block:: python

   from sklearn.preprocessing import QuantileTransformer
   scaler_df = QuantileTransformer(output_distribution='normal')

   X_df_train = scaler_df.fit_transform(load_X_df_train)
   X_df_valid = scaler_df.fit_transform(load_X_df_valid)
   X_df_test = scaler_df.fit_transform(load_X_df_test)

- ``QuantileTransformer`` transforms each feature to follow a normal distribution.
- ``fit_transform`` learns the transformation on the dataset and applies it.
- This normalizes skewed NF datasets and helps stabilize DNN training.


5. Import Keras for Deep Learning
---------------------------------

.. code-block:: python

   from keras.models import Sequential
   from keras.layers import Dense
   from keras.metrics import Recall, Precision

- ``Sequential`` defines a linear stack of layers.
- ``Dense`` creates fully connected neural network layers.
- ``Recall`` and ``Precision`` are used as extra evaluation metrics.


6. Define the DNN Model Function
--------------------------------

.. code-block:: python

   def fit_model(trainX, trainy):
       model = Sequential(name="N2_model")
       model.add(Dense(9, input_dim=39, activation='relu', name="N2_i"))
       model.add(Dense(7, activation='relu', name="N2_l1"))
       model.add(Dense(5, activation='relu', name="N2_l2"))
       model.add(Dense(3, activation='relu', name="N2_l3"))
       model.add(Dense(1, activation='sigmoid', name="N2_o")) 
       model._name="N2_m"

       model.compile(
           loss='binary_crossentropy',
           optimizer='adam',
           metrics=['accuracy', Recall(), Precision()]
       )
       
       model.fit(
           trainX, trainy,
           epochs=10,
           batch_size=1000,
           verbose=1,
           validation_data=(X_df_valid, Y_df_valid)
       )
       return model

Explanation:

- Defines a 5-layer DNN using ReLU activations.
- ``input_dim=39`` specifies number of input features.
- Final layer uses ``sigmoid`` for binary classification (attack vs benign).
- Compiles model with:
  - ``binary_crossentropy`` loss
  - ``adam`` optimizer
  - metrics: accuracy, recall, precision
- Trains for 10 epochs using 1000-sample batches.
- Validation data is provided for monitoring performance during training.


7. Train the Model
------------------

.. code-block:: python

   model = fit_model(X_df_train, Y_df_train)

- Calls the training function.
- Returns a trained DNN model.


8. Save the Trained Model
-------------------------

.. code-block:: python

   filename = './model_N2.keras'
   model.save(filename)
   print('>Saved %s' % filename)

- Saves the model in Keras' native format.
- Allows reloading for inference or evaluation.

Replace the filename with the correct model for other NF variants.


9. Load the Model for Evaluation
--------------------------------

.. code-block:: python

   from keras.models import load_model
   from numpy import argmax
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

   filename_model = './model_N2.keras'
   loaded_model = load_model(filename_model)

- ``load_model`` loads the saved network.
- Additional sklearn metrics are imported for evaluation.

Replace the filename with the correct model for other NF variants.


10. Evaluate Model on Test Set
-------------------------------

.. code-block:: python

   score = loaded_model.evaluate(X_df_test, Y_df_test, verbose=1)

   print('Test loss:', score[0])
   print('Test accuracy:', score[1])
   print('Test recall:', score[2])
   print('Test precision:', score[3])

- ``evaluate`` returns loss + all metrics defined during compile.
- Metrics returned:
  - ``score[0]`` → loss
  - ``score[1]`` → accuracy
  - ``score[2]`` → recall
  - ``score[3]`` → precision


11. Compute F1-Score Manually
------------------------------

.. code-block:: python

   precision = score[3]
   recall = score[2]
   f1_score = 2 * (precision * recall) / (precision + recall)

   print('Test F1-score:', f1_score)

- Calculates F1-score using the harmonic mean formula.
- Not included in model metrics, so computed manually.


Summary
-------

This script trains a DNN classifier on the NF-ToN-IoT-V2 dataset and evaluates
the model using accuracy, recall, precision, and F1-score. The workflow covers:

1. Loading dataset partitions
2. Feature/label separation
3. Quantile normalization
4. DNN model definition
5. Training and saving
6. Evaluation on test data
