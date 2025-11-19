Cross-esting Sample
=====

This document describes the Deep Neural Network (DNN) training and 
cross-testing process for all NF datasets, including:

* NF-UNSW-NB15-V2
* NF-ToN-IoT-V2
* NF-BoT-IoT-V2
* NF-CSE-CIC-IDS2018-V2
* NF-UQ-NIDS-V2

The workflow consists of:

1. Dataset loading and label separation  
2. Preprocessing using ``QuantileTransformer``  
3. Building a DNN classifier  
4. Training with validation data  
5. Saving and evaluating the model  
6. Cross-testing across five different datasets  
7. Plotting metrics for accuracy, recall, precision, and F1-score

---------------------------------------
1. Dataset Loading and Preprocessing
---------------------------------------

.. code-block:: python

    import pandas as pd
    import numpy as np

    df_train = pd.read_parquet("../Dataset/NF-ToN-IoT/NF-ToN-IoT-V2_train.parquet")
    df_valid = pd.read_parquet("../Dataset/NF-ToN-IoT/NF-ToN-IoT-V2_valid.parquet")
    df_test = pd.read_parquet("../Dataset/NF-ToN-IoT/NF-ToN-IoT-V2_test.parquet")

    load_X_df_train = df_train.drop(["Label"],axis=1)
    Y_df_train = df_train["Label"]

    load_X_df_valid = df_valid.drop(["Label"],axis=1)
    Y_df_valid = df_valid["Label"]

    load_X_df_test = df_test.drop(["Label"],axis=1)
    Y_df_test = df_test["Label"]


* ``pd.read_parquet`` loads the NF datasets.  
* The ``Label`` column is separated from feature columns.  
* ``load_X_df_*`` contains only numerical network-flow features.  
* ``Y_df_*`` contains binary labels (Normal/Attack).

Scaling
-------

.. code-block:: python

    from sklearn.preprocessing import QuantileTransformer

    scaler_df = QuantileTransformer(output_distribution='normal')

    X_df_train = scaler_df.fit_transform(load_X_df_train)
    X_df_valid = scaler_df.fit_transform(load_X_df_valid)
    X_df_test = scaler_df.fit_transform(load_X_df_test)


* ``QuantileTransformer`` converts the distribution of each feature to Gaussian.
* It is robust for data with heavy-tailed values (common in network traffic).
* A new scaler is fit for each split.

-------------------------------
2. Building the DNN Model
-------------------------------

.. code-block:: python

    from keras.models import Sequential
    from keras.layers import Dense
    from keras.metrics import Recall, Precision

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
            trainX,
            trainy,
            epochs=10,
            batch_size=1000,
            verbose=1,
            validation_data=(X_df_valid, Y_df_valid)
        )

        return model


* ``Sequential`` builds a feed-forward neural network.
* Layer sizes progressively shrink (9 → 7 → 5 → 3 → 1).
* ``sigmoid`` output is required for binary classification.
* Training runs for **10 epochs** with **batch size = 1000**.
* Validation data monitors overfitting.

-------------------------------
3. Training and Saving the Model
-------------------------------

.. code-block:: python

    model = fit_model(X_df_train, Y_df_train)

    filename = './model_N2.keras'
    model.save(filename)
    print('>Saved %s' % filename)


* The trained DNN is saved in native Keras format.
* Naming scheme follows network index (N1–N5).

-------------------------------
4. Model Evaluation
-------------------------------

.. code-block:: python

    from keras.models import load_model

    filename_model = './model_N2.keras'
    loaded_model = load_model(filename_model)

    score = loaded_model.evaluate(X_df_test, Y_df_test, verbose=1)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print('Test recall:', score[2])
    print('Test precision:', score[3])

    precision = score[3]
    recall = score[2]
    f1_score = 2 * (precision * recall) / (precision + recall)

    print('Test F1-score:', f1_score)


* ``evaluate`` returns loss, accuracy, recall, precision.
* F1 is computed manually using:

  ``F1 = 2 * (Precision * Recall) / (Precision + Recall)``

---------------------------------------
5. Cross-Testing Across NF Datasets
---------------------------------------

Five models (N1–N5) are tested across five datasets:

* UNSW-NB15  
* ToN-IoT  
* BoT-IoT  
* CSE-CIC-IDS2018  
* UQ-NIDS  

Dataset Loading

.. code-block:: python

    df_test_N1 = pd.read_parquet("../Dataset/NF-UNSW-NB15/NF-UNSW-NB15-V2_sample.parquet")
    df_test_N2 = pd.read_parquet("../Dataset/NF-ToN-IoT/NF-ToN-IoT-V2_test.parquet")
    df_test_N3 = pd.read_parquet("../Dataset/NF-BoT-IoT/NF-BoT-IoT-V2_test.parquet")
    df_test_N4 = pd.read_parquet("../Dataset/NF-CSE-CIC-IDS2018/NF-CSE-CIC-IDS2018-V2_test.parquet")
    df_test_N5 = pd.read_parquet("../Dataset/NF-UQ-NIDS/NF-UQ-NIDS-V2_test.parquet")

* Each dataset is scaled using ``QuantileTransformer``.
* ``loaded_model_Nx`` refers to the trained DNN for network ``Nx``.

Example: Evaluating Model N1 on Dataset N1

.. code-block:: python

    score_N1_test_N1 = loaded_model_N1.evaluate(X_df_test_N1, Y_df_test_N1, verbose=1)

    print('Test loss:', score_N1_test_N1[0])
    print('Test accuracy:', score_N1_test_N1[1])
    print('Test recall:', score_N1_test_N1[2])
    print('Test precision:', score_N1_test_N1[3])

    f1_score_N1_test_N1 = 2 * (
        score_N1_test_N1[3] * score_N1_test_N1[2]
    ) / (score_N1_test_N1[3] + score_N1_test_N1[2])

    print('Test F1-score:', f1_score_N1_test_N1)

The same evaluation is repeated for:

* N1 tested on N2–N5  
* N2 tested on N1–N5  
* N3 tested on N1–N5  
* N4 tested on N1–N5  
* N5 tested on N1–N5  

-------------------------------
6. Metric Collection
-------------------------------

Metrics for accuracy, recall, precision, and F1-score are aggregated:

.. code-block:: python

    accuracy_N1 = [score_N1_test_N1[1], score_N1_test_N2[1], ...]
    recall_N1   = [score_N1_test_N1[2], score_N1_test_N2[2], ...]
    precision_N1= [score_N1_test_N1[3], score_N1_test_N2[3], ...]

    f1_score_N1 = [
        2*(precision_N1[i]*recall_N1[i])/(precision_N1[i]+recall_N1[i])
        for i in range(5)
    ]

---------------------------------------
7. Metric Visualization
---------------------------------------

.. code-block:: python

    plot_bar_with_values(accuracy_N1, labels, 'N1 Accuracy on Test Sets (N1–N5)', 'skyblue')
    plot_bar_with_values(recall_N1,   labels, 'N1 Recall on Test Sets (N1–N5)', 'lightgreen')
    plot_bar_with_values(precision_N1,labels, 'N1 Precision on Test Sets (N1–N5)', 'salmon')
    plot_bar_with_values(f1_score_N1, labels, 'N1 F1-score on Test Sets (N1–N5)', 'plum')


Each metric is visualized with:

* X-axis → NF dataset index (N1–N5)
* Y-axis → Score in percentage  
* Values are annotated above each bar  
* Useful for comparing how well a model generalizes across networks

---------------------------------------
Summary
---------------------------------------

This document provides:

* Complete DNN training workflow  
* Preprocessing with ``QuantileTransformer``  
* Saving/loading Keras models  
* Cross-testing against five NF datasets  
* Metric aggregation and visualization  

This structure is consistent for all NF datasets and all DNN models.
