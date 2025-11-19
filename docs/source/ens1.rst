Ensemble Stacking Process - Scenario 1 
=====

This experiment builds a *stacked ensemble model* by combining five pre-trained
deep neural networks (N1–N5). Each model is frozen (non-trainable), then their
outputs are concatenated and passed through a small meta-learner (a dense
network). The meta-learner is trained on a small "sample" dataset from each
network’s corresponding domain.

The same stacking architecture is applied separately to:

- NF-UNSW-NB15
- NF-ToN-IoT
- NF-BoT-IoT
- NF-CSE-CIC-IDS2018
- NF-UQ-NIDS

The ensemble is retrained per dataset to test cross-domain generalization.

Code
----

.. code-block:: python

    import pandas as pd
    import numpy as np
    from numpy import argmax
    from IPython.display import Image, display
    from sklearn.preprocessing import QuantileTransformer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.models import load_model
    from keras.utils import plot_model
    from keras.models import Model
    from keras.layers import Input
    from keras.layers import concatenate
    from keras.metrics import Recall, Precision
    from keras.utils import plot_model
    import tensorflow as tf

    # ------------------------------------------------------------
    # create stacked model
    # ------------------------------------------------------------
    def define_stacked_model(members):
        for i in range(len(members)):
            model = members[i]
            # freeze layers and rename to avoid conflict
            for layer in model.layers:
                layer.trainable = False
                layer._name = 'ensemble_' + str(i+1) + '_' + layer.name

        ensemble_visible = [model.input for model in members]
        ensemble_outputs = [model.output for model in members]

        merge = concatenate(ensemble_outputs)
        hidden = Dense(10, activation='relu')(merge)
        output = Dense(1, activation='sigmoid')(hidden)

        model = Model(inputs=ensemble_visible, outputs=output)
        model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy', Recall(), Precision()]
        )
        return model

    # train stacked model
    def fit_stacked_model(model, inputX, inputy):
        X = [inputX for _ in range(len(model.input))]
        model.fit(X, inputy, epochs=3, batch_size=256, verbose=1)

    # evaluate stacked model
    def evaluate_stacked_model(model, inputX, inputy):
        X = [inputX for _ in range(len(model.input))]
        return model.evaluate(X, inputy, verbose=1)

    # convert a loaded keras model into a functional model
    def make_functional(model):
        inp = tf.keras.Input(shape=(39,))
        out = model(inp)
        func_model = tf.keras.Model(inp, out)
        return func_model

    # ------------------------------------------------------------
    # load base models N1–N5
    # ------------------------------------------------------------
    filename_model_N1 = '../Train/model_N1.keras'
    loaded_model_N1 = load_model(filename_model_N1)

    filename_model_N2 = '../Train/model_N2.keras'
    loaded_model_N2 = load_model(filename_model_N2)

    filename_model_N3 = '../Train/model_N3.keras'
    loaded_model_N3 = load_model(filename_model_N3)

    filename_model_N4 = '../Train/model_N4.keras'
    loaded_model_N4 = load_model(filename_model_N4)

    filename_model_N5 = '../Train/model_N5.keras'
    loaded_model_N5 = load_model(filename_model_N5)

    all_models = [
        make_functional(loaded_model_N1),
        make_functional(loaded_model_N2),
        make_functional(loaded_model_N3),
        make_functional(loaded_model_N4),
        make_functional(loaded_model_N5),
    ]

    # ------------------------------------------------------------
    # Scenario 1 – Apply stacking to each dataset (N1–N5)
    # ------------------------------------------------------------

    def run_scenario(dataset_prefix, image_name):
        stacked_model = define_stacked_model(all_models)

        plot_model(
            stacked_model,
            to_file=image_name,
            show_shapes=True,
            show_layer_names=True,
            expand_nested=True
        )
        display(Image(filename=image_name))

        # load dataset
        df_test = pd.read_parquet(f"../Dataset/{dataset_prefix}_test.parquet")
        df_sample = pd.read_parquet(f"../Dataset/{dataset_prefix}_sample.parquet")
        df_valid = pd.read_parquet(f"../Dataset/{dataset_prefix}_valid.parquet")

        X_sample = df_sample.drop(["Label"], axis=1)
        y_sample = df_sample["Label"]

        X_test = df_test.drop(["Label"], axis=1)
        y_test = df_test["Label"]

        X_valid = df_valid.drop(["Label"], axis=1)
        y_valid = df_valid["Label"]

        scaler = QuantileTransformer(output_distribution='normal')
        X_sample = scaler.fit_transform(X_sample)
        X_test = scaler.fit_transform(X_test)
        X_valid = scaler.fit_transform(X_valid)

        # train and evaluate
        fit_stacked_model(stacked_model, X_sample, y_sample)
        score = evaluate_stacked_model(stacked_model, X_test, y_test)

        print("Test loss:", score[0])
        print("Test accuracy:", score[1])
        print("Test recall:", score[2])
        print("Test precision:", score[3])

        return score

    # ------------------------------------------------------------
    # Run all experiments
    # ------------------------------------------------------------
    score_N1 = run_scenario("NF-UNSW-NB15/NF-UNSW-NB15-V2", "stacked_model_sec1_N1.png")
    score_N2 = run_scenario("NF-ToN-IoT/NF-ToN-IoT-V2", "stacked_model_sec1_N2.png")
    score_N3 = run_scenario("NF-BoT-IoT/NF-BoT-IoT-V2", "stacked_model_sec1_N3.png")
    score_N4 = run_scenario("NF-CSE-CIC-IDS2018/NF-CSE-CIC-IDS2018-V2", "stacked_model_sec1_N4.png")
    score_N5 = run_scenario("NF-UQ-NIDS/NF-UQ-NIDS-V2", "stacked_model_sec1_N5.png")

Explanation
----------

**1. Ensemble Construction**

- Five pre-trained DNNs (N1–N5) are loaded.
- Each model is frozen to preserve its learned knowledge.
- Outputs of all models are concatenated.
- A small DNN (10 → 1 neurons) learns how to combine the predictions.
- This is *stacking*, not bagging or boosting.

**2. Per-Dataset Training**

Each dataset contributes:

- A *sample set* → for training the ensemble meta-learner.
- A *test set* → to evaluate cross-domain performance.
- A *validation set* (not used here but available).

**3. Scaling**

`QuantileTransformer(output_distribution="normal")`  
Normalizes numeric features into a Gaussian distribution.

**4. Results**

The script prints:

- Loss
- Accuracy
- Recall
- Precision

F1-score could also be computed similarly.
