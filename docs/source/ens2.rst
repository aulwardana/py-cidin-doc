Ensemble Stacking Process - Scenario 2 
=====

Scenario 2 evaluates an *integrated stacking architecture*, where each domain
(N1–N5) trains an ensemble using **only the other four pre-trained models**.
This means:

- For N1 ensemble → use N2, N3, N4, N5  
- For N2 ensemble → use N1, N3, N4, N5  
- For N3 ensemble → use N1, N2, N4, N5  
- For N4 ensemble → use N1, N2, N3, N5  
- For N5 ensemble → use N1, N2, N3, N4  

This creates **five integrated networks**, each testing how well the knowledge
from other domains improves detection on the remaining domain.

The process:
- Load pre-trained N1–N5 networks.
- Remove one model depending on the target dataset.
- Freeze the remaining four models.
- Concatenate their outputs.
- Train a small meta-learner on the *sample dataset* of the target domain.
- Evaluate on the full test set of that domain.

Code
----

.. code-block:: python

    import pandas as pd
    import numpy as np
    from numpy import argmax
    from IPython.display import Image, display
    from sklearn.preprocessing import QuantileTransformer
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from keras.models import Sequential, load_model, Model
    from keras.layers import Dense, Input, concatenate
    from keras.metrics import Recall, Precision
    from keras.utils import plot_model
    import tensorflow as tf

    # ------------------------------------------------------------
    # Define stacked model for integrated networks
    # ------------------------------------------------------------
    def define_stacked_model(members):
        for i in range(len(members)):
            model = members[i]
            for layer in model.layers:
                layer.trainable = False
                layer._name = 'ensemble_' + str(i+1) + '_' + layer.name

        ensemble_visible = [model.input for model in members]
        ensemble_outputs = [model.output for model in members]

        merge = concatenate(ensemble_outputs)
        hidden = Dense(10, activation='relu')(merge)
        output = Dense(1, activation='sigmoid')(hidden)

        model = Model(inputs=ensemble_visible, outputs=output)
        model.compile(loss='binary_crossentropy', optimizer='adam',
                      metrics=['accuracy', Recall(), Precision()])
        return model

    # ------------------------------------------------------------
    # Train / Evaluate
    # ------------------------------------------------------------
    def fit_stacked_model(model, inputX, inputy):
        X = [inputX for _ in range(len(model.input))]
        model.fit(X, inputy, epochs=3, batch_size=256, verbose=1)

    def evaluate_stacked_model(model, inputX, inputy):
        X = [inputX for _ in range(len(model.input))]
        return model.evaluate(X, inputy, verbose=1)

    # Convert loaded Keras models into functional form
    def make_functional(model):
        inp = tf.keras.Input(shape=(39,))
        out = model(inp)
        return tf.keras.Model(inp, out)

    # ------------------------------------------------------------
    # Load Pre-trained Base Models (N1–N5)
    # ------------------------------------------------------------
    loaded_model_N1 = load_model('../Train/model_N1.keras')
    loaded_model_N2 = load_model('../Train/model_N2.keras')
    loaded_model_N3 = load_model('../Train/model_N3.keras')
    loaded_model_N4 = load_model('../Train/model_N4.keras')
    loaded_model_N5 = load_model('../Train/model_N5.keras')

    # ------------------------------------------------------------
    # Scenario 2 – Integrated Networks (leave-one-out ensemble)
    # ------------------------------------------------------------

    # ---------- N1 TARGET: use models N2–N5 ----------
    all_models_N1 = [
        make_functional(loaded_model_N2),
        make_functional(loaded_model_N3),
        make_functional(loaded_model_N4),
        make_functional(loaded_model_N5),
    ]

    stacked_model_N1 = define_stacked_model(all_models_N1)
    plot_model(stacked_model_N1, to_file="stacked_model_sec2_N1.png",
               show_shapes=True, show_layer_names=True, expand_nested=True)
    display(Image(filename="stacked_model_sec2_N1.png"))

    # Load UNSW-NB15
    df1_test = pd.read_parquet("../Dataset/NF-UNSW-NB15/NF-UNSW-NB15-V2_test.parquet")
    df1_sample = pd.read_parquet("../Dataset/NF-UNSW-NB15/NF-UNSW-NB15-V2_sample.parquet")
    df1_valid = pd.read_parquet("../Dataset/NF-UNSW-NB15/NF-UNSW-NB15-V2_valid.parquet")

    X_df1_sample = df1_sample.drop(["Label"], axis=1)
    Y_df1_sample = df1_sample["Label"]

    X_df1_test = df1_test.drop(["Label"], axis=1)
    Y_df1_test = df1_test["Label"]

    scaler_df1 = QuantileTransformer(output_distribution='normal')
    X_df1_sample = scaler_df1.fit_transform(X_df1_sample)
    X_df1_test = scaler_df1.fit_transform(X_df1_test)

    fit_stacked_model(stacked_model_N1, X_df1_sample, Y_df1_sample)
    score_stack_N1 = evaluate_stacked_model(stacked_model_N1, X_df1_test, Y_df1_test)

    print("Test loss:", score_stack_N1[0])
    print("Test accuracy:", score_stack_N1[1])
    print("Test recall:", score_stack_N1[2])
    print("Test precision:", score_stack_N1[3])

    # ---------- N2 TARGET: use models N1, N3, N4, N5 ----------
    all_models_N2 = [
        make_functional(loaded_model_N1),
        make_functional(loaded_model_N3),
        make_functional(loaded_model_N4),
        make_functional(loaded_model_N5),
    ]

    stacked_model_N2 = define_stacked_model(all_models_N2)
    plot_model(stacked_model_N2, to_file="stacked_model_sec2_N2.png",
               show_shapes=True, show_layer_names=True, expand_nested=True)
    display(Image(filename="stacked_model_sec2_N2.png"))

    # Load ToN-IoT
    df2_test = pd.read_parquet("../Dataset/NF-ToN-IoT/NF-ToN-IoT-V2_test.parquet")
    df2_sample = pd.read_parquet("../Dataset/NF-ToN-IoT/NF-ToN-IoT-V2_sample.parquet")

    X_df2_sample = df2_sample.drop(["Label"], axis=1)
    Y_df2_sample = df2_sample["Label"]

    X_df2_test = df2_test.drop(["Label"], axis=1)
    Y_df2_test = df2_test["Label"]

    scaler_df2 = QuantileTransformer(output_distribution='normal')
    X_df2_sample = scaler_df2.fit_transform(X_df2_sample)
    X_df2_test = scaler_df2.fit_transform(X_df2_test)

    fit_stacked_model(stacked_model_N2, X_df2_sample, Y_df2_sample)
    score_stack_N2 = evaluate_stacked_model(stacked_model_N2, X_df2_test, Y_df2_test)

    print("Test loss:", score_stack_N2[0])
    print("Test accuracy:", score_stack_N2[1])
    print("Test recall:", score_stack_N2[2])
    print("Test precision:", score_stack_N2[3])

    # ---------- N3 TARGET: use models N1, N2, N4, N5 ----------
    all_models_N3 = [
        make_functional(loaded_model_N1),
        make_functional(loaded_model_N2),
        make_functional(loaded_model_N4),
        make_functional(loaded_model_N5),
    ]

    stacked_model_N3 = define_stacked_model(all_models_N3)
    plot_model(stacked_model_N3, to_file="stacked_model_sec2_N3.png",
               show_shapes=True, show_layer_names=True, expand_nested=True)
    display(Image(filename="stacked_model_sec2_N3.png"))

    # Load BoT-IoT
    df3_test = pd.read_parquet("../Dataset/NF-BoT-IoT/NF-BoT-IoT-V2_test.parquet")
    df3_sample = pd.read_parquet("../Dataset/NF-BoT-IoT/NF-BoT-IoT-V2_sample.parquet")

    X_df3_sample = df3_sample.drop(["Label"], axis=1)
    Y_df3_sample = df3_sample["Label"]

    X_df3_test = df3_test.drop(["Label"], axis=1)
    Y_df3_test = df3_test["Label"]

    scaler_df3 = QuantileTransformer(output_distribution='normal')
    X_df3_sample = scaler_df3.fit_transform(X_df3_sample)
    X_df3_test = scaler_df3.fit_transform(X_df3_test)

    fit_stacked_model(stacked_model_N3, X_df3_sample, Y_df3_sample)
    score_stack_N3 = evaluate_stacked_model(stacked_model_N3, X_df3_test, Y_df3_test)

    print("Test loss:", score_stack_N3[0])
    print("Test accuracy:", score_stack_N3[1])
    print("Test recall:", score_stack_N3[2])
    print("Test precision:", score_stack_N3[3])

    # ---------- N4 TARGET: use models N1, N2, N3, N5 ----------
    all_models_N4 = [
        make_functional(loaded_model_N1),
        make_functional(loaded_model_N2),
        make_functional(loaded_model_N3),
        make_functional(loaded_model_N5),
    ]

    stacked_model_N4 = define_stacked_model(all_models_N4)
    plot_model(stacked_model_N4, to_file="stacked_model_sec2_N4.png",
               show_shapes=True, show_layer_names=True, expand_nested=True)
    display(Image(filename="stacked_model_sec2_N4.png"))

    # Load CSE-CIC-IDS2018
    df4_test = pd.read_parquet("../Dataset/NF-CSE-CIC-IDS2018/NF-CSE-CIC-IDS2018-V2_test.parquet")
    df4_sample = pd.read_parquet("../Dataset/NF-CSE-CIC-IDS2018/NF-CSE-CIC-IDS2018-V2_sample.parquet")

    X_df4_sample = df4_sample.drop(["Label"], axis=1)
    Y_df4_sample = df4_sample["Label"]

    X_df4_test = df4_test.drop(["Label"], axis=1)
    Y_df4_test = df4_test["Label"]

    scaler_df4 = QuantileTransformer(output_distribution='normal')
    X_df4_sample = scaler_df4.fit_transform(X_df4_sample)
    X_df4_test = scaler_df4.fit_transform(X_df4_test)

    fit_stacked_model(stacked_model_N4, X_df4_sample, Y_df4_sample)
    score_stack_N4 = evaluate_stacked_model(stacked_model_N4, X_df4_test, Y_df4_test)

    print("Test loss:", score_stack_N4[0])
    print("Test accuracy:", score_stack_N4[1])
    print("Test recall:", score_stack_N4[2])
    print("Test precision:", score_stack_N4[3])

    # ---------- N5 TARGET: use models N1, N2, N3, N4 ----------
    all_models_N5 = [
        make_functional(loaded_model_N1),
        make_functional(loaded_model_N2),
        make_functional(loaded_model_N3),
        make_functional(loaded_model_N4),
    ]

    stacked_model_N5 = define_stacked_model(all_models_N5)
    plot_model(stacked_model_N5, to_file="stacked_model_sec2_N5.png",
               show_shapes=True, show_layer_names=True, expand_nested=True)
    display(Image(filename="stacked_model_sec2_N5.png"))

    # Load UQ-NIDS
    df5_test = pd.read_parquet("../Dataset/NF-UQ-NIDS/NF-UQ-NIDS-V2_test.parquet")
    df5_sample = pd.read_parquet("../Dataset/NF-UQ-NIDS/NF-UQ-NIDS-V2_sample.parquet")

    X_df5_sample = df5_sample.drop(["Label"], axis=1)
    Y_df5_sample = df5_sample["Label"]

    X_df5_test = df5_test.drop(["Label"], axis=1)
    Y_df5_test = df5_test["Label"]

    scaler_df5 = QuantileTransformer(output_distribution='normal')
    X_df5_sample = scaler_df5.fit_transform(X_df5_sample)
    X_df5_test = scaler_df5.fit_transform(X_df5_test)

    fit_stacked_model(stacked_model_N5, X_df5_sample, Y_df5_sample)
    score_stack_N5 = evaluate_stacked_model(stacked_model_N5, X_df5_test, Y_df5_test)

    print("Test loss:", score_stack_N5[0])
    print("Test accuracy:", score_stack_N5[1])
    print("Test recall:", score_stack_N5[2])
    print("Test precision:", score_stack_N5[3])

Explanation
----------

**Goal of Scenario 2:**  

Test how well a domain can be detected by using knowledge **only from other domains**, forming an *integrated network*.

**Key differences from Scenario 1:**

- Scenario 1 uses **all 5 models** for every ensemble.
- Scenario 2 uses **only 4 models**, excluding the model trained on the target domain.
- This is comparable to *leave-one-domain-out knowledge transfer*.

**What Scenario 2 reveals:**

- Cross-domain generalization  
- Robustness of shared attack patterns  
- Weaknesses of domain-specific detection  
- How much each domain depends on its own training data  

