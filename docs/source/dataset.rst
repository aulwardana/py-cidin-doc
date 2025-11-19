Dataset Processing
=====

This document explains, line by line, the entire data preparation workflow for all other NF datasets.  

- NF-UNSW-NB15-v2
- NF-UQ-NIDS-v2
- NF-ToN-IoT-v2
- NF-CSE-CIC-IDS2018-v2
- NF-BoT-IoT-v2

Only the input file changes.

---

1. Import Required Libraries
----------------------------

.. code-block:: python

   import pandas as pd
   import numpy as np
   from sklearn.model_selection import train_test_split

- ``pandas`` handles loading and manipulating NF dataset files.
- ``numpy`` provides numerical utilities.
- ``train_test_split`` is used to create dataset splits.

---

2. Load the Parquet Dataset
---------------------------

.. code-block:: python

   df = pd.read_parquet(
       '/path/to/NF-UNSW-NB15-V2.parquet',
       engine='pyarrow'
   )

   df.info()

- Loads the dataset using the efficient PyArrow engine.
- ``df.info()`` prints row count, columns, datatypes, and memory usage.

Replace the filename with the correct dataset for other NF variants.

---

3. Check Label Distribution
---------------------------

.. code-block:: python

   df['Label'].value_counts()

- Shows how many benign and attack samples exist.
- Confirms dataset balance and that the label column is intact.

---

4. Drop Identifier Columns
--------------------------

.. code-block:: python

   df = df.drop(df.columns[[0, 1, 42]], axis=1)

- Removes non-feature or metadata columns that are not used in modeling.
- These typically include index values, timestamps, or flow identifiers.

---

5. Verify Label Counts Again
----------------------------

.. code-block:: python

   df['Label'].value_counts()

- Ensures labels were not altered by the column removal process.

---

6. Split Into Training and Testing Sets
---------------------------------------

.. code-block:: python

   train, test = train_test_split(df, test_size=0.3, random_state=100)

- Splits dataset into:
  - 70% training
  - 30% testing
- ``random_state`` ensures reproducibility.

This structure is used for all NF datasets.

---

7. Check Training Label Distribution
------------------------------------

.. code-block:: python

   train['Label'].value_counts()

- Confirms class balance in the training subset.

---

8. Check Test Label Distribution
--------------------------------

.. code-block:: python

   test['Label'].value_counts()

- Ensures test set is representative.

---

9. Create a 10% Sample Subset
-----------------------------

.. code-block:: python

   train, sample = train_test_split(train, test_size=0.1, random_state=100)

- Generates a smaller dataset useful for fast prototyping or resource-limited experiments.

---

10. Check Remaining Training Labels
-----------------------------------

.. code-block:: python

   train['Label'].value_counts()

- Confirms class balance in the reduced training set.

---

11. Check Sample Set Labels
---------------------------

.. code-block:: python

   sample['Label'].value_counts()

- Ensures the 10% sample maintains representative class distribution.

---

12. Save the Training Set
-------------------------

.. code-block:: python

   train.to_parquet("./NF-UNSW-NB15-V2_train.parquet")

- Saves processed training data to a new Parquet file.

---

13. Save the Sample Set
-----------------------

.. code-block:: python

   sample.to_parquet("./NF-UNSW-NB15-V2_sample.parquet")

- Stores the small sample subset for quick experiments.

---

14. Split Test Set Into Test and Validation
-------------------------------------------

.. code-block:: python

   test, valid = train_test_split(test, test_size=0.1, random_state=100)

- Creates:
  - Final test set (90% of original test)
  - Validation set (10% of original test)
- Validation is used for tuning and model selection.

---

15. Check Final Test Distribution
---------------------------------

.. code-block:: python

   test['Label'].value_counts()

---

16. Check Validation Distribution
---------------------------------

.. code-block:: python

   valid['Label'].value_counts()

---

17. Save Final Test Set
-----------------------

.. code-block:: python

   test.to_parquet("./NF-UNSW-NB15-V2_test.parquet")

---

18. Save Validation Set
-----------------------

.. code-block:: python

   valid.to_parquet("./NF-UNSW-NB15-V2_valid.parquet")

---

19. Empty Cell
--------------

.. code-block:: python

   # (empty)

- No operations performed.

---

Summary
-------

The full processing workflow includes:

1. Loading Parquet dataset
2. Inspecting structure and cleaning features
3. Creating training, testing, validation, and sample splits
4. Saving all processed files

This exact workflow applies to all NF-family datasets, with only the file path adjusted.

