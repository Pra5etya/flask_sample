from sklearn import datasets, model_selection, tree
import pandas as pd
import numpy as np

sample_data = datasets.load_iris()
model_train_and_test = model_selection.train_test_split
model_decission_tree = tree.DecisionTreeClassifier()

# look toys data keys
print(sample_data.keys(), '\n')

# Model train and test
# =======================

# sample data
x = pd.DataFrame(sample_data['data'], columns = sample_data['feature_names'])

# sample target
y = pd.DataFrame(sample_data['target'], columns = ['Target'])

x_train, x_test, y_train, y_test = model_train_and_test(x, y, test_size = 0.3, random_state = 1)

# Model decission tree
# =======================

def dec_tree():
    # for implementation must same name in function
    dec_tree = model_decission_tree.fit(x_train, y_train)
    return dec_tree