import pandas as pd
import matplotlib.pyplot as plt

sensitivity = pd.read_csv("./extended_test/sensitivity.csv")

pruned_params = sensitivity['param'].unique()

for param in pruned_params:
    single_param = sensitivity[sensitivity['param'] == param]
    single_param = single_param.set_index(keys=['ratio'])

    print(single_param)
