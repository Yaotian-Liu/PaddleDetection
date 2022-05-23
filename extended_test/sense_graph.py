import pandas as pd
import matplotlib.pyplot as plt

sensitivity = pd.read_csv("./extended_test/sensitivity.csv")

pruned_params = sensitivity['param'].unique()

fig, ax = plt.subplots()

x = sensitivity['ratio'].unique()

for param in pruned_params:
    single_param = sensitivity[sensitivity['param'] == param]
    y = single_param['sensitivity']

    ax.plot(x, y, label=param)
    print(single_param)

ax.set_xlabel("prune ratio")
ax.set_ylabel("sensitivity")
ax.legend()

plt.show()