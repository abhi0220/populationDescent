import matplotlib.pyplot as plt
import numpy as np
data_dict = {
    "FMNIST Without Regularization":
    {
        "Basic Grid Search": [(0.251, 0.010), (0.037, 0.006), 64000],
        "KT RandomSearch": [(0.277, 0.023), (0.112, 0.034), 46800],
        "ESGD": [(0.276, 0.009), (0.114, 0.007), 46800],
        "Population Descent": [(0.249, 0.020), (0.124, 0.052), 32000],
    },
    "FMNIST With Regularization":
    {
        "Basic Grid Search": [(0.309, 0.009), (0.251, 0.007), 160000],
        "KT RandomSearch": [(0.400, 0.061), (0.295, 0.077), 46800],
        "Population Descent": [(0.262, 0.019), (0.152, 0.033), 32000],
    },
    "CIFAR-10 Without Regularization":
    {
        "Basic Grid Search": [(1.176, 0.182), (1.052, 0.250), 19200],
        "KT RandomSearch": [(1.512, 0.275), (1.343, 0.296), 39000],
        "ESGD": [(0.998, 0.025), (0.966, 0.033), 93750],
        "Population Descent": [(0.863, 0.014), (0.577, 0.060), 25600],
    },
    "CIFAR-10 With Regularization":
    {
        "Basic Grid Search": [(0.970, 0.027), (0.770, 0.043), 96000],
        "KT RandomSearch": [(1.195, 0.209), (1.030, 0.249), 39000],
        "Population Descent": [(0.843, 0.030), (0.555, 0.070), 25600],
    },
    "CIFAR-100 Without Regularization":
    {
        "Basic Grid Search": [(3.433, 0.050), (3.304, 0.041), 32000],
        "KT RandomSearch": [(4.129, 0.601), (4.004, 0.617), 39000],
        "ESGD": [(2.876, 0.146), (2.735, 0.157), 156250],
        "Population Descent": [(2.555, 0.093), (2.224, 0.193), 32000],
    },
    "CIFAR-100 With Regularization":
    {
        "Basic Grid Search": [(2.598, 0.061), (2.224, 0.079), 16000],
        "KT RandomSearch": [(1.195, 0.209), (1.030, 0.249), 39000],
        "Population Descent": [(2.584, 0.109), (2.236, 0.193), 32000],
    },
}

colors = {
    "Basic Grid Search": plt.cm.tab10(0),
    "KT RandomSearch": plt.cm.tab10(1),
    "ESGD": plt.cm.tab10(2),
    "Population Descent": plt.cm.tab10(3),
}

hatches = {
    "Basic Grid Search": "",
    "KT RandomSearch": "x",
    "ESGD": "/",
    "Population Descent": "o",
}

legends = []

def bar_reg_vs_noreg(dataset, title, indexer):
    '''
    Show bars for each method in a dataset with 4 categories (no reg test, reg test, no reg steps, reg steps)
    '''
    fig, ax = plt.subplots(figsize=(3, 3), tight_layout=True)
    # x_labels = ["No Reg Test", "Reg Test", "No Reg Steps", "Reg Steps"]
    def sub_bar(i, x_label, test):
        for j, method in enumerate(data_dict[test]):
            indexed = indexer(data_dict[test][method])
            y = indexed[0]
            bar_width = 0.2
            yerr = None if len(indexed) == 1 else indexed[1]
            # give the bars some patterns
            legends.append(ax.bar(i + j * bar_width, y, bar_width, yerr=yerr, color=colors[method], label=method, hatch = hatches[method]))
            
    sub_bar(0, "Unregularized", dataset + " Without Regularization")
    sub_bar(1, "Regularized", dataset + " With Regularization")
    # legend
    # ax.legend(data_dict[dataset + " Without Regularization"].keys())
    
    plt.title(title)
    plt.xticks( [0.3, 1.3], ["Unregularized", "Regularized"])
    plt.savefig("plots/" + dataset + "_" + title + ".pdf")

# scatter_reg_vs_noreg("FMNIST")
for dataset in ["FMNIST", "CIFAR-10", "CIFAR-100"]:
    bar_reg_vs_noreg(dataset, "Test Loss", lambda x: (x[0][0], x[0][1]))
    bar_reg_vs_noreg(dataset, "Gradient Steps", lambda x: (x[2], None))

# legend
fig_legend = plt.figure(figsize=(2, 1))
fig_legend.legend(handles=legends[0:4], loc='center', ncol=1)
fig_legend.tight_layout()
fig_legend.savefig("plots/legend.pdf")