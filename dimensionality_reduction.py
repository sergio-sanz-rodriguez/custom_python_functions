import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

from sklearn.metrics import ( confusion_matrix,
                             accuracy_score,
                             recall_score,
                             precision_score,
                             roc_curve,
                             roc_auc_score,
                             fbeta_score,
                             f1_score)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def plot_pca(explained_variance_ratio):

    """
    This function plots the explained_variance_ratio attribute of a trained PCA object.
    The input of this function is the result of applying this command: pca.explained_variance_ratio_
    where pca is the trained PCA object (pca.train(X))
    Example if usage:
    pca = PCA(n_components=None)
    pca.fit(X_train)
    plot_pca(pca.explained_variance_ratio_)
    """
    
    plt.figure(figsize=(10,6))
    plt.scatter(x=[i+1 for i in range(len(explained_variance_ratio))], y=explained_variance_ratio, s=200, alpha=0.75,c='orange',edgecolor='k')
    plt.grid(True)
    plt.title("Explained variance ratio of the \nfitted principal component vector\n",fontsize=25)
    plt.xlabel("Principal components",fontsize=15)
    #plt.xticks([i+1 for i in range(len(principalComponents.explained_variance_ratio_))],fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel("Explained variance ratio",fontsize=15)
    plt.show()