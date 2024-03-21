import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def predict_and_print_scores(model, X_train, y_train, X_test, y_test, matrix=False, training=True, test=True):
    
    # Prediction
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    y_pred_train_p = model.predict_proba(X_train)[:,1]
    y_pred_test_p = model.predict_proba(X_test)[:,1]

    if training:
        print("Accuracy on training set:", round(accuracy_score(y_train, y_pred_train), 2))
    if test:
        print("Accuracy on test set:", round(accuracy_score(y_test, y_pred_test), 2))
    print("--------"*10)
    if training:
        print("Recall on training set:", round(recall_score(y_train, y_pred_train), 2))
    if test:
        print("Recall on test set:", round(recall_score(y_test, y_pred_test), 2))
    print("--------"*10)
    if training:
        print("Precision on training set:", round(precision_score(y_train, y_pred_train), 2))
    if test:
        print("Precision on test set:", round(precision_score(y_test, y_pred_test), 2))
    print("--------"*10)
       
    if training:
        print('roc_auc_score on trainig set: ', round(roc_auc_score(y_train, y_pred_train_p), 2))
    if test:
        print('roc_auc_score on test set: ', round(roc_auc_score(y_test, y_pred_test_p), 2))
    print("--------"*10)
    
    # Print confusion matrix
    if matrix == True:
        fig,ax = plt.subplots(figsize=(3,2))
        sns.heatmap(confusion_matrix(y_test, y_pred_test), annot=True, cmap='YlGn');
        plt.title('On the test set')

def train_crossval_predict_score(
    model,
    hyperparams,    
    X_train,
    y_train,
    X_test,
    y_test,
    cv=5,
    scoring='accuracy',
    verbose=0,
    n_jobs=-1,
    matrix=False,
    training=True,
    test=True,
    grid_search=True):
    
    # Cross-validation
    if grid_search:
        grid_model = GridSearchCV(model, param_grid=hyperparams, cv=cv, scoring=scoring, verbose=verbose, n_jobs=n_jobs)
    else:        
        grid_model = RandomizedSearchCV(model, param_distributions=hyperparams, cv=cv, scoring=scoring, verbose=verbose, n_jobs=n_jobs)
    
    # Fit
    grid_model.fit(X_train, y_train)
    
    best_model = grid_model.best_estimator_
    
    print('Best params:', grid_model.best_params_)    
    print("--------"*10)
    
    # Predict and print results
    predict_and_print_scores(best_model, X_train, y_train, X_test, y_test, matrix=matrix, training=training, test=test)
    
    return best_model


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    # Confusion matrix
    #cm = confusion_matrix(test_labels, rf_predictions)
    #plot_confusion_matrix(cm, classes = ['Poor Health', 'Good Health'],
    #                      title = 'Health Confusion Matrix')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    # Plot the confusion matrix
    plt.figure(figsize = (10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size = 14)
    plt.yticks(tick_marks, classes, size = 14)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    # Labeling the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize = 20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', size = 18)
    plt.xlabel('Predicted label', size = 18)


def plot_correlation(df, method='pearson', figsize=(10,10)):
    correlations = df.corr(method=method)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()    
    sns.heatmap(correlations , vmax=1, vmin=-1, annot=True, cmap="YlGnBu")
    plt.show()


def plot_distributions(df):
    
    num_features = list(df.columns[df.dtypes!=object])
    #num_features.remove('outcome')

    for feature in num_features:
        fig,axes=plt.subplots(1,2)
        sns.boxplot(data=df, x=feature, ax=axes[0])
        sns.histplot(data=df, x=feature, ax=axes[1], color='#ff4125', kde=True)
        fig.set_size_inches(15, 5)
        plt.suptitle(feature)  # Adds a title to the entire figure
        plt.show()

def plot_roc_curves(model_dic, X_test, y_test, figsize=(6,5)):

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()

    for key, _ in model_dic.items():

        model = model_dic[key][0]

        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:,1])
        plt.plot(fpr, tpr, model_dic[key][1], label=key)

    plt.plot([0,1],[0,1],'k:',label='random')
    plt.plot([0,0,1,1],[0,1,1,1],'k--',label='perfect')
    plt.xlabel('False Positive Rate (1 - Specifity)')
    plt.ylabel('True Positive Rate (Recall)')
    
