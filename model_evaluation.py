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
                             fbeta_score )

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def predict_and_print_scores(model,             #Trained model
                             X_train,           #Training data with features
                             y_train,           #Traning data with labels or targets
                             X_test,            #Testing data with features
                             y_test,            #Testind data with labels or targets                             
                             training=True,     #True: print scores on the traning set
                             test=True,         #True: print scores on the testing set
                             accuracy=True,     #True: print accuracy_score()
                             recall=True,       #True: print recall_score()
                             precision=True,    #True: print precision_score()
                             fbeta=[True, 1.0], #[True, beta]: print fbeta_score. If beta = 1.0: f1_score
                             roc_auc=True,      #True: print roc_auc_score()
                             matrix=True,       #True: plot confusion matrix
                             figsize=(3,2),     #Figure size for the confusion matrix
                             cmap='YlGn'):      #Color map for the confusion matrix
    
    '''
    Given an already trained model, this function predicts and print some performance scores training and/or testing data.
    The supported metrics are: accuracy, recall, precision, fbeta_score (and f1_score if beta = 1.0), roc_auc.
    If the input parameter "matrix" is set to True, the function plot the confusion matrix with a color map given in "cmap".
    Posible color maps: 'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                        'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                        'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'
    '''

    # Prediction
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
            
    # Scores
    if accuracy:
        if training:
            print("Accuracy on training set:", round(accuracy_score(y_train, y_pred_train), 2))
        if test:
            print("Accuracy on test set:", round(accuracy_score(y_test, y_pred_test), 2))
        print("--------"*5)
    
    if recall:        
        if training:
            print("Recall on training set:", round(recall_score(y_train, y_pred_train), 2))
        if test:
            print("Recall on test set:", round(recall_score(y_test, y_pred_test), 2))
        print("--------"*5)
    
    if precision:
        if training:
            print("Precision on training set:", round(precision_score(y_train, y_pred_train), 2))
        if test:
            print("Precision on test set:", round(precision_score(y_test, y_pred_test), 2))
        print("--------"*5)

    if fbeta_score:
        if training:
            print("fbeta_score on training set:", round(fbeta_score(y_train, y_pred_train, beta=fbeta[1]), 2))
        if test:
            print("fbeta_score on test set:", round(fbeta_score(y_test, y_pred_test, beta=fbeta[1]), 2))
        print("--------"*5)

    if roc_auc:
        y_pred_train_p = model.predict_proba(X_train)[:,1]
        y_pred_test_p = model.predict_proba(X_test)[:,1]
        if training:
            print('roc_auc_score on trainig set: ', round(roc_auc_score(y_train, y_pred_train_p), 2))
        if test:
            print('roc_auc_score on test set: ', round(roc_auc_score(y_test, y_pred_test_p), 2))
        print("--------"*5)
    
    # Plot confusion matrix
    if matrix:
        fig,ax = plt.subplots(figsize=figsize)
        sns.heatmap(confusion_matrix(y_test, y_pred_test), annot=True, cmap=cmap);
        plt.title('Test Set')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        return fig, ax



def train_crossval_predict_score(model,                 #Instantiated model
                                 hyperparams,           #Dictionary incluing hyperparameters
                                 X_train,               #Training data with features
                                 y_train,               #Traning data with labels or targets
                                 X_test,                #Testing data with features
                                 y_test,                #Testind data with labels or targets
                                 cv=5,                  #Number of cross-validdation folds
                                 scoring='accuracy',    #Scoring method
                                 verbose=0,             #Verbose
                                 n_jobs=-1,             #Number of jobs in parallel
                                 grid_search=True,      #True: Apply GridSearchCV. False: Apply RandomSearchCV                                 
                                 training=True,         #True: print scores on the traning set
                                 test=True,             #True: print scores on the testing set
                                 accuracy=True,         #True: print accuracy_score()
                                 recall=True,           #True: print recall_score()
                                 precision=True,        #True: print precision_score()
                                 fbeta=[True, 1.0],     #[True, beta]: print fbeta_score. If beta = 1.0: f1_score
                                 roc_auc=True,          #True: print roc_auc_score()
                                 matrix=True,           #True: plot confusion matrix
                                 figsize=(3,2),         #Figure size for the confusion matrix
                                 cmap='YlGn'):          #Color map for the confusion matrix
                                 
    
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
    predict_and_print_scores(best_model,
                             X_train,
                             y_train,
                             X_test,
                             y_test,
                             training=training,
                             test=test,
                             accuracy=accuracy,
                             recall=recall,
                             precision=precision,
                             fbeta=fbeta,
                             roc_auc=roc_auc,                             
                             matrix=matrix,
                             figsize=figsize,
                             cmap=cmap)
    
    return best_model


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges,
                          figsize=(10,10)):
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
    plt.figure(figsize = figsize)
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

    return fig, ax


def plot_distributions(df):
    
    num_features = list(df.columns[df.dtypes!=object])
    #num_features.remove('outcome')

    for feature in num_features:
        fig,ax=plt.subplots(1,2)
        sns.boxplot(data=df, x=feature, ax=ax[0])
        sns.histplot(data=df, x=feature, ax=ax[1], color='#ff4125', kde=True)
        fig.set_size_inches(15, 5)
        plt.suptitle(feature)  # Adds a title to the entire figure
        plt.show()
    
    return fig, ax

def plot_roc_curves(model_dic, X_test, y_test, figsize=(6,5)):

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()

    for key, _ in model_dic.items():

        model = model_dic[key][0]

        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:,1])
        plt.plot(fpr, tpr, model_dic[key][1], label=key)

    plt.plot([0,1],[0,1],'k:',label='random')
    plt.plot([0,0,1,1],[0,1,1,1],'k--',label='perfect')
    ax.set_xlabel('False Positive Rate (1 - Specifity)')
    ax.set_ylabel('True Positive Rate (Recall)')
    ax.legend()
    plt.show()

    return fig, ax