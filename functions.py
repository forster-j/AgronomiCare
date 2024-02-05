
import itertools
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt

# define function for nice formating of the prediction metrics:
def summarize_pred_metrics(model, X_test, y_test, classes, normalize = False,
                            cmap = 'viridis'):
    """
    This function calculates the classification report and the confusion matrix and 
    subsequently prints/plots them in a nice format.
    Normalization can be applied to the confusion matrix by setting 'normalize=True'.
    classes - list containing the two class names

    Inputs:
    model: The trained classification model.
    X_test: Test set features.
    y_test: True labels for the test set.
    classes: List containing class names.
    normalize: Flag to normalize the confusion matrix (default is False).
    cmap: Colormap for the confusion matrix plot (default is 'viridis').
    thresh: Threshold for binary classification (default is 0.5).
    
    Outputs:
    Prints the classification report, ROC AUC score, and confusion matrix.
    Optionally plots a confusion matrix.
    """
    # calculate prediction and confusion matrix
    y_pred_proba = model.predict_proba(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)  # Choose the class with the highest probability


    cm = confusion_matrix(y_test, y_pred, labels=classes)

    # Print classification report
    print('---------'*7)
    print("Classification report:")
    print('---------'*7)
    print(classification_report(y_test,y_pred))
    print('---------'*7)
    print('ROC AUC score:',round(roc_auc_score(y_test,y_pred_proba),2))
    print('---------'*7)

    # calculate normalization of the confusion matrix
    if normalize:
        cm = cm.astype('float') /cm.sum(axis=1)[:,np.newaxis]
        print('Normalized confusion matrix')
        print('---------'*7)
    else:
        print('Confusion matrix, without normalization')
        print('---------'*7)

    # Plot the confusion matrix in a custom format using imshow and multicolored labeling 
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks,classes, rotation = 90, va ='center')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # parameters for the Labeling
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    # Labeling the plot
    for i, j, in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j, i, format(cm[i,j],fmt), horizontalalignment="center",
                 color = "white" if cm[i,j] > thresh else "black")