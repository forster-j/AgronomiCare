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
    
    return plt
        

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap='crest'):
    """
    This function calculates and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure(figsize=(14, 14))  # Increase figure size to accommodate more classes
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size=24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, size=10)  # Rotate labels for better readability
    plt.yticks(tick_marks, classes, size=10)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    # Labeling the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize=8,  # Decrease font size for better fit
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', size=18)
    plt.xlabel('Predicted label', size=18)

    return plt

def plot_training_metrics(history):
    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()