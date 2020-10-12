import matplotlib.pyplot as plt

def plot_cm(cm):
    '''
    Plot confusion matrix
    '''
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = (0, 1)
    classes = ('spoof', 'real')
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    thresh = cm.max() / 2
    for i, j in zip((0, 0, 1, 1),(0,1,0,1)):
        plt.text(j, i, cm[i,j], horizontalalignment='center',
                color = 'white' if cm[i,j] > thresh else 'black')
