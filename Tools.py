from helpers import *
from Preprocess import *
from SVM import *

def load_csv_train(path_train, sub_sample=False):
    """Loads data and returns y (class labels 1 and -1), tX (features) and ids (event ids)"""
    y = np.genfromtxt(path_train, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(path_train, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(int)
    input_data = x[:, 2:] #first 2 columns are Id,Prediction
    ids = np.array([ids]).T
    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == "b")] = -1 
    yb = np.array([yb]).T
    # sub-sample
    if sub_sample:
        yb = yb[:5000,:]
        input_data = input_data[:5000,:]
        ids = ids[:5000,:]

    return yb, input_data, ids

def load_csv_test(path_test, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    x = np.genfromtxt(path_test, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(int)
    input_data = x[:, 2:]
    ids = np.array([ids]).T

    # sub-sample
    if sub_sample:
        input_data = input_data[:500,:]
        ids = ids[:500,:]

    return input_data, ids


def confusion_matrix(y,ypred):
    """ Evaluates the confusion matrix for classification 
    Args:
        y: shape=(N, 1) variable to be predicted
        ypred: shape=(N, 1) variable predicted by the model
    Returns:
         confusion matrix
    """
    fp = 0
    fn = 0
    tp = 0
    tn = 0
    n=len(y)
    for actual_value, predicted_value in zip(y, ypred):
        if predicted_value == actual_value:
            if predicted_value == 1:
                tp += 1
            else: 
                tn += 1
        else: 
            if predicted_value == 1:
                fp += 1
            else:
                fn += 1
                
    confusion_matrix = [[tn/n, fp/n], [fn/n, tp/n]]
    confusion_matrix = np.array(confusion_matrix)
    return confusion_matrix

def accuracy(y,ypred): 
    matrix = confusion_matrix(y,ypred)
    return matrix[0,0]+ matrix[1,1]


def export_to_csv(ypred,Id_test, text):
    output = np.concatenate((Id_test, ypred),axis = 1)
    np.savetxt(text, output, delimiter=",",fmt= "%i", header = "Id,Prediction")
    return None

def launch_predictions (degree_poly):
    Y_train, X_train, Id_train = load_csv_train("train.csv", False)
    X_test, Id_test = load_csv_test("test.csv", False)
    X_train,X_test = X_preprocessing(X_train, X_test, degree_poly) 
    np.savetxt("X_train_preprocessed2.csv", X_train, delimiter=",")
    np.savetxt("X_test_preprocessed2.csv", X_test, delimiter=",")

    Pipeline_SVM(X_train, Y_train,X_test,  Id_test, name ="submission_SVM.csv")
    return None
