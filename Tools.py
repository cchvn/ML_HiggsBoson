from helpers import *

def build_model_data(y, x):
    """Form (y,tX) to get regression data in matrix form."""
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx


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

#def launch_predictions (degree_poly):
    Y_train, X_train, Id_train = load_csv_train("train.csv", False)
    X_test, Id_test = load_csv_test("test.csv", False)
    X_train,X_test = X_preprocessing(X_train, X_test, degree_poly) 
    np.savetxt("X_train_preprocessed2.csv", X_train, delimiter=",")
    np.savetxt("X_test_preprocessed2.csv", X_test, delimiter=",")

    Pipeline_SVM(X_train, Y_train,X_test,  Id_test, name ="submission_SVM.csv")
    return None
