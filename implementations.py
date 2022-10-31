import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import csv
from numpy import linalg as LA
from helpers import *
# MSE 

def compute_mse_loss(y, tx, w):

    """Calculate the loss using either MSE or MAE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute loss by MSE
    # MSE : 
    
    y_pred = tx@w
    x = (y_pred-y)**2
    MSE = 1/2*x.mean()
    return MSE

def compute_mse_gradient(y, tx, w):
    """Computes the gradient at w.
        
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.
        
    Returns:
        An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    y_pred = tx.dot(w)
    error = y - y_pred
    n = len(error)
    dloss = -2*tx.T.dot(error)/n
    return dloss

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.
        
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
        
    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of GD 
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        loss = compute_mse_loss(y,tx,w)
        grad = compute_mse_gradient(y,tx,w)
        w = w - gamma*grad        
        # store w and loss
        #ws.append(w)
        #losses.append(loss)
        print("GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
             bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return w, loss

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).
            
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize
        
    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD 
    """
    
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    batch_size = 1
    
    for n_iter in range(max_iters):
        index = np.random.randint(0, len(tx), batch_size) # random sample
        loss = compute_mse_loss(y[index,],tx[index,],w)
        grad = compute_mse_gradient(y[index,], tx[index,], w)
        w = w - gamma*grad

        print("SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
              
        ws.append(w)
        losses.append(loss)
    return w, loss

def least_squares(y, tx):
    
    """
    compute the least square regression using normal equation
    inputs:
        y: shape=(N, 1)
        tx: shape=(N,M)
    outputs:
        w: shape=(M,1)
        MSE: float 
    """
    left = tx.T@tx
    right = tx.T@y
    w,_,_,_ = np.linalg.lstsq(left,right)
    loss = compute_mse_loss(y,tx,w)
    return w, loss

def ridge_regression(y, tx, lambda_):
    """implement ridge regression.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: compute with MSE loss 
    """
    #compute lambda_prime
    lp = lambda_*2*len(y)
    
    #the linear system we need to solve is: X.T@X + lp*I = X.T@y
    left = tx.T@tx + lp*np.identity(tx.shape[1])
    right = tx.T@y
    w,_,_,_ = np.linalg.lstsq(left,right)
    
    loss = compute_mse_loss(y,tx,w)

    return w, loss

#1) Régression logistique

#1.1) Transformer les y en dans l'intervale [0,1]

def log_transform (y):
    """ 
    Transforme les valeurs de classification dans l'intervale [0,1] pour pourvoir être utilisés dans la régression logistique.
    """
    y = (y+1)/2
    return y

def sigmoid(x):
    """Computes the sigmoid function for a vector x"""
    return 1 / (1 + np.exp(-x))

def compute_logistic_cost(tx, y, w):
    
    """Computes the logistic loss at w.
    Args:
        y: shape=(N, 1)
        tx: shape=(N,M)
        w: shape=(M, 1). The vector of model parameters.
    Returns:
        Value of cost function at w  
    """
            
    m = len(y)
    h = sigmoid(tx @ w) #vérifier les dimensions des vecteurs
    epsilon = 1e-5
    cost = (1/m)*(((-y).T @ np.log(h + epsilon))-((1-y).T @ np.log(1-h + epsilon)))
    return cost

def gradient_descent_log(tx, y, initial_w, gamma, max_iters):
    
    """Computes the optimal model parameters w for the logistic model
    Args:
        y: shape=(N, 1)
        tx: shape=(N,M)
        w: shape=(M, 1). The vector of model parameters.
    Returns:
        Value of cost function at w  
    """
    
    m = len(y)
    w = initial_w

    for i in range(max_iters):
        w = w - (gamma/m) * (tx.T @ (sigmoid(tx @ w) - y)) 
        
    cost = compute_logistic_cost(tx, y, w)
    #return (cost[0], w)
    return w, cost

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    #logistic_regression(tx,y,initial_w, gamma, max_iters):
    """Computes the optimal model parameters w for the logistic model and the loss function 
    Args:
        y: shape=(N, 1)
        tx: shape=(N,M)
        initial_w: shape=(M, 1). The vector of model parameters.
        max_iters : number of iteration desired
        gamma: parameter for gradient descent
        
    Returns:
         (w,loss) at final iteration
    
    """

    return gradient_descent_log(tx, y, initial_w, gamma, max_iters)


def gradient_descent_log_ridge(tx, y,lambda_ridge, initial_w, max_iters,gamma):
    
    """Computes the optimal model parameters w with Ridge regularization for the logistic model 
    Args:
        y: shape=(N, 1)
        tx: shape=(N,M)
        w: shape=(M, 1). The vector of model parameters.
        lambda_ : Ridge regularization parameter
    Returns:
        Value of cost function at w  
    """
    m = len(y)
    w = initial_w

    for i in range(max_iters):
        w = w - (gamma/m) * (tx.T @ (sigmoid(tx @ w) - y) + w * lambda_ridge ) 
        
    cost = compute_logistic_cost(tx, y, w)
    return w, cost


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    #Returns (w,loss) at final itteration
    #Loss does not include the penalty term
    
    """Computes the optimal model parameters w for the logistic model xith ridge
    regression and the loss function 
    Args:
        y: shape=(N, 1)
        tx: shape=(N,M)
        initial_w: shape=(M, 1). The vector of model parameters.
        max_iters : number of iteration desired
        gamma: parameter for gradient descent
        lambda_ : Ridge regularisation parameter
        
    Returns:
         (w,loss) at final iteration
         Loss does not include the penalty term
    
    """    
    
    return gradient_descent_log_ridge(tx, y, lambda_, initial_w, max_iters,gamma)


def log_predict(X,w):

    """ Predicts the class for each data point 
    Args:
        tx: shape=(N,M)
        w: shape=(M, 1). The vector of model parameters
    Returns:
         y: shape=(N, 1), vector of predictions (between 0 and 1)
    """
    return np.round(sigmoid(X @ w)) 


def K_fold_pipeline_log(X_train,Y_train, k, lambda_ridge,lambda_lasso, max_iters, gamma):

    """
    This function implements the K fold cross validation technique on logistic models with penalties
    output : Accuracy metric (average on all)  

    This function can be use to find the best value lambda, gamma of a penalizer or gradient descent

    """
    Y_train = log_transform(Y_train) #Transformer les y en dans l'intervale [0,1]

    # Divides the index into K groups 
    n_dimension = np.shape(X_train)[1]
    original = np.concatenate((X_train,Y_train), axis = 1)
    index = np.arange(original.shape[0])   
    index_group = np.array_split(index,k)
    # Itération sur les k-folds
    error = []
    error_log = []
    w_log = []
    accuracy_ = []  
    for i in index_group:
        testX = original[i,0:-1]
        testY = np.array([original[i,-1]]).T
        train_index = np.setdiff1d(index,i)
        trainX = original[train_index,0:-1]
        trainY = np.array([original[train_index,-1]]).T
        
        # Logistic Model fit with Ridge penalization
        initial_w = np.ones((n_dimension, 1))
        w_temp, error_temp =  reg_logistic_regression(trainY,trainX,lambda_ridge,initial_w, max_iters, gamma)
        error_log.append(error_temp)
        w_log.append(w_temp)
        #Do predictions for the given models
        pred_log = log_predict(trainX,w_temp)
        #compute evaluation metrics
        accuracy_.append(accuracy(testY,pred_log))
        log_transform
    # Itération sur les pénalisateurs ridge et Lasso (lambda_lasso, lambda_ridge) - dans la boucle main
    print("----- Regression logistique ----- ")
    print("k: ",k)
    print("lambda_ridge: ",lambda_ridge)
    print("lambda_lasso: ",lambda_lasso)
    print("max_iters: ",max_iters)
    print("gamma: ",gamma)
    print("Accuracy", np.mean(accuracy_),"\n")
    return None


def log_transform_inverse(y):
    """ 
    Transforme les valeurs de classification dans l'intervale [0,1] pour pourvoir être utilisés dans la régression logistique.
    """
    y= 2*y-1
    y = y.astype(int) 
    return y

class SVM:

    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None


    def fit(self, X, y):

        """Computes the optimal model parameters w and b for the soft vector model 
        Args:
        y: shape=(N, 1)
        X: shape=(N,M)
        w: shape=(M, 1). The vector of model parameters.
        lambda_ : Ridge regularization parameter
    Returns:
        Value of optimal w  
        """
        n_samples, n_features = X.shape
        
        y_ = np.where(y <= 0, -1, 1)
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]


    def predict(self, X):
        """Predictions with new data on SVM 
        """
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

def K_fold_pipeline_SVM(X_train,Y_train, k, lambda_ridge,lambda_lasso, max_iters, gamma):
    """
    This function implements the K fold cross validation technique on SVM models with penalties
    output : Accuracy metric (average on all folds)  
    This function can be use to find the best value lambda, gamma of a penalizer or gradient descent
    """
    # Divides the index into K groups 
    n_dimension = np.shape(X_train)[1]
    original = np.concatenate((X_train,Y_train), axis = 1)
    index = np.arange(original.shape[0])   
    index_group = np.array_split(index,k)
    accuracy_ = []  
    # Itération sur les k-folds

    for i in index_group:
        testX = original[i,0:-1]
        testY = np.array([original[i,-1]]).T
        train_index = np.setdiff1d(index,i)
        trainX = original[train_index,0:-1]
        trainY = np.array([original[train_index,-1]]).T

    # SVM Model fit
        support_vector_temp = SVM(gamma,lambda_ridge, max_iters) # Initialise la classe
        support_vector_temp.fit( X_train, Y_train.flatten()) 
        
    #Do predictions for the given models
        pred_SVM = support_vector_temp.predict(testX)
    #compute evaluation metrics
        accuracy_.append(accuracy(testY,pred_SVM))
        
    # Itération sur les pénalisateurs ridge et Lasso (lambda_lasso, lambda_ridge) - dans la boucle main
    print("----- Soft Vector Machine ----- ")
    print("k: ",k)
    print("lambda_ridge: ",lambda_ridge)
    print("lambda_lasso: ",lambda_lasso)
    print("max_iters: ",max_iters)
    print("gamma: ",gamma)
    print("Accuracy", np.mean(accuracy_),"\n")
    return None

def accuracy(y,ypred): 
    matrix = confusion_matrix(y,ypred)
    return matrix[0,0]+ matrix[1,1]

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

def build_model_data(y, x):
    """Form (y,tX) to get regression data in matrix form."""
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx

def standardize(x):
        centered_data = x - np.mean(x, axis=0)
        
        std = np.std(centered_data, axis=0)
        std[std==0.] = 1
        
        std_data = centered_data / std
        
    
        return std_data

def treat_data(x, mean = True):
    
    '''
    get an imput array of features (columns) and replace all -999. values 
    by the mean value of their column.
    '''
    
    xx = x.copy()
    for ind, column in enumerate(xx.T[:]):
        if mean:
            if np.isnan(column[column!=-999.].mean())==False:
                column[column==-999.] = column[column!=-999.].mean()
            else: 
                column[column==-999.] = 0
            
        else:
            return
            ##rien pour l'instant
        xx.T[ind] = column
        
    return xx

def log_right_skewed(x):
    eps = 1e-6
    xx = x.copy()
    #ids = np.array([0, 1, 2, 3, 5, 8, 9, 10, 13, 16, 23, 26, 29])
    idx_3 = np.array([0, 1, 2, 5, 8, 9, 10, 13, 19, 23, 26, 29])
    for i in idx_3:
        xxx = xx[xx[:,i]!=-999.] 
        minimum = xxx[:, i].min()
        xx[xx[:,i]!=-999.][:, i] = np.log(1+xxx[:, i]+eps-minimum)
    return xx

def remove_unique(xx):
        xxx = np.array(xx.copy())
        liste = []
        for ind, column in enumerate(xx.T[:]):
            if np.std(column)==0.:
                liste.append(ind)
        xxx = np.delete(xxx, obj = liste  , axis = 1) 
        return xxx

def polynomial_exp(x, degree):
    xx = x.copy()
    for d in range(2, degree+1):
        x = np.concatenate((x, np.power(xx, d)), axis = 1)
    return x

def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    # ***************************************************
    # split the data based on the given ratio:
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    split = int( ratio * len(y) )
    train_indices, test_indices= np.split(indices, np.array([split]))
    train_data = x[train_indices]
    train_labels = y[train_indices]
    test_data = x[test_indices]
    test_labels = y[test_indices]
    return train_data, train_labels, test_data, test_labels

