import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import csv
from numpy import linalg as LA
from helpers import *
from Tools import *

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
    return losses, ws

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
    #return (cost[0], w)
    return w, cost


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    #reg_logistic_regression(tx,y,lambda_ridge,initial_w, max_iters,gamma):
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
        error_temp, w_temp =  reg_logistic_regression(trainX,trainY,lambda_ridge,initial_w, max_iters, gamma)
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

def Pipeline_log(X_train, Y_train,X_test,  Id_test, name ="submission_logistic.csv") :
    """ 
    Fit, predict et export pour le modèle logistique
    Choisir les hyperparamètres en fonction des K fold
    """
    #Fit logistic regression
    n_dimension = np.shape(X_train)[1]
    initial_w = np.ones((n_dimension, 1))
    loss_log, w_log = reg_logistic_regression(X_train,Y_train, 0.1, initial_w, 100, 0.01)

    #Predict logistic regression
    y_predict_log = log_predict(X_test,w_log)
    y_predict_log = log_transform_inverse(y_predict_log)
    export_to_csv(y_predict_log, Id_test, "submission_logistic.csv")
    return None
