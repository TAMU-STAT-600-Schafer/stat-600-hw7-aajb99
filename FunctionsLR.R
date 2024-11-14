# Function that implements multi-class logistic regression.
#############################################################
# Description of supplied parameters:
# X - n x p training data, 1st column should be 1s to account for intercept
# y - a vector of size n of class labels, from 0 to K-1
# Xt - ntest x p testing data, 1st column should be 1s to account for intercept
# yt - a vector of size ntest of test class labels, from 0 to K-1
# numIter - number of FIXED iterations of the algorithm, default value is 50
# eta - learning rate, default value is 0.1
# lambda - ridge parameter, default value is 1
# beta_init - (optional) initial starting values of beta for the algorithm, should be p x K matrix 

## Return output
##########################################################################
# beta - p x K matrix of estimated beta values after numIter iterations
# error_train - (numIter + 1) length vector of training error % at each iteration (+ starting value)
# error_test - (numIter + 1) length vector of testing error % at each iteration (+ starting value)
# objective - (numIter + 1) length vector of objective values of the function that we are minimizing at each iteration (+ starting value)
LRMultiClass <- function(X, y, Xt, yt, numIter = 50, eta = 0.1, lambda = 1, beta_init = NULL){
  ## Check the supplied parameters as described. You can assume that X, Xt are matrices; y, yt are vectors; and numIter, eta, lambda are scalars. You can assume that beta_init is either NULL (default) or a matrix.
  ###################################

  # Initialize K, p, n:
  K <- length(unique(y))
  p <- dim(X)[2]
  n <- length(y)
  # Initialize vector objects (errors/objective):
  error_train <- vector()
  error_test <- vector()
  objective <- vector()
  # Initialize beta:
  beta <- beta_init
  
  # Check that the first column of X and Xt are 1s, if not - display appropriate message and stop execution.
  if (!all(X[,1] == 1) | !all(Xt[,1] == 1)){
    
    stop(paste("Either X or Xt (or both) contains values other than 1 in first column. Check and readjust."))
    
  }
  
  ###
  
  # Check for compatibility of dimensions between X and Y
  if (dim(X)[1] != n) {
    
    stop(paste("Dimensions between X and Y are not compatible. Check and readjust."))
    
  }
  
  ###
  
  # Check for compatibility of dimensions between Xt and Yt
  if (dim(Xt)[1] != length(yt)) {
    
    stop(paste("Dimensions between Xt and Yt are not compatible. Check and readjust."))
    
  }
  
  ###
  
  # Check for compatibility of dimensions between X and Xt
  if (p != dim(Xt)[2]) {
    
    stop(paste("Dimensions between X and Xt are not compatible. Check and readjust."))
    
  }
  
  ###
  
  # Check eta is positive
  if (eta <= 0){
    
    stop(paste("eta parameter is not positive. Readjust."))
    
  }
  
  ###
  
  # Check lambda is non-negative
  if (lambda < 0){
    
    stop(paste("lambda parameter is negative. Readjust."))
    
  }
  
  ###
  
  # Check whether beta is NULL. If NULL, initialize beta with p x K matrix of zeroes. If not NULL, check for compatibility of dimensions with what has been already supplied.
  if (all(is.null(beta)) | all(is.na(beta))) {
    
    beta <- matrix(0, nrow = dim(X)[2], ncol = K)
    
  }
  # If not all NULL/NA, check compatibility with X and K
  else {
    
    # If dimensions incompatible: stop job, print error statement
    if (dim(beta)[1] != p | dim(beta)[2] != K) {
      
      stop(paste("Dimensions of beta not compatible with p and/or K. Check and readjust."))
      
    }
    # If compatible, continue
    
  }
  
  
  ## Calculate corresponding pk, objective value f(beta), training error and testing error given the starting point beta
  ##########################################################################
  
  # Compute pk: #
  ###############
  
  # For X:
  # Num
  Xb <- X %*% beta
  exp_Xb <- exp(Xb)
  # Denom
  sum_exp_Xb <- rowSums(exp_Xb)
  # pk:
  p_k <- exp_Xb / sum_exp_Xb
  
  # For Xt:
  # Num
  Xtb <- Xt %*% beta
  exp_Xtb <- exp(Xtb)
  # Denom
  sum_exp_Xtb <- rowSums(exp_Xtb)
  # pk:
  p_kt <- exp_Xtb / sum_exp_Xtb
  
  ###
  
  # Compute Objective Value f(beta) #
  ########################################
  
  y_factor <- as.factor(y)
  y_indicator <- model.matrix(~ y_factor - 1)
  
  objective_obj <- 
    -sum(diag(y_indicator %*% t(log(p_k)))) + # Negative Log Likelihood
    ((lambda / 2) * sum(colSums(beta^2))) # Ridge Penalty
  
  objective <- append(objective, objective_obj)
  
  ###
  
  # Compute Training/Testing Errors #
  ###################################
  
  # Train
  y_preds <- apply(p_k, 1, which.max) - 1
  # Compute percent
  error_train_obj <- (1 - mean(y_preds == y)) * 100
  
  error_train <- append(error_train, error_train_obj)
  
  # Test
  yt_preds <- apply(p_kt, 1, which.max) - 1
  # Compute percent
  error_test_obj <- (1 - mean(yt_preds == yt)) * 100
  
  error_test <- append(error_test, error_test_obj)
  
  
  ############## ############## ############## ############## ##############
  
  
  ## Newton's method cycle - implement the update EXACTLY numIter iterations
  ##########################################################################
  
  # Initialize Terms:
  X_tran <- t(X)
  lambda_I <- lambda * diag(1, nrow = p)
  g <- rep(NA, K)
  W <- rep(NA, n)
  h <- matrix(NA, K, K)
  X_W <- matrix(NA, n, p)
  
 
  # Within one iteration: perform the update, calculate updated objective function and training/testing errors in %
  
  for (i in 1:numIter){
    
    for (k in 1:K){
      
      # W term configuration (in Hessian):
      W <- p_k[, k] * (1 - p_k[, k])
      
      # Weighted multiplication of matrix X
      X_W <- X * sqrt(W)
      
      # Gradient Update:
      g <- X_tran %*% (p_k[, k] - y_indicator[, k]) + (lambda * beta[, k])
      
      # Hessian Update:
      h <- (t(X_W) %*% X_W) + (lambda_I)
      
      # Damped Newton's Update:
      beta[, k] <- beta[, k] -eta * (solve(h) %*% g)
      
    }
    
    # Update p_k #
    ##############
    
    # Train:
    # Num
    Xb <- X %*% beta
    exp_Xb <- exp(Xb)
    # Denom
    sum_exp_Xb <- rowSums(exp_Xb)
    # pk:
    p_k <- exp_Xb / sum_exp_Xb
    # Test:
    #######
    Xtb <- Xt %*% beta
    exp_Xtb <- exp(Xtb)
    # Denom
    sum_exp_Xtb <- rowSums(exp_Xtb)
    # pkt:
    p_kt <- exp_Xtb / sum_exp_Xtb
    
    
    # Compute Training/Testing Errors #
    ###################################
    
    # Train
    y_preds <- apply(p_k, 1, which.max) - 1
    # Compute percent
    error_train_obj <- (1 - mean(y_preds == y)) * 100
    
    error_train <- append(error_train, error_train_obj)
    
    # Test
    yt_preds <- apply(p_kt, 1, which.max) - 1
    # Compute percent
    error_test_obj <- (1 - mean(yt_preds == yt)) * 100
    
    error_test <- append(error_test, error_test_obj)
    
    
    # Compute Objective Value #
    ###########################
    # objective_obj <- 
    #   -sum(diag(y_indicator %*% t(log(p_k)))) + # Negative Log Likelihood
    #   ((lambda / 2) * sum(colSums(beta^2))) # Ridge Penalty
    
    log_pk <- log(p_k)
    neg_log_lik <- -sum(y_indicator * log_pk) # Negative Log Likelihood
    ridge_reg <- (lambda / 2) * sum(beta^2) # Ridge Penalty
    objective_obj <- neg_log_lik + ridge_reg
    
    objective <- append(objective, objective_obj)
    
  }
  
  
  ## Return output
  ##########################################################################
  # beta - p x K matrix of estimated beta values after numIter iterations
  # error_train - (numIter + 1) length vector of training error % at each iteration (+ starting value)
  # error_test - (numIter + 1) length vector of testing error % at each iteration (+ starting value)
  # objective - (numIter + 1) length vector of objective values of the function that we are minimizing at each iteration (+ starting value)
  return(list(beta = beta, error_train = error_train, error_test = error_test, objective =  objective))
}