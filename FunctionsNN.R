# Initialization
#####################################################
# p - dimension of input layer
# hidden_p - dimension of hidden layer
# K - number of classes, dimension of output layer
# scale - magnitude for initialization of W_k (standard deviation of normal)
# seed - specified seed to use before random normal draws
initialize_bw <- function(p, hidden_p, K, scale = 1e-3, seed = 12345){
  # [ToDo] Initialize intercepts as zeros
  b1 <- rep(0, hidden_p)
  b2 = 0
  
  # [ToDo] Initialize weights by drawing them iid from Normal
  # with mean zero and scale as sd
  set.seed(seed)
  W1 <- scale * matrix(rnorm(p * hidden_p), nrow =  p, ncol = hidden_p)
  W2 <- scale * matrix(rnorm(hidden_p ), nrow = hidden_p, ncol = K)
  
  # Return
  return(list(b1 = b1, b2 = b2, W1 = W1, W2 = W2))
}

# Function to calculate loss, error, and gradient strictly based on scores
# with lambda = 0
#############################################################
# scores - a matrix of size n by K of scores (output layer)
# y - a vector of size n of class labels, from 0 to K-1
# K - number of classes
loss_grad_scores <- function(y, scores, K){
  
  # [ToDo] Calculate loss when lambda = 0
  
  # Compute pk: #
  ###############
  
  # For X:
  # Num
  exp_scores <- exp(scores)
  # Denom
  sum_exp_scores <- rowSums(exp_scores)
  # pk:
  p_k <- exp_scores / sum_exp_scores
  
  ###
  
  # Compute Objective Value f(beta) (loss) #
  ##########################################
  
  y_factor <- as.factor(y)
  y_indicator <- model.matrix(~ y_factor - 1)
  
  loss <- -sum(diag(y_indicator %*% t(log(p_k)))) / n
  
  ###
  
  # [ToDo] Calculate misclassification error rate (%)
  # when predicting class labels using scores versus true y
  y_preds <- apply(p_k, 1, which.max) - 1
  # Compute percent
  error <- (1 - mean(y_preds == y)) * 100
  
  
  # [ToDo] Calculate gradient of loss with respect to scores (output)
  # when lambda = 0
  # grad <- y_indicator * (p_k - 1) / n
  grad <- (p_k - y_indicator) / n
  
  
  # Return loss, gradient and misclassification error on training (in %)
  return(list(loss = loss, grad = grad, error = error))
}

# One pass function
################################################
# X - a matrix of size n by p (input)
# y - a vector of size n of class labels, from 0 to K-1
# W1 - a p by h matrix of weights
# b1 - a vector of size h of intercepts
# W2 - a h by K matrix of weights
# b2 - a vector of size K of intercepts
# lambda - a non-negative scalar, ridge parameter for gradient calculations
one_pass <- function(X, y, K, W1, b1, W2, b2, lambda){

  # [To Do] Forward pass
  # From input to hidden 
  H1 <- X %*% W1 + matrix(b1, nrow = n, ncol = length(b1), byrow = TRUE)
  
  # ReLU
  H1 <- (abs(H1) + H1)/2
  
  # From hidden to output scores
  scores <- H1 %*% W2 + b2
  
  # [ToDo] Backward pass
  # Get loss, error, gradient at current scores using loss_grad_scores function
  out <- loss_grad_scores(y, scores, K)
  
  # Get gradient for 2nd layer W2, b2 (use lambda as needed)
  dW2 <- crossprod(H1, out$grad) + lambda * W2
  db2 <- colSums(out$grad)
  
  # Get gradient for hidden, and 1st layer W1, b1 (use lambda as needed)
  dH = tcrossprod(loss_grad$grad, W2)
  dH[H1 == 0] <- 0
  dW1 = crossprod(X, dH) + lambda * W1
  db1 = colSums(dH)
  
  # Return output (loss and error from forward pass,
  # list of gradients from backward pass)
  return(list(loss = out$loss, error = out$error, grads = list(dW1 = dW1, db1 = db1, dW2 = dW2, db2 = db2)))
}

# Function to evaluate validation set error
####################################################
# Xval - a matrix of size nval by p (input)
# yval - a vector of size nval of class labels, from 0 to K-1
# W1 - a p by h matrix of weights
# b1 - a vector of size h of intercepts
# W2 - a h by K matrix of weights
# b2 - a vector of size K of intercepts
evaluate_error <- function(Xval, yval, W1, b1, W2, b2){
  # [ToDo] Forward pass to get scores on validation data
  nval <- nrow(Xval)
  H1 <- Xval %*% W1 + matrix(b1, nrow = nval, ncol = length(b1), byrow = TRUE)
  # ReLU
  H1 <- (abs(H1) + H1)/2
  # From hidden to output scores
  scores <- H1 %*% W2 + b2
  
  # [ToDo] Evaluate error rate (in %) when 
  # comparing scores-based predictions with true yval
  error <- (1 - mean(scores == yval)) * 100
  
  return(error)
}


# Full training
################################################
# X - n by p training data
# y - a vector of size n of class labels, from 0 to K-1
# Xval - nval by p validation data
# yval - a vector of size nval of of class labels, from 0 to K-1, for validation data
# lambda - a non-negative scalar corresponding to ridge parameter
# rate - learning rate for gradient descent
# mbatch - size of the batch for SGD
# nEpoch - total number of epochs for training
# hidden_p - size of hidden layer
# scale - a scalar for weights initialization
# seed - for reproducibility of SGD and initialization
NN_train <- function(X, y, Xval, yval, lambda = 0.01,
                     rate = 0.01, mbatch = 20, nEpoch = 100,
                     hidden_p = 20, scale = 1e-3, seed = 12345){
  # Get sample size and total number of batches
  n = length(y)
  nBatch = floor(n/mbatch)

  # [ToDo] Initialize b1, b2, W1, W2 using initialize_bw with seed as seed,
  # and determine any necessary inputs from supplied ones
  # Necessary Inputs:
  p <- ncol(X) # Number of features
  # Initializing weights/intercepts via initialize_bw:
  init <- initialize_bw(p, hidden_p, scale = scale, seed = seed)
  b1 <- init$b1
  b2 <- init$b2
  W1 <- init$W1
  W2 <- init$W2
  
  # Initialize storage for error to monitor convergence
  error = rep(NA, nEpoch)
  error_val = rep(NA, nEpoch)
  
  # Set seed for reproducibility
  set.seed(seed)
  # Start iterations
  for (i in 1:nEpoch){
    # Allocate bathes
    batchids = sample(rep(1:nBatch, length.out = n), size = n)
    # [ToDo]
    # Accumulate loss over batches, and compute average:
    error_loss <- 0
    # [ToDo] For each batch
    #  - do one_pass to determine current error and gradients
    #  - perform SGD step to update the weights and intercepts
    for (j in 1:nBatch){
      
      # Get loss and gradient on the batch
      pass <- one_pass(X[batchids == j, ], y[batchids == j], W1, b1, W2, b2)
      
      # Keep track of loss
      error_loss <- error_loss + pass$loss
      
      # [ToDo] Make an update of W1, b1, W2, b2
      W1 <- W1 - rate * pass$grads$dW1
      b1 <- b1 - rate * pass$grads$db1
      W2 <- W2 - rate * pass$grads$dW2
      b2 <- b2 - rate * pass$grads$db2
      
    }
    
    # [ToDo] In the end of epoch, evaluate
    # - average training error across batches
    error[i] <- error_loss / nBatch
    
    # - validation error using evaluate_error function
    error_val[i] <- loss_only(Xval, yval, W1, b1, W2, b2)
    
  }
  # Return end result
  return(list(error = error, error_val = error_val, params =  list(W1 = W1, b1 = b1, W2 = W2, b2 = b2)))
}