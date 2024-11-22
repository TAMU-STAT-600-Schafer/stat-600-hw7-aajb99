# This is a script to save your own tests for the function
source('FunctionsNN.R')

#####################
# Functions testing #
#####################

### Generate Test Data Function:
# Function to generate data with a "true" underlying structure (generate data, then model Y according to "true" parameters)
gen_data <- function(p, hidden_p, K, n, sd_val, drop_out_rate, nval = 0, seed = 0){
  set.seed(seed)
  
  # Calculate the probabilities of each class by assigning "true" parameters to pass 
  # through a fully connected NN. 
  # This allows for the target function to be explicitly in the model space 
  Xdata <- matrix(rnorm(n*p), nrow = n)
  
  # For W1, randomly generate values and drop some of them
  W1_vals = rnorm(p*hidden_p, sd = sd_val)
  W1_vals[sample(1:(hidden_p*p), size = floor(drop_out_rate*hidden_p*p), replace = F)] <- 0 # drop some of the values to 0
  W1_true = matrix(W1_vals, nrow = p, ncol = hidden_p)
  
  # for b1, randomly generate values and drop some of them 
  b1_true = rnorm(hidden_p, sd = sd_val)
  b1_true[sample(hidden_p, size = floor(drop_out_rate*hidden_p), replace = F)] <- 0 # drop some of the values to 0
  
  # For W2, randomly generate values and drop some of them
  W2_vals = rnorm(K*hidden_p, sd = sd_val)
  W2_vals[sample(1:(hidden_p*K), size = floor(drop_out_rate*hidden_p*K), replace = F)] <- 0 # drop some of the values to 0
  W2_true = matrix(W2_vals, nrow = hidden_p, ncol = K)
  
  # for b2, randomly generate values and drop some of them 
  b2_true = rnorm(K, sd = sd_val)
  b2_true[sample(K, size = floor(drop_out_rate*K), replace = F)] <- 0 # drop some of the values to 0
  
  # Pass X through fully connected network with true parameters
  # First layer and bias
  scores_true = Xdata %*% W1_true + matrix(b1_true, nrow = n, ncol = hidden_p, byrow = T)
  # RELU
  scores_true = (abs(scores_true) + scores_true) / 2
  
  # 2nd layer and bias
  scores_true = scores_true %*% W2_true + matrix(b2_true, nrow = n, ncol = K, byrow = T)
  # Calculate probabilities based on these scores
  true_probs = exp(scores_true)
  true_probs = true_probs/rowSums(true_probs)
  
  # Sometimes values in scores_true are too large and result in NaN in true_probs
  # If this occurs, find NaN values and replace with 1
  # if (anyNA(true_probs)) {
  #   true_probs[is.na(true_probs)] <- 1
  # }
  
  # assign classifications of y based on true probs
  ydata <- vector(mode = 'double', length = n)
  for (i in 1:n) {
    ydata[i] = sample.int(K, size = 1, prob = true_probs[i, ]) - 1
  }
  
  Xval = NULL; yval = NULL
  
  if(nval > 0){
    Xval = Xdata[(n-nval+1):n, , drop = FALSE]
    yval = ydata[(n-nval+1):n]
  }
  
  Xtrain = Xdata[1:(n-nval), , drop = FALSE]
  ytrain = ydata[1:(n-nval)]
  
  return(list(X = Xtrain, y = ytrain, Xval = Xval, yval = yval,
              W1_true = W1_true, b1_true = b1_true, W2_true = W2_true, b2_true = b2_true,
              scores_true = scores_true, probs_true = true_probs))
  
}



### Test initialize_bw():

# Inputs:
p <- 30
hidden_p <- 20
K <- 3

# Test function:
out1 <- initialize_bw(p, hidden_p, K)
# Dimensions:
# b1: correct initialization of 0 vector size hidden_p
out1$b1
length(out1$b1)
# b2: correct initialization of 0 vector size K
out1$b2
length(out1$b2)
# W1: correct p x hidden_p dimensions and initial values
out1$W1
min(out1$W1)
max(out1$W1)
dim(out1$W1)
# W2: correct p x K dimensions and initial values
out1$W2
min(out1$W2)
max(out1$W2)
dim(out1$W2)



### Test loss_grad_scores:
# Inputs:
p <- round(runif(1, 5, 8))
hidden_p <- round(runif(1, 8, 12))
K <- round(runif(1, 3, 5))
n <- round(runif(1, 100, 300))
sd_val <- 1
drop_out_rate <- runif(1, .4, .8)
nval <- n * .2

params_test1 <- gen_data(p, hidden_p, K, n, sd_val, drop_out_rate, nval)

# Test function:
out2 <- loss_grad_scores(params_test1$y, head(params_test1$scores_true, length(params_test1$y)), K) # Dimensions of ouputs are correct



### Test one_pass:

# Inputs:
# Use params_test1

# Test Function:
out3 <- one_pass(params_test1$X, params_test1$y, K, params_test1$W1_true, params_test1$b1_true, params_test1$W2_true, params_test1$b2_true, lambda = 0.001)
out3$loss # Correct dimension
out3$error # Correct dimension
out3$grads # All correct dimensions



### Test evaluate_error:
# Inputs:
# Use params_test1

# Test Function:
out4 <- evaluate_error(params_test1$Xval, params_test1$yval, params_test1$W1_true, params_test1$b1_true, params_test1$W2_true, params_test1$b2_true)
out4 # Correct dimension



### Test NN_train:
# Inputs:
p <- 5
hidden_p <- 50
K <- 4
n <- 1000
sd_val <- 1
drop_out_rate <- .4
nval <- n * .2

params_test2 <- gen_data(p, hidden_p, K, n, sd_val, drop_out_rate, nval)

out5 <- NN_train(params_test1$X, params_test1$y,
                 params_test1$Xval, params_test1$yval,
                 lambda = 0.001, rate = 0.05, 
                 mbatch = 30, nEpoch = 200,
                 hidden_p = 130, scale = 1e-3, 
                 seed = 12345)
# Check error rates:
plot(out5$error)
plot(out5$error_val)



#######################################################
# Speed Tests

# Load Letter Data:
# Training data
letter_train <- read.table("Data/letter-train.txt", header = F, colClasses = "numeric")
Y <- letter_train[, 1]
X <- as.matrix(letter_train[, -1])

# Update training to set last part as validation
id_val = 1801:2000
Yval = Y[id_val]
Xval = X[id_val, ]
Ytrain = Y[-id_val]
Xtrain = X[-id_val, ]

# Testing data
letter_test <- read.table("Data/letter-test.txt", header = F, colClasses = "numeric")
Yt <- letter_test[, 1]
Xt <- as.matrix(letter_test[, -1])

# Required packages:
library(microbenchmark)
microbenchmark(
  
  my_NN_train <- NN_train(Xtrain, Ytrain, Xval, Yval, lambda = 0.001,
                          rate = 0.1, mbatch = 50, nEpoch = 150,
                          hidden_p = 100, scale = 1e-3, seed = 12345),

  times = 10L
  
)

# Speed: Median is ~6.4 seconds (Intel Chip) on Letter Data


########################################################
# Garbage Collection

library(profvis)
profvis(
  {
    NN_train(Xtrain, Ytrain, Xval, Yval, lambda = 0.001,
             rate = 0.1, mbatch = 50, nEpoch = 150,
             hidden_p = 100, scale = 1e-3, seed = 12345)
  }
)


