rm(list = ls(all=TRUE))
library(SuperLearner)
library(tidyverse)
library(reticulate)
np <- import('numpy')

### Simulation the data 


simulate <-  function(n, B = 1, probs = 0.1, size_MC = 1e5) {
  data <- data.frame()
  test <- all(probs <= 1 & probs >= 0)
  if (!test) {
    stop("Elements of 'probs' must be between 0 and 1.")
  }
  if (length(probs) == 1) {
    probs <- rep(probs, B)
  } else if (length(probs) != B) {
    stop("Length of 'probs' is either 1 or 'B'.")
  }
  for (bb in 1:B) {
    ## adjusting 'tau'
    r <- rchisq(size_MC, 1)
    tau <- uniroot(function(x, R, pi){mean(plogis(x + R)) - pi},
                   interval = c(-10, 0),
                   R = r, pi = probs[bb])$root
    ##
    r <- rchisq(n, 1)
    theta <- runif(n, min = 0, max = 2 * pi)
    x1 <- r * cos(theta)
    x2 <- r * sin(theta)
    y <- rbinom(n, 1, plogis(tau + r))
    data <- rbind(data,
                  data.frame(x1 = x1, x2 = x2, y = y))
  }
  return(data)
}


if (TRUE){
  rm(list = ls(all=TRUE))
  library(SuperLearner)
  library(tidyverse)
  library(reticulate)
  np <- import('numpy')
  simulate <-  function(n, B = 1, probs = 0.1, size_MC = 1e5) {
    data <- data.frame()
    test <- all(probs <= 1 & probs >= 0)
    if (!test) {
      stop("Elements of 'probs' must be between 0 and 1.")
    }
    if (length(probs) == 1) {
      probs <- rep(probs, B)
    } else if (length(probs) != B) {
      stop("Length of 'probs' is either 1 or 'B'.")
    }
    for (bb in 1:B) {
      ## adjusting 'tau'
      r <- rchisq(size_MC, 1)
      tau <- uniroot(function(x, R, pi){mean(plogis(x + R)) - pi},
                     interval = c(-10, 0),
                     R = r, pi = probs[bb])$root
      ##
      r <- rchisq(n, 1)
      theta <- runif(n, min = 0, max = 2 * pi)
      x1 <- r * cos(theta)
      x2 <- r * sin(theta)
      y <- rbinom(n, 1, plogis(tau + r))
      data <- rbind(data,
                    data.frame(x1 = x1, x2 = x2, y = y))
    }
    return(data)
  }
  SL.library <- c("SL.mean", "SL.glm", "SL.rpart", "SL.ranger", "SL.knn")
  rmse_SL1 = c()
  mae_SL1 = c()
  rmse_SL2 = c()
  mae_SL2 = c()
  
  trainbis = list()
  testbis = list()
  
  theta_SL1 = list()
  theta_SL2 = list()
  
  for (i in 1:30){
    
    train <- simulate(1e3, B = 3, probs = c(0.1, 0.15, 0.05))
    test <- simulate(1e3, 1, probs = 0.05)
    trainbis[[i]] = as.matrix(train)
    testbis[[i]] = as.matrix(test)
    
    
    
    
    
    # first super learner
    SL <- SuperLearner(X = train[, -match("y", colnames(train))],
                       Y = train[, match("y", colnames(train))],
                       newX = test[, -match("y", colnames(test))],
                       SL.library = SL.library,
                       family = "binomial")
    
    rmse <- sqrt(mean((SL$SL.predict - test[, "y"])^2))
    mae  <- mean(abs(SL$SL.predict - test[, "y"]))
    
    rmse_SL1[i] = rmse
    mae_SL1[i] = mae
    theta_SL1[[i]] = SL$SL.predict
    
    
    cat(sprintf('rmse_SL: %f', rmse))
    cat(sprintf('mae_SL: %f', mae))
    
    # second super learner
    train$r <- sqrt(train$x1^2 + train$x2^2)
    test$r <- sqrt(test$x1^2 + test$x2^2)
    
    SL2 <- SuperLearner(X = train[, -match("y", colnames(train))],
                        Y = train[, match("y", colnames(train))],
                        newX = test[, -match("y", colnames(test))],
                        SL.library = SL.library,
                        family = "binomial")
    
    rmse <- sqrt(mean((SL2$SL.predict - test[, "y"])^2))
    mae <- mean(abs(SL2$SL.predict - test[, "y"]))
    
    cat(sprintf('rmse_SL2: %f', rmse))
    cat(sprintf('mae_SL2: %f', mae))
    rmse_SL2[i] = rmse
    mae_SL2[i] = mae
    theta_SL2[[i]]= SL2$SL.predict  
    
    np <- import('numpy')
    np$savez_compressed('data_sim_sl_with_knn_new', train= trainbis, test = testbis, rmse_SL1 = rmse_SL1, mae_SL1 = mae_SL1,
                        rmse_SL2 = rmse_SL2, mae_SL2 = mae_SL2, theta_SL2 = theta_SL2, theta_SL1 = theta_SL1)   
  }
}





