library(tidyverse)
library(caret)
library(pROC)
library(e1071)
library(dplyr)

#set for reproducibility
set.seed(1)

#simulating logistic distribution  with 10 explanatory variables (7 continuous and 3 categorical)
logistic_data <- function(sample_size=1000, 
                          beta_0=7, 
                          beta_1=5, 
                          beta_2=-10,
                          beta_3=25, 
                          beta_4=6, 
                          beta_5=-9,
                          beta_6=20,
                          beta_7=8,
                          beta_8=3,
                          beta_9=-14,
                          beta_10=-2) {
  x1 = rnorm(n=sample_size)
  x2 = rbinom(n=sample_size, size=1, prob=0.6)
  x3 = rnorm(n=sample_size, mean=10, sd=50)
  x4 = rbinom(n=sample_size, size=1, prob=0.4)
  x5 = rnorm(n=sample_size, mean=75, sd=100)
  x6 = rnorm(n=sample_size, mean=-2.5, sd=8)
  x7 = rnorm(n=sample_size, mean=-25, sd=50)
  x8 = rnorm(n=sample_size, mean=250, sd=500)
  x9 = rbinom(n=sample_size, size=1, prob=0.2)
  x10 = rnorm(n=sample_size, mean=-0.8, sd=9)
  linpred = beta_0 + beta_1*x1 + beta_2*x2 + beta_3*x3 + beta_4*x4 + 
    beta_5*x5 + beta_6*x6 + beta_7*x7 + beta_8*x8 + beta_9*x9 + beta_10*x10 + rnorm(n=sample_size, mean=0, sd=250) #add noise term
  p = 1 / (1 + exp(-linpred))
  y = rbinom(n = sample_size, size = 1, prob = p)
  data.frame(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, y)
}
data <- logistic_data()
head(data)
######################### Playing around #########################
# print(typeof(data))
# print(data.shape)
# df = data.frame(Reduce(rbind, data))
# 
# print(data[1:5,1:3])
# 
# l <- list(a = list(var.1 = 1, var.2 = 2, var.3 = 3)
#           , b = list(var.1 = 4, var.2 = 5, var.3 = 6)
#           , c = list(var.1 = 7, var.2 = 8, var.3 = 9)
#           , d = list(var.1 = 10, var.2 = 11, var.3 = 12)
# )
# 
# print(typeof(l))
# 
# df <- ldply(data, data.frame)
# 
# for (x in data) {
#   print(x[1,1])
# }
# 
# sample_size=1000 
# beta_0=7
# beta_1=5 
# beta_2=-10
# beta_3=25 
# beta_4=6 
# beta_5=-9
# beta_6=20
# beta_7=8
# beta_8=3
# beta_9=-14
# beta_10=-2
# x1 = rnorm(n=sample_size)
# x2 = rbinom(n=sample_size, size=1, prob=0.6)
# x3 = rnorm(n=sample_size, mean=10, sd=50)
# x4 = rbinom(n=sample_size, size=1, prob=0.4)
# x5 = rnorm(n=sample_size, mean=75, sd=100)
# x6 = rnorm(n=sample_size, mean=-2.5, sd=8)
# x7 = rnorm(n=sample_size, mean=-25, sd=50)
# x8 = rnorm(n=sample_size, mean=250, sd=500)
# x9 = rbinom(n=sample_size, size=1, prob=0.2)
# x10 = rnorm(n=sample_size, mean=-0.8, sd=9)
# linpred = beta_0 + beta_1*x1 + beta_2*x2 + beta_3*x3 + beta_4*x4 + 
# beta_5*x5 + beta_6*x6 + beta_7*x7 + beta_8*x8 + beta_9*x9 + beta_10*x10 + rnorm(n=sample_size, mean=0, sd=250) #add noise term
# p = 1 / (1 + exp(-linpred))
# y = rbinom(n = sample_size, size = 1, prob = p)
# new_data <- data.frame(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, y)
# 
# print(typeof(new_data))
# ggplot(data = new_data)+
#   geom_point(mapping = aes(x = x1,y = x3))

######################### Playing around #########################

#repetitions stored in an array with 1000 samples and 1000 repetitions
rep=1000
sim <- replicate(n = rep, expr = logistic_data(), simplify = FALSE) #original dataset, no missing values
targets <- data.frame(sapply(sim, "[[", 11)) #get the target variable of each dataset

#target variable y distribution
freq <- lapply(targets, function(y) as.data.frame(table(y)))

target_1<-colMeans(targets[,]) 
min(target_1) #minimum count of target=1
max(target_1) #maximum count of target=1
plot(target_1, xlab="Simulation Number", ylab="Proportion of target=1")

#correlated matrix
correlationMatrix <- abs(cor(sim[[1]][,1:10]))
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.7, names=TRUE)
highlyCorrelated

# Function for splitting training and validation set 
splitdata = function(x) {
  set.seed(321)
  
  trainind = function(x) {
    x[sort(sample(nrow(x), nrow(x)*0.7)),]
  }
  
  validind = function(x) {
    x[-sort(sample(nrow(x), nrow(x)*0.7)),]
  }
  
  trainset <- lapply(x, trainind)
  validset <- lapply(x, validind)
  
  return(list(trainset, validset))
}

fulltrainset <- splitdata(sim)[[1]]
fullvalidset <- splitdata(sim)[[2]]

#Scenario 1: 15% missing values and remove missing values MCAR
# library(missForest)
# set.seed(321)
# generatmissingvalues <- function(df, var.name, prop.of.missing) {
#   df.buf <- subset(df, select=c(var   1315.name))                      # Select variable
#   df.buf <- prodNA(x = df.buf, prop.of.missing)                 # change original value to NA randomly
#   df.col.order <- colnames(x = df)                              # save the column order       
#   df <- subset(df, select=-c(which(colnames(df)==var.name)))    # drop the variable with no NAs    
#   df <- cbind(df, df.buf)                                       # add the column with NA          
#   df <- subset(df, select=df.col.order)                         # restore the original order sequence 
#   
#   return(df)  
# }
# 
# missing15 <- lapply(sim,generatmissingvalues, var.name = "x3", prop.of.missing = .15)
# missing15 <- lapply(missing15,generatmissingvalues, var.name = "x5", prop.of.missing = .15)
# missing15 <- lapply(missing15,generatmissingvalues, var.name = "x8", prop.of.missing = .15)
# 
# #testing on first repetition
# colSums(is.na(missing15[[1]])) #exactly 15% missing values 
# #logmissing15[[1]][rowSums(is.na(logmissing15[[1]])) > 0, ]  
# 
# #remove rows with missing values
# missing15removed <- lapply(missing15, na.omit)
# missing15removedtrainset <- splitdata(missing15removed)[[1]]
# missing15removedvalidset <- splitdata(missing15removed)[[2]]


#################################SECOND METHOD with MAR#######################################
library(mice)  
missing15generator <- function(x) {
  set.seed(0) #generate same missing values 
  my_patterns <- c(1,1,0,1,0,1,1,0,1,1,1, 
                   1,1,0,1,0,1,1,1,1,1,1,
                   1,1,1,1,0,1,1,0,1,1,1,
                   1,1,0,1,1,1,1,0,1,1,1)
  dataset=ampute(x,prop=0.15, patterns=my_patterns)
  dataset$amp
}

missing15trainset <- lapply(fulltrainset,missing15generator)
missing15removedtrainset <- lapply(missing15trainset, na.omit)
missing15validset <- lapply(fullvalidset,missing15generator) #don't delete rows with missing values to keep same sample size for comparison


#Scenario 2: 15% missing values and replace with mean
meanimpute <- function(x) {
  mean1 <- mean(x[,3], na.rm=T)
  mean2 <- mean(x[,5], na.rm=T)
  mean3 <- mean(x[,8], na.rm=T)
  return(list(mean1, mean2, mean3))
}

calculatemean <- lapply(missing15trainset, meanimpute) #calculate mean on training set

replacemean <- function(x, y) {
  x[is.na(x[,3]),3]<-y[1]
  x[is.na(x[,5]),5]<-y[2]
  x[is.na(x[,8]),8]<-y[3]
  x
}

missing15meantrainset <- Map(function(x,y) replacemean(x,y), missing15trainset, calculatemean) #replace mean
missing15meanvalidset <- Map(function(x,y) replacemean(x,y), missing15validset, calculatemean) #replace mean

#testing if there are missing values 
colSums(is.na(missing15meantrainset[[1]])) 
colSums(is.na(missing15meanvalidset[[1]])) 

#Scenario 3: 15% missing values and CART
library(rpart)
missing15carttrainset=missing15trainset
missing15cartvalidset=missing15validset
imputemodels=lapply(paste(names(missing15carttrainset[[1]]),"~.-y"),as.formula) #same formulas for all every simulation

for (j in 1:rep) {
  imputation_tree123=lapply(imputemodels,rpart,data=missing15carttrainset[[j]])
  for(i in c(3,5,8)){ #select columns with missing values
    #replace missing values with predictions 
    missing15carttrainset[[j]][i][is.na(missing15carttrainset[[j]][i])]=predict(imputation_tree123[[i]],
                                                                                newdata=missing15carttrainset[[j]][is.na(missing15carttrainset[[j]][i]),],type="vector")
    
    missing15cartvalidset[[j]][i][is.na(missing15cartvalidset[[j]][i])]=predict(imputation_tree123[[i]],
                                                                                newdata=missing15cartvalidset[[j]][is.na(missing15cartvalidset[[j]][i]),],type="vector")
  }
}


#testing if there are missing values
colSums(is.na(missing15carttrainset[[1]])) 
colSums(is.na(missing15cartvalidset[[1]])) 

#Scenario 4: 30% missing values and remove missing values
# missing30 <- lapply(sim,generatmissingvalues, var.name = "x3", prop.of.missing = .3)
# missing30 <- lapply(missing30,generatmissingvalues, var.name = "x5", prop.of.missing = .3)
# missing30 <- lapply(missing30,generatmissingvalues, var.name = "x8", prop.of.missing = .3)
# 
# missing30removed <- lapply(missing30, na.omit) #remove and then split
# missing30removedtrainset <- splitdata(missing30removed)[[1]]
# missing30removedvalidset <- splitdata(missing30removed)[[2]]

#################################SECOND METHOD with MAR#######################################
missing30generator <- function(x) {
  set.seed(0)
  my_patterns <- c(1,1,0,1,0,1,1,0,1,1,1, 
                   1,1,0,1,0,1,1,1,1,1,1,
                   1,1,1,1,0,1,1,0,1,1,1,
                   1,1,0,1,1,1,1,0,1,1,1)
  dataset=ampute(x,prop=0.30, patterns=my_patterns)
  dataset$amp
}

missing30trainset <- lapply(fulltrainset,missing30generator)
missing30removedtrainset <- lapply(missing30trainset, na.omit)
missing30validset <- lapply(fullvalidset,missing30generator)


#Scenario 5: 30% missing values and replace with mean
calculatemean2 <- lapply(missing30trainset, meanimpute) #calculate mean on training set
missing30meantrainset <- Map(function(x,y) replacemean(x,y), missing30trainset, calculatemean2) #replace mean
missing30meanvalidset <- Map(function(x,y) replacemean(x,y), missing30validset, calculatemean2) #replace mean

#Scenario 6: 30% missing values and CART
missing30carttrainset=missing30trainset
missing30cartvalidset=missing30validset
imputemodels=lapply(paste(names(missing30carttrainset[[1]]),"~.-y"),as.formula) #same formulas for all every simulation

for (j in 1:rep) {
  imputation_tree123=lapply(imputemodels,rpart,data=missing30carttrainset[[j]])
  for(i in c(3,5,8)){ #select columns with missing values
    #replace missing values with predictions 
    missing30carttrainset[[j]][i][is.na(missing30carttrainset[[j]][i])]=predict(imputation_tree123[[i]],
                                                                                newdata=missing30carttrainset[[j]][is.na(missing30carttrainset[[j]][i]),],type="vector")
    
    missing30cartvalidset[[j]][i][is.na(missing30cartvalidset[[j]][i])]=predict(imputation_tree123[[i]],
                                                                                newdata=missing30cartvalidset[[j]][is.na(missing30cartvalidset[[j]][i]),],type="vector")
  }
}


##################### Logistic Regression ########################
logisticmodel = function(train,valid) {
  x_validset <- lapply(valid, "[", 1:10)
  y_validset <- lapply(valid, "[", 11)
  
  logisticfunction = function(i) {
    glm(y ~ x1+x2+x3+x4+x5+x6+x7+x8+x9+x10, data = i, family = binomial)
  }
  logmodel <- lapply(train, logisticfunction)
  
  logpredprob <- Map(function(s, l) predict(s, newdata=l, type="response"), logmodel, x_validset)
  
  logpredclass = function(set) {
    as.factor(ifelse(set > 0.5, 1, 0))
  }
  logpredictions <- lapply(logpredprob, logpredclass)
  
  actualclass = function(x) {
    as.factor(x[[1]])
  }
  actual <- lapply(y_validset, actualclass) 
  logconfusionmatrix <- Map(function(s, l) confusionMatrix(s,l), logpredictions, actual)
  misclassificationrate <- as.numeric(paste(unlist(lapply(logconfusionmatrix, function(x) x$overall[1]))))
  roc <-  Map(function(s, l) roc(s,l), actual, logpredprob)
  auc <- as.numeric(paste(unlist(lapply(roc, auc)))) 
  
  print(logconfusionmatrix[[1]]) #confusion matrix of first simulation
  print(caret::varImp(logmodel[[1]]))   #variable importance of first simulation
  #print(which(logpredictions[[1]] != actual[[1]]))
  print(summary(logmodel[[1]]))
  #plot ROC curve for all simulations
  color <- c(rep("#ff000010", rep/2-1), rep("#0000ff10", rep/2))
  plot(roc[[1]], legacy.axes=TRUE, main="ROC curve", xlim=c(1,0))
  for (i in 2:rep){
    lines(roc[[i]], type="l", col=color[i])
  }
  text(0,0.6,labels=paste("Avg AUC=",round(mean(auc),4)))
  
  #create output for comparisons
  output <- list("misclassification rate", misclassificationrate, 
                 "avg misclassification rate", mean(misclassificationrate), 
                 "range of misclassification rate", range(misclassificationrate),
                 "auc score", auc, 
                 "avg auc score", mean(auc),
                 "range of auc score", range(auc))
  return(output)
}

#call function
log_scenario1<- logisticmodel(fulltrainset,fullvalidset)
log_scenario2 <- logisticmodel(missing15removedtrainset, missing15validset)
log_scenario3 <- logisticmodel(missing15meantrainset, missing15meanvalidset)
log_scenario4 <- logisticmodel(missing15carttrainset, missing15cartvalidset)
log_scenario5 <- logisticmodel(missing30removedtrainset, missing30validset)
log_scenario6 <- logisticmodel(missing30meantrainset, missing30meanvalidset)
log_scenario7 <- logisticmodel(missing30carttrainset, missing30cartvalidset)

#create dataframe of results
model_performance <- c("Logistic: Avg_Accuracy", "Logistic: Avg_AUC")
true_model <- c(round(unlist(log_scenario1[4]),4), round(unlist(log_scenario1[10]),4))
remove_0.15 <- c(round(unlist(log_scenario2[4]),4), round(unlist(log_scenario2[10]),4))
mean_0.15 <- c(round(unlist(log_scenario3[4]),4), round(unlist(log_scenario3[10]),4))
cart_0.15 <- c(round(unlist(log_scenario4[4]),4), round(unlist(log_scenario4[10]),4))
remove_0.3 <- c(round(unlist(log_scenario5[4]),4), round(unlist(log_scenario5[10]),4))
mean_0.3 <- c(round(unlist(log_scenario6[4]),4), round(unlist(log_scenario6[10]),4))
cart_0.3 <- c(round(unlist(log_scenario7[4]),4), round(unlist(log_scenario7[10]),4))
df_results <- data.frame(model_performance, true_model ,remove_0.15, mean_0.15, cart_0.15, remove_0.3, mean_0.3, cart_0.3)
df_results

#boxplot for misclassification rate 
model1a <- unlist(log_scenario1[2])
model2a <- unlist(log_scenario2[2])
model3a <- unlist(log_scenario3[2])
model4a <- unlist(log_scenario4[2])
model5a <- unlist(log_scenario5[2])
model6a <- unlist(log_scenario6[2])
model7a <- unlist(log_scenario7[2])
df_accuracy <- data.frame(model1a, model2a, model3a, model4a, model5a, model6a, model7a)
boxplot(df_accuracy,main="Logistic Regression Comparison of Misclassification Rate", 
        xlab="Models", 
        names=c("model1_original", "model2_0.15", "model3_0.15", "model4_0.15", "model5_0.3", "model6_0.3", "model7_0.3"),
        ylab="Misclassification Rate",
        col = c("green","yellow","purple", "orange", "yellow", "purple", "orange"),
        cex.axis=0.70,
        ylim = c(0.55, 0.9))

legend("topright", inset=.02, title="Missing Value Methods",
       c("Removing rows","Imputing Mean","Cart Impute"), fill=c("yellow", "purple", "orange"), horiz=TRUE, cex=0.8)

######################### CART ###############################
library(ggplot2)
library(rpart.plot)

firstsim <- data.frame(sim[[1]])
qplot(x3, x8, data=firstsim, colour=y)

# trainID = sample(1:1000,700)
# training <- firstsim[trainID,]
# validate <- firstsim[-trainID,]
# x_valid <- subset(validate, select = -c(y))
# y_valid <- validate %>% select(y)
# 
# fit <- rpart(y ~ x1+x2+x3+x4+x5+x6+x7+x8+x9+x10, data = training, method = 'class')
# summary(fit)
# rpart.plot(fit)
# 
# predict_tree <-predict(fit, x_valid, type = 'class')
# #table_mat <- table(predict_tree, y_valid$y)
# #table_mat
# treemat <- confusionMatrix(predict_tree,as.factor(y_valid$y))
# treemat
# misclassificationrate <- as.numeric(paste(unlist(treemat$overall[1])))
# misclassificationrate
# 
# #AUC
# pred_prob <- predict(fit, x_valid, type = 'prob')[,2] #prob of success
# ROC_cftree <- roc(y_valid$y ~ pred_prob, plot=TRUE, print.auc=TRUE, legacy.axes=TRUE)
# ROC_cftree$auc

#classification tree function
cartmodel = function(train, valid) {
  x_validset <- lapply(valid, "[", 1:10)
  y_validset <- lapply(valid, "[", 11)
  
  cartfunction = function(i) {
    rpart(y ~ x1+x2+x3+x4+x5+x6+x7+x8+x9+x10, data=i , method='class')
  }
  cartmod <- lapply(train, cartfunction)
  
  summary(cartmod[[1]]) #summary of first simulation and feature importance
  rpart.plot(cartmod[[1]])
  
  cartpredprob <- Map(function(s, l) predict(s, newdata=l, type = 'prob')[,2], cartmod, x_validset)
  cartpredclass <- Map(function(s, l) predict(s, newdata=l, type = 'class'), cartmod, x_validset)
  
  actualclass = function(x) {
    as.factor(x[[1]])
  }
  actual <- lapply(y_validset, actualclass) 
  
  cartconfusionmatrix <- Map(function(s, l) confusionMatrix(s,l), cartpredclass, actual)
  misclassificationrate <- as.numeric(paste(unlist(lapply(cartconfusionmatrix, function(x) x$overall[1]))))
  roc <-  Map(function(s, l) roc(s,l), actual, cartpredprob)
  auc <- as.numeric(paste(unlist(lapply(roc, auc)))) 
  
  print(cartconfusionmatrix[[1]]) #confusion matrix of first simulation
  
  #plot ROC curve for all simulations
  color <- c(rep("#ff000010", rep/2-1), rep("#0000ff10", rep/2))
  plot(roc[[1]], legacy.axes=TRUE, main="ROC curve", xlim=c(1,0)) 
  for (i in 2:rep){
    lines(roc[[i]], type="l", col=color[i])
  }
  text(0,0.6,labels=paste("Avg AUC=",round(mean(auc),4)))
  
  #create output for comparisons
  output <- list("misclassification rate", misclassificationrate, 
                 "avg misclassification rate", mean(misclassificationrate), 
                 "range of misclassification rate", range(misclassificationrate),
                 "auc score", auc, 
                 "avg auc score", mean(auc),
                 "range of auc score", range(auc))
  return(output)
}

#call functions
cart_scenario1 <- cartmodel(fulltrainset,fullvalidset)
cart_scenario2 <- cartmodel(missing15removedtrainset, missing15validset)
cart_scenario3 <- cartmodel(missing15meantrainset, missing15meanvalidset)
cart_scenario4 <- cartmodel(missing15trainset, missing15validset) #automatic handling of cart impute
cart_scenario5 <- cartmodel(missing30removedtrainset, missing30validset)
cart_scenario6 <- cartmodel(missing30meantrainset, missing30meantrainset)
cart_scenario7 <- cartmodel(missing30trainset, missing30validset) #automatic handling of cart impute

#add results to dataframe
model_performance <- append(model_performance, c("Cart: Avg_Accuracy", "Cart: Avg_AUC"))
true_model <- append(true_model, c(round(unlist(cart_scenario1[4]),4), round(unlist(cart_scenario1[10]),4)))
remove_0.15 <- append(remove_0.15, c(round(unlist(cart_scenario2[4]),4), round(unlist(cart_scenario2[10]),4)))
mean_0.15 <- append(mean_0.15, c(round(unlist(cart_scenario3[4]),4), round(unlist(cart_scenario3[10]),4)))
cart_0.15 <- append(cart_0.15,c(round(unlist(cart_scenario4[4]),4), round(unlist(cart_scenario4[10]),4)))
remove_0.3 <- append(remove_0.3, c(round(unlist(cart_scenario5[4]),4), round(unlist(cart_scenario5[10]),4)))
mean_0.3 <- append(mean_0.3, c(round(unlist(cart_scenario6[4]),4), round(unlist(cart_scenario6[10]),4)))
cart_0.3 <- append(cart_0.3, c(round(unlist(cart_scenario7[4]),4), round(unlist(cart_scenario7[10]),4)))
df_results2 <- data.frame(model_performance, true_model ,remove_0.15, mean_0.15, cart_0.15, remove_0.3, mean_0.3, cart_0.3)
df_results2

######################## Random Forest ###############################
library(randomForest)

# firstsim <- data.frame(sim[[1]])
# trainID = sample(1:1000,700)
# training <- firstsim[trainID,]
# validate <- firstsim[-trainID,]
# x_valid <- subset(validate, select = -c(y))
# y_valid <- validate %>% select(y)
# 
# output.forest <- randomForest(as.factor(y) ~ x1+x2+x3+x4+x5+x6+x7+x8+x9+x10, data = training)
# print(output.forest)
# plot(output.forest)
# print(importance(output.forest))
# 
# predict_rf <- predict(output.forest, x_valid)
# rfmat <- confusionMatrix(predict_rf,as.factor(y_valid$y))
# rfmat
# misclassificationrate <- as.numeric(paste(unlist(rfmat$overall[1])))
# misclassificationrate
# 
# #AUC
# pred_probtree <- as.numeric(predict(output.forest, x_valid, type = 'prob')[,2]) #prob of success
# ROC_cftree <- roc(y_valid$y ~ pred_probtree, plot=TRUE, print.auc=TRUE, legacy.axes=TRUE)
# ROC_cftree$auc

#random forest function
rfmodel = function(train, valid) {
  x_validset <- lapply(valid, "[", 1:10)
  y_validset <- lapply(valid, "[", 11)
  
  rffunction = function(i) {
    randomForest(as.factor(y) ~ x1+x2+x3+x4+x5+x6+x7+x8+x9+x10, data = i, ntree=20)
  }
  rfmod <- lapply(train, rffunction)
  
  print(rfmod[[1]]) #summary of first simulation
  #plot(rfmod[[1]]) #error vs trees
  print(importance(rfmod[[1]])) #variable importance
  
  rfpredprob <- Map(function(s, l) predict(s, newdata=l, type = 'prob')[,2], rfmod, x_validset)
  rfpredprob <- lapply(rfpredprob, as.numeric)
  
  rfpredclass <- Map(function(s, l) predict(s, newdata=l, type = 'class'), rfmod, x_validset)
  
  convertfactor = function(x) {
    as.factor(x[[1]])
  }
  actual <- lapply(y_validset, convertfactor) 
  
  rfconfusionmatrix <- Map(function(s, l) confusionMatrix(s,l), rfpredclass, actual)
  misclassificationrate <- as.numeric(paste(unlist(lapply(rfconfusionmatrix, function(x) x$overall[1]))))
  roc <-  Map(function(s, l) roc(s,l), actual, rfpredprob)
  auc <- as.numeric(paste(unlist(lapply(roc, auc)))) 
  
  #print(cartconfusionmatrix[[1]]) #confusion matrix of first simulation
  
  #plot ROC curve for all simulations
  color <- c(rep("#ff000010", rep/2-1), rep("#0000ff10", rep/2))
  plot(roc[[1]], legacy.axes=TRUE, main="ROC curve", xlim=c(1,0)) 
  for (i in 2:rep){
    lines(roc[[i]], type="l", col=color[i])
  }
  text(0,0.6,labels=paste("Avg AUC=",round(mean(auc),4)))
  
  #create output for comparisons
  output <- list("misclassification rate", misclassificationrate, 
                 "avg misclassification rate", mean(misclassificationrate), 
                 "range of misclassification rate", range(misclassificationrate),
                 "auc score", auc, 
                 "avg auc score", mean(auc),
                 "range of auc score", range(auc))
  return(output)
}

#call functions
rf_scenario1<- rfmodel(fulltrainset,fullvalidset)
rf_scenario2 <- rfmodel(missing15removedtrainset, missing15validset)
rf_scenario3 <- rfmodel(missing15meantrainset, missing15meanvalidset)
rf_scenario4 <- rfmodel(missing15carttrainset, missing15cartvalidset)
rf_scenario5 <- rfmodel(missing30removedtrainset, missing30validset)
rf_scenario6 <- rfmodel(missing30meantrainset, missing30meanvalidset)
rf_scenario7 <- rfmodel(missing30carttrainset, missing30cartvalidset)

#add results to dataframe
model_performance <- append(model_performance, c("Random Forest: Avg_Accuracy", "Random Forest: Avg_AUC"))
true_model <- append(true_model, c(round(unlist(rf_scenario1[4]),4), round(unlist(rf_scenario1[10]),4)))
remove_0.15 <- append(remove_0.15, c(round(unlist(rf_scenario2[4]),4), round(unlist(rf_scenario2[10]),4)))
mean_0.15 <- append(mean_0.15, c(round(unlist(rf_scenario3[4]),4), round(unlist(rf_scenario3[10]),4)))
cart_0.15 <- append(cart_0.15,c(round(unlist(rf_scenario4[4]),4), round(unlist(rf_scenario4[10]),4)))
remove_0.3 <- append(remove_0.3, c(round(unlist(rf_scenario5[4]),4), round(unlist(rf_scenario5[10]),4)))
mean_0.3 <- append(mean_0.3, c(round(unlist(rf_scenario6[4]),4), round(unlist(rf_scenario6[10]),4)))
cart_0.3 <- append(cart_0.3, c(round(unlist(rf_scenario7[4]),4), round(unlist(rf_scenario7[10]),4)))
df_results3 <- data.frame(model_performance, true_model ,remove_0.15, mean_0.15, cart_0.15, remove_0.3, mean_0.3, cart_0.3)
df_results3

df_results3[c(2,4,6),] #AUC


############################compare with all models and methods#################################
logmodel_true <- unlist(log_scenario1[8])
logmodel_0.15_remove <- unlist(log_scenario2[8])
logmodel_0.15_mean <- unlist(log_scenario3[8])
logmodel_0.15_cart <- unlist(log_scenario4[8])
logmodel_0.3_remove <- unlist(log_scenario5[8])
logmodel_0.3_mean <- unlist(log_scenario6[8])
logmodel_0.3_cart <- unlist(log_scenario7[8])
cartmodel_true <- unlist(cart_scenario1[8])
cartmodel_0.15_remove <- unlist(cart_scenario2[8])
cartmodel_0.15_mean <- unlist(cart_scenario3[8])
cartmodel_0.15_cart <- unlist(cart_scenario4[8])
cartmodel_0.3_remove <- unlist(cart_scenario5[8])
cartmodel_0.3_mean <- unlist(cart_scenario6[8])
cartmodel_0.3_cart <- unlist(cart_scenario7[8])
rfmodel_true <- unlist(rf_scenario1[8])
rfmodel_0.15_remove <- unlist(rf_scenario2[8])
rfmodel_0.15_mean <- unlist(rf_scenario3[8])
rfmodel_0.15_cart <- unlist(rf_scenario4[8])
rfmodel_0.3_remove <- unlist(rf_scenario5[8])
rfmodel_0.3_mean <- unlist(rf_scenario6[8])
rfmodel_0.3_cart <- unlist(rf_scenario7[8])

bigdf_auc <- data.frame(
  logmodel_true, logmodel_0.15_remove, logmodel_0.15_mean, logmodel_0.15_cart, logmodel_0.3_remove, logmodel_0.3_mean, logmodel_0.3_cart,
  cartmodel_true, cartmodel_0.15_remove, cartmodel_0.15_mean, cartmodel_0.15_cart, cartmodel_0.3_remove, cartmodel_0.3_mean, cartmodel_0.3_cart,
  rfmodel_true, rfmodel_0.15_remove, rfmodel_0.15_mean, rfmodel_0.15_cart, rfmodel_0.3_remove, rfmodel_0.3_mean, rfmodel_0.3_cart
)

#############################AUC Box plot of Logistic Model
boxplot(bigdf_auc[,1:7], main="Logistic Regression Comparison of AUC scores", 
        xlab="Models", 
        ylab="AUC",
        col = c("green","purple","cyan", "magenta", "purple", "cyan", "magenta"),
        cex.axis=0.70,
        ylim = c(0.6, 0.9))

legend("bottomleft", inset=.02, title="Missing Value Methods",
       c("Removing rows","Imputing Mean","Cart Impute"), fill=c("purple", "cyan", "magenta"), horiz=TRUE, cex=0.8)

#AUC Box plot of CART
boxplot(bigdf_auc[,8:14], main="CART Comparison of AUC scores", 
        xlab="Models", 
        ylab="AUC",
        col = c("green","purple","cyan", "magenta", "purple", "cyan", "magenta"),
        cex.axis=0.70,
        ylim = c(0.55, 0.9))

legend("bottomleft", inset=.02, title="Missing Value Methods",
       c("Removing rows","Imputing Mean","Cart Impute"), fill=c("purple", "cyan", "magenta"), horiz=TRUE, cex=0.8)

#AUC Box plot of Random Forest
boxplot(bigdf_auc[,15:21], main="Random Forest Comparison of AUC scores", 
        xlab="Models", 
        ylab="AUC",
        col = c("green","purple","cyan", "magenta", "purple", "cyan", "magenta"),
        cex.axis=0.70,
        ylim = c(0.85, 1))

legend("bottomleft", inset=.02, title="Missing Value Methods",
       c("Removing rows","Imputing Mean","Cart Impute"), fill=c("purple", "cyan", "magenta"), horiz=TRUE, cex=0.8)

##################################Box plot of ALL models
bymedian <- bigdf_auc[order(sapply(-bigdf_auc, median))]

boxplot(bymedian, main="Model Comparison of AUC scores", 
        #ylab="AUC", 
        col = c("yellow", "yellow", "yellow", "yellow", "yellow", "yellow", "yellow",
                "green", "green", "green", "green", "green", "orange", "orange",
                "green", "green", "orange", "orange", "orange", "orange", "orange"),
        cex.axis=0.75,
        ylim = c(0.65, 1),
        las=2,
        par(mar=c(8,3,3,1))
)

legend("bottomleft", inset=.001, title="Classification Methods",
       c("Random Forest","Logistic Regression", "CART"), fill=c("yellow", "green", "orange"), horiz=TRUE, cex=0.6)

###########################Proportion of times true model is better than the other
#Logistic Regression
names=c("remove_0.15", "mean_0.15", "cart_0.15", "remove_0.3", "mean_0.3", "cart_0.3")
model1diff1 <- c( sum(ifelse(bigdf_auc$logmodel_true>bigdf_auc$logmodel_0.15_remove, 1, 0))/nrow(bigdf_auc),
                  sum(ifelse(bigdf_auc$logmodel_true>bigdf_auc$logmodel_0.15_mean, 1, 0))/nrow(bigdf_auc), 
                  sum(ifelse(bigdf_auc$logmodel_true>bigdf_auc$logmodel_0.15_cart, 1, 0))/nrow(bigdf_auc), 
                  sum(ifelse(bigdf_auc$logmodel_true>bigdf_auc$logmodel_0.3_remove, 1, 0))/nrow(bigdf_auc),
                  sum(ifelse(bigdf_auc$logmodel_true>bigdf_auc$logmodel_0.3_mean, 1, 0))/nrow(bigdf_auc),
                  sum(ifelse(bigdf_auc$logmodel_true>bigdf_auc$logmodel_0.3_cart, 1, 0))/nrow(bigdf_auc))
model1_diffauc1 <- data.frame(names, model1diff1)
barplot(model1_diffauc1$model1diff1, 
        names.arg=model1_diffauc1$names, 
        xlab="Models", 
        ylim=c(0,1),
        col = c("blue","purple", "red", "blue", "purple", "red"),
        main="Logistic Regression: Proportion of Times True Full Model is better")

#CART 
model1diff2 <- c( sum(ifelse(bigdf_auc$cartmodel_true>bigdf_auc$cartmodel_0.15_remove, 1, 0))/nrow(bigdf_auc),
                  sum(ifelse(bigdf_auc$cartmodel_true>bigdf_auc$cartmodel_0.15_mean, 1, 0))/nrow(bigdf_auc), 
                  sum(ifelse(bigdf_auc$cartmodel_true>bigdf_auc$cartmodel_0.15_cart, 1, 0))/nrow(bigdf_auc), 
                  sum(ifelse(bigdf_auc$cartmodel_true>bigdf_auc$cartmodel_0.3_remove, 1, 0))/nrow(bigdf_auc),
                  sum(ifelse(bigdf_auc$cartmodel_true>bigdf_auc$cartmodel_0.3_mean, 1, 0))/nrow(bigdf_auc),
                  sum(ifelse(bigdf_auc$cartmodel_true>bigdf_auc$cartmodel_0.3_cart, 1, 0))/nrow(bigdf_auc))
model1_diffauc2 <- data.frame(names, model1diff2)
barplot(model1_diffauc2$model1diff2, 
        names.arg=model1_diffauc2$names, 
        xlab="Models", 
        ylim=c(0,1),
        col = c("blue","purple", "red", "blue", "purple", "red"),
        main="CART: Proportion of Times True Full Model is better")

#Random Forest
model1diff3 <- c( sum(ifelse(bigdf_auc$rfmodel_true>bigdf_auc$rfmodel_0.15_remove, 1, 0))/nrow(bigdf_auc),
                  sum(ifelse(bigdf_auc$rfmodel_true>bigdf_auc$rfmodel_0.15_mean, 1, 0))/nrow(bigdf_auc), 
                  sum(ifelse(bigdf_auc$rfmodel_true>bigdf_auc$rfmodel_0.15_cart, 1, 0))/nrow(bigdf_auc), 
                  sum(ifelse(bigdf_auc$rfmodel_true>bigdf_auc$rfmodel_0.3_remove, 1, 0))/nrow(bigdf_auc),
                  sum(ifelse(bigdf_auc$rfmodel_true>bigdf_auc$rfmodel_0.3_mean, 1, 0))/nrow(bigdf_auc),
                  sum(ifelse(bigdf_auc$rfmodel_true>bigdf_auc$rfmodel_0.3_cart, 1, 0))/nrow(bigdf_auc))
model1_diffauc3 <- data.frame(names, model1diff3)
barplot(model1_diffauc3$model1diff3, 
        names.arg=model1_diffauc3$names, 
        xlab="Models", 
        ylim=c(0,1),
        col = c("blue","purple", "red", "blue", "purple", "red"),
        main="Random Forest: Proportion of Times True Full Model is better")

legend("topright", title="Missing Value Methods",
       c("Removing rows","Imputing Mean","Cart Impute"), fill=c("blue", "purple", "red"), horiz=TRUE, cex=0.8)

#Comparing ROC curve


df_results3[c(2,4,6),]



