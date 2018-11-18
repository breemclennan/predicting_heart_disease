# Machine Learning Algorithm Evaluation
# https://machinelearningmastery.com/machine-learning-in-r-step-by-step/

# ==================================================== #
# PART 4
# ==================================================== #

# Now it is time to create some models of the data and estimate their accuracy on unseen data.

#Here is what we are going to cover in this step:

#  Set-up the test harness to use 10-fold cross validation.
#Build 5 different models to predict species from flower measurements
#Select the best model.

# ============== #
# 4.1 Test Harness
# ============== #

# We will perform 10-fold crossvalidation to estimate accuracy.
# this will split our dataset into 10 parts, train in 9 and test on 1 and release for all combinations of train-test splits.
# We will also repeat the process 3 times for each algorithm with different splits of the data into 10 groups, 
# in an effort to get a more accurate estimate.

# Run algorithms using 10-fold cross validation
control <- trainControl(method="cv", number=10)

#, classProbs=TRUE, savePredictions=TRUE, summaryFunction = LogLoss)
metric <- "Accuracy"
#metric <- "LogLoss"


# EXPLORE DIFFERENT METHODS WITH DIFFERENT TUNING PARAMETERS AND COMPARE MODEL PERFORMANCE =====
AUC = list()
Accuracy = list()
LogLoss = list()

# We are using the metric of “Accuracy” to evaluate models.
# This is a ratio of the number of correctly predicted instances in divided by the total number of instances in the dataset
# multiplied by 100 to give a percentage (e.g. 95% accurate).
# We will be using the metric variable when we run build and evaluate each model next.

# ============== #
# 4.2 DATASET SETUP
# ============== #

# TODO:
# DO WE HAVE ANY NEAR-ZERO VARIANCE VARS? REMOVE THEM?

# MODEL: DATASET SETUP =======================================================================
# split the exploration set back into original datasets(with the new features created)
mod.training_00 <- wrk.ObservationCheckSet_01 %>%
  filter(CATDataSetOrigin == "Training") %>%
  select(-CATDataSetOrigin, -patient_id) %>% # removed index/patient id column
  mutate(heart_disease_present = as.factor(heart_disease_present))  #  recode target vaiable as FACTOR

str(mod.training_00)

mod.FINAL_Test_00 <- wrk.ObservationCheckSet_01 %>% # USE THIS SET FOR FINAL PREDICTIONS AND SUBMISSION ----
filter(CATDataSetOrigin == "Testing") %>%
  select(-CATDataSetOrigin) # and keep patient id for later

# TRAINING AND TESTING DATA FOR VALIDATION
# SPLITTING THE Training set into Train (70%) and Validation/test (30%)
library(caret)
set.seed(10)
inTrainRows <- createDataPartition(mod.training_00$heart_disease_present,p=0.7,list=FALSE)
trainData <- mod.training_00[inTrainRows,]
testData <-  mod.training_00[-inTrainRows,]
nrow(trainData)/(nrow(testData)+nrow(trainData)) #checking whether really 70% -> OK

# ============== #
# 4.3 BUILD MODELS
# ============== #

#We don’t know which algorithms would be good on this problem or what configurations to use. 
#We get an idea from the plots that some of the classes are partially linearly separable in some dimensions,
#so we are expecting generally good results.

#Let’s evaluate 5 different algorithms:

# 1. Linear Discriminant Analysis (LDA)
# 2. Classification and Regression Trees (CART).
# 3. k-Nearest Neighbors (kNN).
# 4. Support Vector Machines (SVM) with a linear kernel.
# 5. Random Forest (RF)

#This is a good mixture of simple linear (LDA), nonlinear (CART, kNN) and complex nonlinear methods (SVM, RF).
# We reset the random number seed before reach run to ensure that the evaluation of each algorithm is performed using exactly the same data splits. 
# It ensures the results are directly comparable.

# Let’s build our five models:
# 1) linear algorithms
# LDA
set.seed(7)
fit.lda <- train(heart_disease_present~., data=trainData, method="lda", metric=metric, trControl=control)
# 2) nonlinear algorithms
# CART
set.seed(7)
fit.cart <- train(heart_disease_present~., data=trainData, method="rpart", metric=metric, trControl=control)
# 3) kNN
set.seed(7)
fit.knn <- train(heart_disease_present~., data=trainData, method="knn", metric=metric, trControl=control)
# 4) advanced algorithms
# SVM # TODO: CONFIGURE THIS TO WORK WITH PROBABILITIES
set.seed(7)
fit.svm <- train(as.factor(heart_disease_present)~., data=trainData, method="svmRadial", metric=metric, trControl = trainControl(method = "repeatedcv", repeats = 10, 
                                                                                                                                 classProbs =  TRUE), preProc = c("center", "scale"))
# 5) advanced algorithms
#Random Forest
set.seed(7)
fit.rf <- train(heart_disease_present~., data=trainData, method="rf", metric=metric, trControl=control)
# Advanced ML (deep learning)
library(MASS)
fit.nnet <- train(heart_disease_present~., data=trainData, method = "nnet", metric=metric, trControl=control)
# Best general performing algorithms: 
# https://datascience.stackexchange.com/questions/10745/which-of-the-180-algorithms-in-rs-caret-package-are-feasible
# more detail on xgboost: http://mlr-org.github.io/How-to-win-a-drone-in-20-lines-of-R-code/

#fit.caret.xgboost <- train(heart_disease_present~., data=trainData, method = "xgbDART", metric=metric, trControl=control)
# Stochastic gradient boosting machine
fit.gbm <- train(heart_disease_present~., data=trainData, method = "gbm", metric=metric, trControl=control)

# Gradient Boosting machine (H2O)
#library(h2o)
#fit.gbm.h2O <- train(heart_disease_present~., data=trainData, method = "gbm_h2o", metric=metric, trControl=control)

# Model Averaged Neural Network
fit.avnnet <- train(heart_disease_present~., data=trainData, method = "avNNet", metric=metric, trControl=control)

#Tree-Based ensembles
library(nodeHarvest)
fit.nodeHarvest <- train(heart_disease_present~., data=trainData, method = "nodeHarvest", metric=metric, trControl=control)

# Parallel random forest
library(import)
fit.parRF <- train(heart_disease_present~., data=trainData, method = "parRF", metric=metric, trControl=control)

# Learning Vector Quantization
fit.lvq <- train(heart_disease_present~., data=trainData, method = "lvq", metric=metric, trControl=control)

fit.glm <- train(heart_disease_present~., data=trainData, method = "glm", metric=metric, trControl=control)
library(mboost)
fit.glmboost <- train(heart_disease_present~., data=trainData, method = "glmboost", metric=metric, trControl=control)
fit.glmnet   <- train(heart_disease_present~., data=trainData, method = "glmnet", metric=metric, trControl=control)
library(h2o)
fit.glmnet_h2o   <- train(heart_disease_present~., data=trainData, method = "glmnet_h2o", metric=metric, trControl=control)

# Caret does support the configuration and tuning of the configuration of each model, we will explore this later.

# ============== #
# 4.4 SELECT THE BEST MODEL
# ============== #
#We now have 5 models and accuracy estimations for each. 
#We need to compare the models to each other and select the most accurate.
# We can report on the accuracy of each model by first creating a list of the created models and using the summary function.

results <- resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf, nnet=fit.nnet,
                          gbm=fit.gbm, avnnet=fit.avnnet, nodeharvest=fit.nodeHarvest, parRF=fit.parRF, lvq=fit.lvq,
                          glm=fit.glm, glmboost=fit.glmboost, glmnet=fit.glmnet, glmnet_h2o=fit.glmnet_h2o ))
summary(results)
# We can see the accuracy of each classifier and also other metrics like Kappa:

#We can also create a plot of the model evaluation results and compare the spread and the mean accuracy of each model. 
#There is a population of accuracy measures for each algorithm because each algorithm was evaluated 10 times (10 fold cross validation).

# compare accuracy of models
dotplot(results)
# We can see that the most accurate model in this case was LDA

# The results for just the LDA model can be summarized.
# summarize Best Model
print(fit.gbm)

# ============== #
# 5 Make Predictions
# ============== #
#The GBM was the most accurate model. Now we want to get an idea of the accuracy of the model on our validation set.

#This will give us an independent final check on the accuracy of the best model.
#It is valuable to keep a validation set just in case you made a slip during such as overfitting to the training set or a data leak. Both will result in an overly optimistic result.

#We can run the LDA model directly on the validation set and summarize the results in a confusion matrix.

#========== GBM ===================================== #
# estimate skill of GBM on the validation dataset
Prediction_GBM <- predict(fit.gbm, testData) # For confusion matrix
PredictionProb.gbm <- predict(fit.gbm, testData, type = "prob")[,2] # FOR probability outputs, [] for binary predcition, we will have 2 pred columns
confusionmat.gbm <- confusionMatrix(Prediction_GBM, testData[,"heart_disease_present"])

library(pROC)
AUC$GBM <- roc(as.numeric(testData$heart_disease_present),as.numeric(as.matrix((PredictionProb.gbm))))$auc
Accuracy$GBM <- confusionmat.gbm$overall['Accuracy']  
library(Metrics)
LogLoss$GBM <- logLoss(as.numeric(testData$heart_disease_present), as.numeric(as.matrix(PredictionProb.gbm)))
# AUC 0.93, ACC 0.87, LogLoss 0.7

# FOR SUBMISSION ===
# Predict using the test set
SUBMISSION_Prediction_GBM <- predict(fit.gbm, mod.FINAL_Test_00, type = 'prob')

# Save the solution to a dataframe with two columns: patient_id and heart_disease_present (prediction)
SOLUTION_GBM <- data.frame(patient_id = mod.FINAL_Test_00$patient_id, heart_disease_present = SUBMISSION_Prediction_GBM$`1`)

# Write the solution to file
write.csv(SOLUTION_GBM, file = 'submission_v6_R_GBM_20180930.csv', row.names = FALSE)

#========== LDA ===================================== #
# estimate skill of LDA on the validation dataset
Prediction_LDA <- predict(fit.lda, testData) # For confusion matrix
PredictionProb.lda <- predict(fit.lda, testData, type = "prob")[,2] # FOR probability outputs, [] for binary predcition, we will have 2 pred columns
confusionmat.lda <- confusionMatrix(Prediction_LDA, testData[,"heart_disease_present"])

library(pROC)
AUC$LDA <- roc(as.numeric(testData$heart_disease_present),as.numeric(as.matrix((PredictionProb.lda))))$auc
Accuracy$LDA <- confusionmat.lda$overall['Accuracy']  
library(Metrics)
LogLoss$LDA <- logLoss(as.numeric(testData$heart_disease_present), as.numeric(as.matrix(PredictionProb.lda)))
# AUC 0.94, ACC 0.90, LogLoss 0.58

#========== nodeHarvest ===================================== #
# estimate skill of nodeHarvest on the validation dataset
Prediction_NodeHarvest <- predict(fit.nodeHarvest, testData) # For confusion matrix
PredictionProb.nodeharvest <- predict(fit.nodeHarvest, testData, type = "prob")[,2] # FOR probability outputs, [] for binary predcition, we will have 2 pred columns
confusionmat.nodeharvest <- confusionMatrix(Prediction_NodeHarvest, testData[,"heart_disease_present"])

library(pROC)
AUC$NodeHarvest <- roc(as.numeric(testData$heart_disease_present),as.numeric(as.matrix((PredictionProb.nodeharvest))))$auc
Accuracy$NodeHarvest <- confusionmat.nodeharvest$overall['Accuracy']  
library(Metrics)
LogLoss$NodeHarvest <- logLoss(as.numeric(testData$heart_disease_present), as.numeric(as.matrix(PredictionProb.nodeharvest)))
# AUC 0.91, ACC 0.88, LogLoss 0.67

#========== SVM ===================================== #
# TODO: need SVM to output probabilities correctly [see previous usage in 00_dataload_explore.r]
# estimate skill of SVM on the validation dataset
Prediction_SVM <- predict(fit.svm, testData) # For confusion matrix
PredictionProb.SVM <- predict(fit.svm, testData, type = "prob")[,2] # FOR probability outputs, [] for binary predcition, we will have 2 pred columns
confusionmat.SVM<- confusionMatrix(Prediction_SVM, testData[,"heart_disease_present"])

library(pROC)
AUC$SVM <- roc(as.numeric(testData$heart_disease_present),as.numeric(as.matrix((PredictionProb.SVM))))$auc
Accuracy$SVM <- confusionmat.SVM$overall['Accuracy']  
library(Metrics)
LogLoss$SVM <- logLoss(as.numeric(testData$heart_disease_present), as.numeric(as.matrix(PredictionProb.SVM)))
# AUC 0.91, ACC 0.88, LogLoss 0.67

# ===============================================================================================================#
# MODEL EXPERIMENT RESULTS: COMPARING PERFORMANCE RESULTS =======================================================#
# ===============================================================================================================#
row.names <- names(Accuracy)
col.names <- c("AUC", "Accuracy","LogLoss")
cbind(as.data.frame(matrix(c(AUC,Accuracy,LogLoss),nrow = 3, ncol = 3, 
                           dimnames = list(row.names, col.names)))) ## make nrows and ncols bigger when more models created
