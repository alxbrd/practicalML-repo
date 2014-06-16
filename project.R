library(caret)

# Read datasets
training <- read.csv("pml-training.csv", header=TRUE)
test <- read.csv("pml-testing.csv", header=TRUE)

# See class of each column in the dataframe
lapply(training, class)

trainDescr<-training[,names(training)!="classe"]
                
# Convert factors to their underlying numeric (integer) levels
trainDescr <- data.matrix(training)
trainClass <- training$classe
testDescr <- data.matrix(test)
testClass <- test$classe

# Print total number of columns
ncol(trainDescr)
# Find correlations between variables
descrCorr <- cor(trainDescr)
# Convert NA to 0
descrCorr[is.na(descrCorr)] <- 0
# Find high correlations between variables
highCorr <- findCorrelation(descrCorr, 0.90)
# Remove highly correlated columns
trainDescr <- trainDescr[, -highCorr]
testDescr <- testDescr[, -highCorr]
# Print the new number of columns
ncol(trainDescr)



fitControl <- trainControl(## 10-fold CV
  method = "repeatedcv",
  number = 10,
  ## repeated ten times
  repeats = 10)

set.seed(825)
gbmFit1 <- train(Class ~ ., data = training,
                 method = "gbm",
                 trControl = fitControl,
                 ## This last option is actually one
                 ## for gbm() that passes through
                 verbose = FALSE)
gbmFit1






xTrans <- preProcess(trainDescr)

trainDescr <- predict(xTrans, trainDescr)
testDescr <- predict(xTrans, testDescr)


svmFit <- train(trainDescr, training, method = "svmRadial", tuneLength = 5, trControl = bootControl, scaled = FALSE)











library(party)
library(rpart)
rpartFull<-rpart(classe~.,data=training)
rpartPred<-predict(rpartFull,testing,type="class")
confusionMatrix(rpartPred,testing$Class)
                