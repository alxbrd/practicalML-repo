Course Submission for Practical Machine Learning at Coursera
========================================================

The goal of this assignment is to create a model for predicting how well the lifting activity was performed by an athlete based on measurements from sensors placed on the arms, fore_arms, belt and dumbbells. The quality of execution was divided into five fashions (A-E) corresponding to the perfect execution (A) and common mistakes (B-E). We were provided with a dataset containing various measurements labelled with the correct executed movements (supervised problem).

Firstly, we loaded the necessary libraries and read the input dataset by specifying that both NA and "" must be interpreted as NA values. To make processing easier, we separated the available features from the response variable. Finally, we splitted the available dataset into training and validation set.


```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(randomForest)
```

```
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```r
data <- read.csv(file="pml-training.csv", head = TRUE, sep=",", na.strings=c("","NA"))
```

```
## Warning: cannot open file 'pml-training.csv': No such file or directory
```

```
## Error: cannot open the connection
```

```r
label <- data[, 160]  # Creates a factor vector for the indexing
```

```
## Error: object of type 'closure' is not subsettable
```

```r
label <- as.factor(label)
```

```
## Error: object 'label' not found
```

```r
features <- data[, 1:159]
```

```
## Error: object of type 'closure' is not subsettable
```

```r
set.seed(1)
inTrain <- createDataPartition(y = data$classe, p = 0.7, list = FALSE)
```

```
## Error: object of type 'closure' is not subsettable
```

```r
# Create the train and test feature set
training.f <- features[inTrain,]
```

```
## Error: object 'features' not found
```

```r
validation.f <- features[-inTrain,]
```

```
## Error: object 'features' not found
```

```r
# Create the train and validation label
training.label <- label[inTrain]
```

```
## Error: object 'label' not found
```

```r
validation.label <- label[-inTrain]
```

```
## Error: object 'label' not found
```

```r
# Get an idea about the content of the available feature set
# head(training.f)
```

After noticing that there are many NA values in the dataset, we remove the columns which contain more than 2/3 of NAs.


```r
# Returns the columns for which NAs are more than 2/3 of their values 
find_na_columns = function(dataset) {
        rows <- nrow(dataset)

        columns_names <- c()
        
        for (i in 1:dim(dataset)[2] - 1 ) {
                nas_col <- sum(is.na(dataset[,i])) # Number of NAs in the column
                if (nas_col > 0.667*rows) { # If NAs are more than 2/3 of the column 
                        columns_names <- c(columns_names, colnames(dataset[])[i])
                }
        }
        
        columns_names
}

remove_columns <- find_na_columns(training.f)
```

```
## Error: object 'training.f' not found
```

```r
training.f <- training.f[,-which(names(training.f) %in% remove_columns)]
```

```
## Error: object 'training.f' not found
```

```r
validation.f <- validation.f[,-which(names(validation.f) %in% remove_columns)]
```

```
## Error: object 'validation.f' not found
```

Then, we converted all the feature columns to numeric, and we remove the highly correlated features from the training and validation set.


```r
# Convert all columns to numeric
training.f <- as.data.frame(lapply(training.f, as.numeric))
```

```
## Error: object 'training.f' not found
```

```r
validation.f <- as.data.frame(lapply(validation.f, as.numeric))
```

```
## Error: object 'validation.f' not found
```

```r
# Remove highly correlated features
featureCorr <- cor(training.f)
```

```
## Error: object 'training.f' not found
```

```r
highCorr <- findCorrelation(featureCorr, 0.95)
```

```
## Error: object 'featureCorr' not found
```

```r
remove_columns2 <- lapply(highCorr, function(x) colnames(training.f[])[x])
```

```
## Error: object 'highCorr' not found
```

```r
training.f <- training.f[,-which(names(training.f) %in% remove_columns2)]
```

```
## Error: object 'training.f' not found
```

```r
validation.f <- validation.f[,-which(names(validation.f) %in% remove_columns2)]
```

```
## Error: object 'validation.f' not found
```

```r
# Get an idea about the content of the available feature set
# head(training.f)
plot(training.label, training.f$new_window)
```

```
## Error: object 'training.label' not found
```

```r
plot(training.label, training.f$X)
```

```
## Error: object 'training.label' not found
```

```r
plot(training.label, training.f$user_name)
```

```
## Error: object 'training.label' not found
```

Then, we remove the variables "user_name", "new_window", and "X" which do not seem helpful for predicting the five classes of interest. 


```r
# Remove useless variables
useless_features <- c("user_name", "new_window", "X") 
training.f <- training.f[,-which(names(training.f) %in% useless_features)]
```

```
## Error: object 'training.f' not found
```

```r
validation.f <- validation.f[,-which(names(validation.f) %in% useless_features)]
```

```
## Error: object 'validation.f' not found
```

```r
dim(training.f)[2] # Number of remaining features
```

```
## Error: object 'training.f' not found
```

As a last step of our preprocessing, we center and scale the values of the remaining features.


```r
# Center and scale data 
xTrans <- preProcess(training.f, method = c("center", "scale"))
```

```
## Error: object 'training.f' not found
```

```r
training.f <- predict(xTrans, training.f)      # Transformed training dataset
```

```
## Error: object 'xTrans' not found
```

```r
validation.f <- predict(xTrans, validation.f)       # Transformed validation dataset
```

```
## Error: object 'xTrans' not found
```

In the model selection step, we use 10-fold cross-validation for tuning the parameters of the created models.

```r
ctrl <- trainControl(method = "cv", number = 10, classProbs=TRUE)
```

We use two models: Random Forest and Support Vector Machine, for predicting the labels of our training dataset which are tuned based on the defined train control.

```r
#
# Random Forest model
#
model1 = train(training.f, # x - predictors
               training.label, # y - response
               method="rf", 
               trControl=ctrl, 
               #tuneGrid = rfGrid,
               # preProcess="pca",
               importance=TRUE)
```

```
## Error: object 'training.f' not found
```

```r
#
# Support Vector Machine model 
#
model2 <- train(training.f, # x - predictors
                 training.label, # y - response 
                 method = "svmRadial", 
                 tuneLength = 8, 
                 trControl = ctrl, 
                 metric = "Kappa",
                 preProcess="pca")
```

```
## Error: object 'training.f' not found
```

In the Model Validation step, we evaluate the performance of the created models on the validation step and we select as our final model the best performing one.


```r
#
# Select the best performing (tuned) model based on the validation set
#
find_accuracy = function(model){
        #plot(model)
        pred <- predict(model, newdata = validation.f)
        
        # confusion matrix (incl. confidence interval)
        confusionMatrix(data = pred, validation.label)$overall[[1]]   
} 

models <- c()
models[[1]] <- model1
```

```
## Error: object 'model1' not found
```

```r
models[[2]] <- model2
```

```
## Error: object 'model2' not found
```

```r
accuracy <- unlist(lapply(models, find_accuracy))
best_model <- models[which(accuracy == max(accuracy))]
```

```
## Warning: no non-missing arguments to max; returning -Inf
```

Finally, we use the selected model to predict the classes of the given testing dataset. Note that we performed to the testing set the same  preprocessing steps as with the training set.


```r
# Apply the same preprocessing to external testing dataset
data_evaluation <- read.csv(file="pml-testing.csv", head = TRUE)
```

```
## Warning: cannot open file 'pml-testing.csv': No such file or directory
```

```
## Error: cannot open the connection
```

```r
data_evaluation.labels <- data_evaluation[, 160]
```

```
## Error: object 'data_evaluation' not found
```

```r
data_evaluation.f <- data_evaluation[, 1:159]
```

```
## Error: object 'data_evaluation' not found
```

```r
data_evaluation.f <- data_evaluation.f[,-which(names(data_evaluation.f) %in% remove_columns)]
```

```
## Error: object 'data_evaluation.f' not found
```

```r
data_evaluation.f <- data_evaluation.f[,-which(names(data_evaluation.f) %in% useless_features)]
```

```
## Error: object 'data_evaluation.f' not found
```

```r
data_evaluation.f <- as.data.frame(lapply(data_evaluation.f, as.numeric))
```

```
## Error: object 'data_evaluation.f' not found
```

```r
data_evaluation.f <- data_evaluation.f[,-which(names(data_evaluation.f) %in% remove_columns2)]
```

```
## Error: object 'data_evaluation.f' not found
```

```r
dim(data_evaluation.f)
```

```
## Error: object 'data_evaluation.f' not found
```

```r
preProcValues <- preProcess(data_evaluation.f, method = c("center", "scale"))
```

```
## Error: object 'data_evaluation.f' not found
```

```r
dim(data_evaluation.f)
```

```
## Error: object 'data_evaluation.f' not found
```

```r
data_evaluation.f <- predict(preProcValues, data_evaluation.f)
```

```
## Error: object 'preProcValues' not found
```

```r
answers = predict(model1, newdata = data_evaluation.f)
```

```
## Error: object 'model1' not found
```

```r
answers = predict(best_model[[1]], newdata = data_evaluation.f)
```

```
## Error: no applicable method for 'predict' applied to an object of class
## "NULL"
```

```r
#
# Create answer files
#
pml_write_files = function(x){
        n = length(x)
        for(i in 1:n){
                filename = paste0("problem_id_",i,".txt")
                write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
        }
}

pml_write_files(answers)
```

```
## Error: object 'answers' not found
```
