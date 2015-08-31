library(readr)
library(randomForest)

set.seed(1)

cat("reading the train and test data\n")
train <- read_csv("../input/train.csv")
test  <- read_csv("../input/test.csv")

feature.names <- names(train)[2:ncol(train)-1]

cat("assuming text variables are categorical & replacing them with numeric ids\n")
for (f in feature.names) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}

cat("replacing missing values with -1\n")
train[is.na(train)] <- -1
test[is.na(test)]   <- -1

cat("sampling train to get around memory limitations\n")
train <- train[sample(nrow(train), 40000),]
gc()

cat("training a Random Forest classifier\n")
clf <- randomForest(train[,feature.names], factor(train$target), ntree=40, sampsize=5000, nodesize=2)

cat("making predictions in batches due to 8GB memory limitation\n")
submission <- data.frame(ID=test$ID)
submission$target <- NA 
for (rows in split(1:nrow(test), ceiling((1:nrow(test))/10000))) {
    submission[rows, "target"] <- predict(clf, test[rows,feature.names], type="prob")[,2]
}

cat("saving the submission file\n")
write_csv(submission, "random_forest_submission.csv")