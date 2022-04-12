train <- read.csv('Full Set.csv')
#test <- read.csv('Full Set Test.csv')
#Data_Bal$Severity <- as.factor(Data_Bal$Severity)
Data_Bal <- SMOTE(Severity ~ ., Data_Bal, perc.under = 100, perc.over = 800)		#Toggle

train$Summary <- paste(train$Summary,train$Priority)		#Toggle
train$Summary <- paste(train$Summary,train$Component)		#Toggle

training_data <- cbind.data.frame(train$Summary)
training_codes <- cbind.data.frame(train$Severity)
# testing_data <- cbind.data.frame(test$Summary)
# testing_codes <- cbind.data.frame(test$Severity)

matrix_train <- create_matrix(training_data, language = "english", removeNumbers = FALSE, stemWords = TRUE, removePunctuation = TRUE, removeStopwords = TRUE, stripWhitespace = TRUE, toLower = TRUE)
container_train <- create_container(matrix_train, t(training_codes), trainSize = 1:11040, testSize = 11041:nrow(train),virgin = FALSE)
# matrix_test <- create_matrix(testing_data, language = "english", removeNumbers = FALSE, stemWords = TRUE, removePunctuation = TRUE, removeStopwords = TRUE, stripWhitespace = TRUE, toLower = TRUE)
# container_test <- create_container(matrix_test, t(testing_codes), testSize = 1:nrow(test),virgin = FALSE)

models <- train_models(container_train, algorithms = "SVM")
results <- classify_models(container_train, models)
create_precisionRecallSummary(container_train, results, b_value = 1)

#prop.table(table(Data_Bal$Severity))
cross_validate(container_train,5,"SVM",seed=2)
