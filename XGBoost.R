train <- read.csv('Full Set Train.csv')
test <- read.csv('Full Set Test.csv')

train$Summary <- paste(train$Summary,train$Priority)
train$Summary <- paste(train$Summary,train$Component)

test$Summary <- paste(test$Summary,test$Priority)	#Toggle
test$Summary <- paste(test$Summary,test$Component)	#Toggle


library(DMwR)
train <- SMOTE(Severity ~ ., train, perc.under = 100, perc.over = 500)		#Toggle

#test$Severity <- NA
combi <- rbind(train,test)

library(NLP)
library(tm)
corpus <- Corpus(VectorSource(combi$Summary))
corpus <- tm_map(corpus,Boost_tokenizer)
corpus <- tm_map(corpus,tolower)
corpus <- tm_map(corpus,removePunctuation)
corpus <- tm_map(corpus,removeWords,c(stopwords('english')))
corpus <- tm_map(corpus,stripWhitespace)
corpus <- tm_map(corpus,stemDocument)
corpus <- tm_map(corpus,PlainTextDocument)

corpus <- Corpus(VectorSource(combi$Summary))
frequencies <- DocumentTermMatrix(corpus)

freq <- colSums(as.matrix(frequencies))
ord <- order(freq)

sparse <- removeSparseTerms(frequencies,1-10/nrow(frequencies))

newsparse <- as.data.frame(as.matrix(sparse))
colnames(newsparse) <- make.names(colnames(newsparse))

newsparse$Severity <- as.factor(c(combi$Severity))

mytrain <- newsparse[1:nrow(train),]
mytest <- newsparse[-(1:nrow(train)),]

library(xgboost)
library(Matrix)

labels <- mytrain$Severity
labels <- as.numeric(labels)-1

tdata = as.matrix(mytrain[,!colnames(mytrain) %in% c('Severity')])

ctrain <- xgb.DMatrix(data = tdata,label = labels)
dtest <- xgb.DMatrix(Matrix(data.matrix(mytest[,!colnames(mytest) %in% c('Severity')])))
watchlist <- list(train = ctrain, test = dtest)

xgbmodel <- xgboost(data = ctrain, max.depth = 25, eta = 0.3, nround = 200, objective = "multi:softmax", num_class = 7, verbose = 1, watchlist)
xgbmodel.predict <- predict(xgbmodel,newdata = data.matrix(mytest[,!colnames(mytest) %in% c('Severity')]))
xgbmodel.predict.text <- levels(mytrain$Severity)[xgbmodel.predict+1]

xgbmodel2 <- xgboost(data = ctrain, max.depth = 20, eta = 0.2, nrounds = 250, objective = "multi:softmax", num_class = 7, watchlist)
xgbmodel.predict2 <- predict(xgbmodel2, newdata = data.matrix(mytest[, !colnames(mytest) %in% c('Severity')])) 
xgbmodel.predict2.text <- levels(mytrain$Severity)[xgbmodel.predict2 + 1]

xgbmodel3 <- xgboost(data = ctrain, max.depth = 25, eta = 0.1, nround = 250, objective = "multi:softmax", num_class = 7, verbose = 2,watchlist)
xgbmodel.predict3 <- predict(xgbmodel3, newdata = data.matrix(mytest[, !colnames(mytest) %in% c('Severity')])) 
xgbmodel.predict3.text <- levels(mytrain$Severity)[xgbmodel.predict3 + 1]


sum(diag(table(mytest$Severity, xgbmodel.predict)))/nrow(mytest)
sum(diag(table(mytest$Severity, xgbmodel.predict2)))/nrow(mytest)
sum(diag(table(mytest$Severity, xgbmodel.predict3)))/nrow(mytest)

library(data.table)

submit_match1 <- cbind(as.data.frame(test$Bug.ID), as.data.frame(xgbmodel.predict.text))
colnames(submit_match1) <- c('Bug.ID','Severity')
submit_match1 <- data.table(submit_match1, key = 'Bug.ID')

submit_match2 <- cbind(as.data.frame(test$Bug.ID), as.data.frame(xgbmodel.predict2.text))
colnames(submit_match2) <- c('Bug.ID','Severity')
submit_match2 <- data.table(submit_match2, key = 'Bug.ID')

submit_match3 <- cbind(as.data.frame(test$Bug.ID), as.data.frame(xgbmodel.predict3.text))
colnames(submit_match3) <- c('Bug.ID','Severity')
submit_match3 <- data.table(submit_match3, key = 'Bug.ID')

submit_match3$Severity2 <- submit_match2$Severity 
submit_match3$Severity1 <- submit_match1$Severity

Mode <- function(x) {
  u <- unique(x)
  u[which.max(tabulate(match(x, u)))]
}

x <- Mode(submit_match3[,c('Severity','Severity2','Severity1')])
y <- apply(submit_match3,1,Mode)
final_submit <- data.frame(Bug.ID= submit_match3$Bug.ID, Severity = y)

data.table(final_submit)
write.csv(final_submit, 'D:/VIT/Academics/CSE499 Project Work/Data/VerifiedFixed/ensemble.csv', row.names = FALSE)