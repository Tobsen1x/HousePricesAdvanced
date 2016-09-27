train.raw <- read.csv('data/train.csv',stringsAsFactors=FALSE)
test.raw <- read.csv('data/test.csv',stringsAsFactors=FALSE)

##### Data Preprocessing #####
ppL0 <- preprocL0(train = train.raw, test = test.raw, nzvRemove = TRUE, 
                  oneHot = TRUE, skewedRemoveBound = NA)
# Transform to Interface
# Remove Outliers #
outliers <- c(1299, 524, 822)
l0Data <- extractL0Data(ppL0, train.raw, outliers = outliers, asMatrix = TRUE)

alpha <- 1
lambda <- 0.003
enetFit <- glmnet(x = l0Data$train$predictors, y = l0Data$train$y,
                   family = 'gaussian', alpha = alpha, lambda = lambda)

enetTestPred <- exp(predict(object = enetFit, newx = l0Data$test$predictors, 
                             s = lambda, type = 'response'))
enetSub <- data.frame('Id' = l0Data$test$id, 'SalePrice' = enetTestPred[,1])
write.csv(enetSub, file="submissions/l0_lasso_5.csv",row.names=FALSE)
