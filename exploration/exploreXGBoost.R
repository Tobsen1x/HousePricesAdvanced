data <- read.csv('data/train.csv', header = TRUE, stringsAsFactors = TRUE, quote = '')
rawX <- select(data, -Id, -SalePrice)
rawY <- select(data, SalePrice)

xgbGrid <- expand.grid(nrounds = c(10, 50, 100, 250),
                       max_depth = c(3, 4, 5),
                       eta = c(.1, .3),
                       gamma = 0,
                       colsample_bytree = .8,
                       min_child_weight = 1)

cvControl <- trainControl(method = "cv", number = 3)

xToImputeBool <- colSums(is.na(rawX)) == 0
xToImpute <- rawX[, xToImputeBool]
impXTrain <- preProcess(xToImpute, method = 'knnImpute', k = 10)
#impX <- 

xgBoostTrain <- tuneModelFacade(x = rawX, y = rawY, trControl = cvControl, tuneGrid = xgbGrid)
