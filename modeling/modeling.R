evaluatePrediction <- function(predictions) {
    modelPreds <- select(predictions, -c(id, y))
    allMetrics <- data.frame()
    for(actModelStr in colnames(modelPreds)) {
        actPreds <- modelPreds[,actModelStr]
        actRmsle <- rmsle(actual = predictions$y, predicted = actPreds)
        actR2 <- R2(pred = actPreds, obs = predictions$y)
        actRmse <- rmse(actual = predictions$y, predicted = actPreds)
        # Constructing list
        actMetrics <- data.frame('model' = actModelStr, 'rmsle' = actRmsle, 
                                 'r2' = actR2, 'rmse' = actRmse)
        if(nrow(allMetrics) == 0) {
            allMetrics <- actMetrics
        } else {
            allMetrics <- rbind(allMetrics, actMetrics)
        }
    }
    return(allMetrics)
}

trainLevelModels <- function(modelList, trainData, seed) {
    trainedModels <- list()
    for(actModelStr in names(modelList)) {
        actTrainParas <- modelList[[actModelStr]]
        #### GLMNET ####
        if(grepl('^lasso', actModelStr) |
           grepl('^ridge', actModelStr) |
           grepl('^elasticnet', actModelStr)) {
            set.seed(seed)
            actFit <- glmnet(x = trainData$predictors, y = trainData$y,
                             family = actTrainParas$family, 
                             alpha = actTrainParas$alpha, 
                             lambda = actTrainParas$lambda)
        } else if(actModelStr == 'rf') {
            set.seed(seed)
            actFit <- randomForest(x = trainData$predictors, y = trainData$y, 
                                   mtry = actTrainParas$mtry, 
                                   ntree = actTrainParas$ntree)
        }
        
        # Constructing list
        actModelList <- list(actFit, actTrainParas)
        names(actModelList) <- c(actModelStr, 'trainParas')
        if(length(trainedModels) == 0) {
            trainedModels <- list(actModelList)
        } else {
            trainedModels <- append(trainedModels, list(actModelList))
        }
    }
    names(trainedModels) <- names(modelList)
    return(trainedModels)
}

predictLevel <- function(models, testData) {
    preds <- data.frame('id' = testData$id)
    if(!is.null(testData$y)) {
        preds <- cbind(preds, 'y' = exp(testData$y))
    }
    
    for(actModelStr in names(models)) {
        actModel <- models[[actModelStr]][[actModelStr]]
        actParas <- models[[actModelStr]][['trainParas']]
        if(grepl('^lasso', actModelStr) |
           grepl('^ridge', actModelStr) |
           grepl('^elasticnet', actModelStr)) {
            p <- exp(predict(object = actModel, newx = testData$predictors, 
                             s = actParas$lambda.min, type = 'response'))
            preds <- cbind(preds, p)
        } else if(actModelStr == 'rf') {
            p <- exp(predict(object = actModel, newdata = testData$predictors, 
                             type = 'response'))
            preds <- cbind(preds, p)
        }
        colnames(preds)[length(colnames(preds))] <- actModelStr
    }
    return(preds)
}

foldData <- function(allData, foldArray, i) {
    testPreds <- allData$predictors[foldArray == i,]
    trainPreds <- allData$predictors[foldArray != i,]
    testY <- allData$y[foldArray == i]
    trainY <- allData$y[foldArray != i]
    testId <- allData$id[foldArray == i]
    trainId <- allData$id[foldArray != i]
    
    result <- list('train' = list('id' = trainId, 'y' = trainY, 'predictors' = trainPreds),
                   'test' = list('id' = testId, 'y' = testY, 'predictors' = testPreds))
    return(result)
}