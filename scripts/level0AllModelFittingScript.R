setwd('C:/RStudioWorkspace/HousePricesAdvanced')
source(file = 'C:/RStudioWorkspace/HousePricesAdvanced/scripts/initScripting.R', echo = FALSE, encoding = 'UTF-8')
loginfo('Fitting all Level 0 Models')
train.raw <- read.csv('data/train.csv',stringsAsFactors=FALSE)
test.raw <- read.csv('data/test.csv',stringsAsFactors=FALSE)

noVarImpData <- level0Preprocess(train.raw, test.raw, factorEncoding = 'ONEHOT', asPredictorMatrix = TRUE,
                                 resolveSkewness = TRUE)
varImpData <- level0Preprocess(train.raw, test.raw, nearZeroVars = TRUE, factorEncoding = 'ONEHOT', 
                                 asPredictorMatrix = TRUE, resolveSkewness = TRUE)

folds <- 20
xgbGrid1 <- expand.grid(nrounds = 1000,
                             max_depth = 4,
                             eta = 0.025,
                             gamma = 0,
                             colsample_bytree = 0.8,
                             min_child_weight = 1)
gbmGrid1 <- expand.grid(interaction.depth = 9,
                           n.trees = 900, 
                           shrinkage = 0.01,
                           n.minobsinnode = 10)
cubistGrid1 <- expand.grid(committees = 20, 
                           neighbors = 5)
rfGrid1 <- expand.grid(mtry = 48)
tuneGrids <- list('xgb1' = xgbGrid1, 'gbm1' = gbmGrid1, 'cubist1' = cubistGrid1, 'rf1' = rfGrid1)

modelList <- c('brnn2', 'gbm1', 'rf1', 'cubist1', 'lasso.9', 'xgb1', 'knn2', 'knn4', 'knn8', 'knn16', 'knn32')
level0Result <- fitWholeLevel(noVarImpData = noVarImpData, varImpData = varImpData, 
                              foldCount = folds, modelList = modelList, 
                              levelNr = 0, tuneGrids = tuneGrids)


fileName <- paste("models/level0/", length(modelList), '-', format(Sys.time(), "%d_%b_%Y_%H-%M-%S"), ".Rds", sep = '')
saveRDS(level0Result, file=fileName)
loginfo(paste('Saved Level 0 Models:', fileName))