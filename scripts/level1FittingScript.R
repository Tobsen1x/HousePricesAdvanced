setwd('C:/RStudioWorkspace/HousePricesAdvanced')
source(file = 'C:/RStudioWorkspace/HousePricesAdvanced/scripts/initScripting.R', echo = FALSE, encoding = 'UTF-8')
loginfo('Fitting Level 1 Models')
train.raw <- read.csv('data/train.csv',stringsAsFactors=FALSE)
test.raw <- read.csv('data/test.csv',stringsAsFactors=FALSE)

noVarImpData <- level0Preprocess(train.raw, test.raw, factorEncoding = 'ONEHOT', asPredictorMatrix = TRUE)
varImpData <- level0Preprocess(train.raw, test.raw, nearZeroVars = TRUE, factorEncoding = 'ONEHOT', 
                               asPredictorMatrix = TRUE)

level0Result <- readRDS('models/level0/allModelsLevel0_1.Rds')

level0Preds <- level0Result$cvPredictions
level0Models <- level0Result$finalModels
level0TestPreds <- level0Result$testPredictions

level1Data <- level1Preprocess(train = level0Preds, test = level0TestPreds)

folds <- 2
xgbGrid1 <- expand.grid(nrounds = 100 + (1:10)*100,
                        max_depth = c(4, 6, 8),
                        eta = c(0.2, 0.1, 0.05, 0.025),
                        gamma = 0,
                        colsample_bytree = 0.8,
                        min_child_weight = 1)
gbmGrid1 <- expand.grid(interaction.depth = c(6, 9),
                        n.trees = c(700, 900, 1100), 
                        shrinkage = c(0.05, 0.01, 0.005),
                        n.minobsinnode = 10)
cubistGrid1 <- expand.grid(committees = c(10, 20), 
                           neighbors = c(3, 5, 7))
rfGrid1 <- expand.grid(mtry = c(2, 4, 6, 8, 10))
tuneGrids <- list('xgb1' = xgbGrid1, 'gbm1' = gbmGrid1, 'cubist1' = cubistGrid1, 'rf1' = rfGrid1)

modelList <- c('gbm1', 'rf1', 'cubist1', 'xgb1', 'lasso.9', 'knn2', 'knn3', 'knn5', 'knn10', 'knn20', 'knn50')
level1Result <- fitWholeLevel(noVarImpData = level1Data, varImpData = level1Data, 
                              foldCount = folds, modelList = modelList, 
                              levelNr = 1, tuneGrids = tuneGrids)

fileName <- paste("models/level1/", length(modelList), '-', format(Sys.time(), "%d_%b_%Y_%H-%M-%S"), ".Rds", sep = '')
saveRDS(level1Result, file=fileName)
loginfo(paste('Saved Level 1 Models:', fileName))