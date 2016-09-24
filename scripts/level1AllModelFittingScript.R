setwd('C:/RStudioWorkspace/HousePricesAdvanced')
source(file = 'C:/RStudioWorkspace/HousePricesAdvanced/scripts/initScripting.R', echo = FALSE, encoding = 'UTF-8')
loginfo('Fitting Level 1 Models')

level0Result <- readRDS('models/level0/allModelsLevel0_2.Rds')

level0Preds <- level0Result$cvPredictions
level0Models <- level0Result$finalModels
level0TestPreds <- level0Result$testPredictions

level1Data <- level1Preprocess(train = level0Preds, test = level0TestPreds)

folds <- 20
xgbGrid1 <- expand.grid(nrounds = 300,
                        max_depth = 4,
                        eta = 0.025,
                        gamma = 0,
                        colsample_bytree = 0.8,
                        min_child_weight = 1)
gbmGrid1 <- expand.grid(interaction.depth = 6,
                        n.trees = 900, 
                        shrinkage = 0.005,
                        n.minobsinnode = 10)
cubistGrid1 <- expand.grid(committees = 10, 
                           neighbors = 7)
rfGrid1 <- expand.grid(mtry = 2)
tuneGrids <- list('xgb1' = xgbGrid1, 'gbm1' = gbmGrid1, 'cubist1' = cubistGrid1, 'rf1' = rfGrid1)

modelList <- c('brnn2', 'gbm1', 'rf1', 'cubist1', 'lasso.9', 'xgb1', 'knn2', 'knn3', 'knn5', 'knn10', 'knn20', 'knn50')
level1Result <- fitWholeLevel(noVarImpData = level1Data, varImpData = level1Data, 
                              foldCount = folds, modelList = modelList, 
                              levelNr = 1, tuneGrids = tuneGrids)

fileName <- paste("models/level1/", length(modelList), '-', format(Sys.time(), "%d_%b_%Y_%H-%M-%S"), ".Rds", sep = '')
saveRDS(level1Result, file=fileName)
loginfo(paste('Saved Level 1 Models:', fileName))