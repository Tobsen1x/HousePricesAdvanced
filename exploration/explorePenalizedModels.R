train.raw <- read.csv('data/train.csv',stringsAsFactors=FALSE)
test.raw <- read.csv('data/test.csv',stringsAsFactors=FALSE)

##### Data Preprocessing #####
ppL0 <- preprocL0(train.raw, test.raw)

# Dummy Vars
dmy <- dummyVars(" ~ .", data = ppL0)
ppL0 <- data.frame(predict(dmy, newdata = ppL0))

# Near Zero Variance feature reduction
#nearZeroVar(ppL0, freqCut = 300, saveMetrics = TRUE)
nzvL0 <- ppL0[, -nearZeroVar(ppL0, freqCut = 100)]

# Remove Outliers #
outliers <- c(1299, 524, 822)
nzvL0 <- filter(nzvL0, !Id %in% outliers)

# Transform to Interface
l0Data <- extractL0Data(nzvL0, train.raw$SalePrice[!train.raw$Id %in% outliers])

##### LM Modeling #####
lmData <- as.data.frame(cbind('id' = l0Data$train$id, 'SalePrice' = l0Data$train$y,
                as.data.frame(l0Data$train$predictors)))
featureFormula <- paste(colnames(l0Data$train$predictors), sep = ' ', collapse = ' + ')
lmFormula <- as.formula(paste('SalePrice ~ -id +', featureFormula))

lmFit <- lm(formula = lmFormula, data = lmData )
summary(lmFit)
hv <- hatvalues(model = lmFit)
cd <- cooks.distance(model = lmFit)
obsExpl <- data.frame(id = as.numeric(names(hv)), hatvalues = hv, cooksDistance = cd)
ggplot(obsExpl, aes(x='Hatvalues', y=hatvalues)) + geom_violin()
ggplot(obsExpl, aes(x='Cooks Distance', y=cooksDistance)) + geom_violin()
obsExpl <- arrange(obsExpl, desc(hatvalues))
obsExpl <- arrange(obsExpl, desc(cooksDistance))
head(obsExpl, 10)


l <- data.frame(id = lmData$id, y = exp(lmData$SalePrice), lm = exp(predict(lmFit)))
rmsle(l$y, l$lm)
l[l$id == 1299,]

##### GLMNET Modelling #####
# Tuning lambda for lasso #
cvLassoFit <- cv.glmnet(x = l0Data$train$predictors, y = l0Data$train$y,
                        alpha = 1, keep = FALSE)
paste('Lasso lambda min:', cvLassoFit$lambda.min)


# Variable Importance #
lassoCoefs <- coef(cvLassoFit, s = cvLassoFit$lambda.min)
lassoImp <- data.frame(feature = rownames(lassoCoefs), Overall = lassoCoefs[,1])
lassoImp$Overall <- abs(lassoImp[, 'Overall', drop = TRUE])
lassoImp <- arrange(lassoImp, desc(Overall))
relImp <- filter(lassoImp, Overall > 0.02, feature != '(Intercept)')
dotchart2(data = relImp$Overall, labels = relImp$feature)

# lasso Fit
lassoFit <- glmnet(x = l0Data$train$predictors, y = l0Data$train$y,
                 family = 'gaussian', alpha = 1, lambda = cvLassoFit$lambda.min)
lassoFit

lassoTestPred <- exp(predict(object = lassoFit, newx = l0Data$test$predictors, 
                             s = cvLassoFit$lambda.min, type = 'response'))
lassoSub <- data.frame('Id' = l0Data$test$id, 'SalePrice' = lassoTestPred[,1])
#write.csv(lassoSub,file="submissions/l0_lasso_2.csv",row.names=FALSE)

# Ridge Regression
cvRidgeFit <- cv.glmnet(x = l0Data$train$predictors, y = l0Data$train$y,
                        alpha = 0)
paste('Ridge regression lambda min:', cvRidgeFit$lambda.min)

# Ridge Regression Fit
ridgeFit <- glmnet(x = l0Data$train$predictors, y = l0Data$train$y,
                   family = 'gaussian', alpha = 0, lambda = cvRidgeFit$lambda.min)
ridgeFit

##### Random Forest Modeling #####
#cvRfFit <- rfcv(trainx = l0Data$train$predictor, trainy = l0Data$train$y, cv.fold=5, scale="log", step=0.5)
cvRfFit <- readRDS('models/cvRfFit_1.Rds')
sqrt(cvRfFit$error.cv)

rfFit <- randomForest(x = l0Data$train$predictors, y = l0Data$train$y, mtry = 80, ntree = 1000)
rfFit

##### Cross validation to ensure Performance R2 = 0.9047 #####

foldCount <- 10
seed <- 16450
modelList <- list(
    'lasso' = list('family' = 'gaussian', 'alpha' = 1, 'lambda' = cvLassoFit$lambda.min),
    'rf' = list('mtry' = 80, 'ntree' = 500)
    #'ridge' = list('family' = 'gaussian', 'alpha' = 0, 'lambda' = cvRidgeFit$lambda.min)
    )

set.seed(seed)
foldArray <- createFolds(l0Data$train$y, k = foldCount, 
                         list = FALSE)
resultList <- list()

# CV Predict Trainset
allPreds <- data.frame()
allMetrics <- data.frame()
for(i in unique(foldArray)) {
    trainFold <- foldData(l0Data$train, foldArray, i)
    
    foldModels <- trainLevelModels(modelList = modelList, trainData = trainFold$train, seed = seed)
    foldPreds <- predictLevel(models = foldModels, testData = trainFold$test)
    
    actMetrics <- evaluatePrediction(foldPreds)
    actMetrics <- cbind('fold' = i, actMetrics)
    
    if(nrow(allPreds) == 0) {
        allPreds <- foldPreds
        allMetrics <- actMetrics
    } else {
        allPreds <- rbind(allPreds, foldPreds)
        allMetrics <- rbind(allMetrics, actMetrics)
    }
}
allPreds <- arrange(allPreds, id)
resultList <- append(resultList, list('cvPredictions' = allPreds))

lassoMetrics <- filter(allMetrics, model == 'lasso')
lassoMetrics


metrics <- evaluatePrediction(resultList$cvPredictions)
metrics

qplot(x = resultList$cvPredictions$y, y = resultList$cvPredictions$rf)

lassoErrors <- as.data.frame(cbind('id' = resultList$cvPredictions$id, 'lassoErr' = resultList$cvPredictions$y -
                         resultList$cvPredictions$lasso))
sortErrs <- arrange(lassoErrors, desc(abs(lassoErr)))
head(sortErrs, 10)
filter(sortErrs, lassoErr > 75000)
problemObs <- c(1299, 524)

rfErrors <- as.data.frame(cbind('id' = resultList$cvPredictions$id, 'rfErr' = resultList$cvPredictions$y -
                                       resultList$cvPredictions$rf))
sortRfErrs <- arrange(rfErrors, desc(abs(rfErr)))
head(sortRfErrs, 10)
filter(sortRfErrs, rfErr > 75000)

rawProblems <- filter(train.raw, Id %in% problemObs)
preprocProblems <- cbind(Id = l0Data$train$id, SalePrice = exp(l0Data$train$y), as.data.frame(l0Data$train$predictors))
preprocProblems <- filter(preprocProblems, Id %in% problemObs)



# TODO Caret to natural algo usage comparison #