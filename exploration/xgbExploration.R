#outliers <- c(1183, 524, 899, 692, 689)
outliers <- c()
preprocConfig <- list('featureReduction' = 'klyus', 
                      'addFeats1' = FALSE,
                      'addFeats2' = FALSE,
                      'interactions' = FALSE,
                      'nzvFreqCut' = NA, 
                      'scale' = FALSE,
                      'outliers' = outliers)
modelList <- list(
    'xgb1' = list('nrounds' = 1000, 
                  'max_depth' = 3, 
                  'eta' = 0.01, 
                  'gamma' = 0,
                  'subsample' = 0.8,
                  'colsample_bytree' = 0.8,
                  'min_child_weight' = 1)
)

train.raw <- read.csv('data/train.csv',stringsAsFactors=FALSE)
test.raw <- read.csv('data/test.csv',stringsAsFactors=FALSE)
xgbRes <- predictFeatureSet(train.raw, test.raw, preprocConfig, modelList, foldCount = 10) 

#### Identified Outliers ####
# outliers <- c(1183, 524, 899, 692, 689)
explData <- data.frame(xgbRes$cvPreds$id, xgbRes$cvPreds$y, xgbRes$cvPreds$predictors[,1])
predcolname <- colnames(xgbRes$cvPreds$predictors)[1]
colnames(explData) <- c('id', 'SalePrice', predcolname)
explData <- mutate(explData, error = abs(SalePrice - xgb1))
outlierExp <- arrange(explData, desc(error))
head(outlierExp, n = 25)
xgbOutliers <- filter(outlierExp, error > 150000)$id

ggplot(explData, aes(xgbRes$cvPreds$predictors[,1], SalePrice)) +
    geom_point() +
    xlab(predcolname) +
    geom_abline(intercept = 0, slope = 1, color = 'red')

#### Loading Model ####
xgbTuning1 <- readRDS('models/xgbTune_2016-10-23_20-59-37.Rds')
xgbTuning2 <- readRDS('models/xgbTune_2016-10-23_18-12-41.Rds')


### Exploration
xgb1 <- arrange(xgbTuning1, rmsle)
xgb2 <- arrange(xgbTuning2, rmsle)
head(xgb1)
head(xgb2)

