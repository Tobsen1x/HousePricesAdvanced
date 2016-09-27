train.raw <- read.csv('data/train.csv',stringsAsFactors=FALSE)
test.raw <- read.csv('data/test.csv',stringsAsFactors=FALSE)

##### Data Preprocessing #####
ppL0 <- preprocL0(train = train.raw, test = test.raw, nzvRemove = FALSE, 
                  oneHot = TRUE, skewedRemoveBound = NA)

# Transform to Interface
# Remove Outliers #
outliers <- c(1299, 524, 822)
l0Data <- extractL0Data(ppL0, train.raw, outliers = outliers, asMatrix = TRUE)

set.seed(16450)
cvLassoFit <- cv.glmnet(x = l0Data$train$predictors, y = l0Data$train$y,
                       alpha = 1, keep = FALSE)
min(sqrt(cvLassoFit$cvm))

modelStr <- 'elasticnet'
lassoGrid <- expand.grid(alpha = c(1, 0.9, 0.8),
                        lambda = seq(0.0001, 0.01, by = 0.0001),
                        family = 'gaussian')
lassoTuneResult <- tuneItMan(train = l0Data$train, modelStr = modelStr, tuningGrid = lassoGrid)
lassoTuneResult <- arrange(lassoTuneResult, rmsle)
head(lassoTuneResult)
