setwd('C:/RStudioWorkspace/HousePricesAdvanced')
source(file = 'scripts/initScripting.R', echo = FALSE, encoding = 'UTF-8')
loginfo('Fitting Level 0 Models')
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
# Transform to Interface
l0Data <- extractL0Data(nzvL0, train.raw$SalePrice)

cvRfFit <- rfcv(trainx = l0Data$train$predictor, trainy = l0Data$train$y, cv.fold=5, scale="log", step=0.5)

fileName <- paste("models/cvRfFit_", 1, ".Rds", sep = '')
saveRDS(cvRfFit, file=fileName)
loginfo(paste('Saved Level 0 Model:', fileName))