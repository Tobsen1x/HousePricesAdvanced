train.raw <- read.csv('data/train.csv',stringsAsFactors=FALSE)
test.raw <- read.csv('data/test.csv',stringsAsFactors=FALSE)

# EARTH Outliers #
outliers <- c()
#outliers <- c(969,463,633,496,31,1325,917,971,682)
preprocConfig <- list('featureReduction' = 'klyus', 
                      'addFeats1' = FALSE,
                      'addFeats2' = FALSE,
                      'interactions' = FALSE,
                      'nzvFreqCut' = 200, 
                      'scale' = TRUE,
                      'outliers' = outliers)

ppL01 <- basePreprocL0(train.raw, test.raw, 
                       featureReduction = preprocConfig$featureReduction,
                       addFeats1 = preprocConfig$addFeats1, 
                       addFeats2 = preprocConfig$addFeats2,
                       nzvFreqCut = preprocConfig$nzvFreqCut, 
                       interactions = preprocConfig$interactions,
                       scale = preprocConfig$scale)

l0Data <- extractL0Data(ppL01, train.raw, 
                        outliers = preprocConfig$outliers, asMatrix = TRUE)

# EARTH Tuning Prameters: nk, degree, nprune
earthFit <- earth(x = l0Data$train$predictors, y = l0Data$train$y,
                  pmethod = 'backward',
                  nk = 100,
                  nprune = NULL,
                  degree = 2,
                  trace = 0)
summary(earthFit)
plot(earthFit)

earthPreds <- predict(earthFit, l0Data$test$predictors)

# TODO Tuning
earthFit <- earth(x = l0Data$train$predictors, y = l0Data$train$y)
                  




#### Neural Net ####
train.raw <- read.csv('data/train.csv',stringsAsFactors=FALSE)
test.raw <- read.csv('data/test.csv',stringsAsFactors=FALSE)

# Outliers #
outliers <- c()
preprocConfig <- list('featureReduction' = 'klyus', 
                      'addFeats1' = FALSE,
                      'addFeats2' = FALSE,
                      'interactions' = FALSE,
                      'nzvFreqCut' = 19, 
                      'scale' = TRUE,
                      'outliers' = outliers)

ppL01 <- basePreprocL0(train.raw, test.raw, 
                       featureReduction = preprocConfig$featureReduction,
                       addFeats1 = preprocConfig$addFeats1, 
                       addFeats2 = preprocConfig$addFeats2,
                       nzvFreqCut = preprocConfig$nzvFreqCut, 
                       interactions = preprocConfig$interactions,
                       scale = preprocConfig$scale)

l0Data <- extractL0Data(ppL01, train.raw, 
                        outliers = preprocConfig$outliers, asMatrix = TRUE)

tuningGrid <- list('nnet1' = expand.grid('decay' = seq(0.000001, 0.00001, 0.000001), 
                                       'size' = 1:4,
                                       'rang' = 0.7,
                                       'maxit' = 1000))
tuneResult <- tuneItMan(l0Data$train, tuningGrid, l0Data$out, foldCount = 5)
?nnet
nnetFit <- nnet(x = l0Data$train$predictors, y = l0Data$train$y, trace = TRUE,
                linout = TRUE, size = 4, rang = 0.1, decay = 5e-4, maxit = 1000)
preds <- exp(predict(nnetFit, type = 'raw'))
summary(nnetFit)


# PLS Tuning Parameters: ncomp
formulaData <- data.frame('y' = l0Data$train$y, l0Data$train$predictors)
plsFit <- plsr(formula = as.formula('y ~ .'),
               data = formulaData,
               ncomp = 23,
               validation = 'CV')
summary(plsFit)
plsPreds <- predict(plsFit, l0Data$test$predictors, comps = 2)



