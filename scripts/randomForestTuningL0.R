setwd('C:/RStudioWorkspace/HousePricesAdvanced')
source(file = 'scripts/initScripting.R', echo = FALSE, encoding = 'UTF-8')
loginfo('Tuning Random Forest Level 0')

train.raw <- read.csv('data/train.csv',stringsAsFactors=FALSE)
test.raw <- read.csv('data/test.csv',stringsAsFactors=FALSE)


##### Level 0 Model Configurations #####
#### Random Forest ####
outliers <- c(1183, 524, 899, 692, 689)
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

tuningGrid <- list('rf1' = expand.grid('mtry' = c(10, 20, 40, 60, 80, 100, 150, 200), 
                                       'ntree' = 1000))
tuneResult <- tuneItMan(l0Data$train, tuningGrid, l0Data$out, foldCount = 5)

timeStr <- format(Sys.time(), format = '%Y-%m-%d_%H-%M-%S', tz = "")
fileName <- paste('models/rfTune_', timeStr, '.Rds', sep = '')
saveRDS(tuneResult, file=fileName)
loginfo(paste('File saved', fileName))