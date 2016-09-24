cat("Loading necessary packages...\n")
suppressWarnings(suppressMessages(library(glmnet)))
suppressWarnings(suppressMessages(library(Metrics)))
suppressWarnings(suppressMessages(library(outliers)))
suppressWarnings(suppressMessages(library(randomForest)))
suppressWarnings(suppressMessages(library(caret)))
suppressWarnings(suppressMessages(library(ggplot2)))
suppressWarnings(suppressMessages(library(vioplot)))
suppressWarnings(suppressMessages(library(corrplot)))
suppressWarnings(suppressMessages(library(Hmisc)))
suppressWarnings(suppressMessages(library(plyr)))
suppressWarnings(suppressMessages(library(dplyr)))
suppressWarnings(suppressMessages(library(logging)))
cat("Packages loaded.\n\n")

# Logging
basicConfig()

# Loading source
source(file = 'C:/RStudioWorkspace/HousePricesAdvanced/modeling/preprocess.R', echo = FALSE, encoding = 'UTF-8')
source(file = 'C:/RStudioWorkspace/HousePricesAdvanced/modeling/modeling.R', echo = FALSE, encoding = 'UTF-8')