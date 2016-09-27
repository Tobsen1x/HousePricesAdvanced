extractL0Data <- function(allData, trainSalesprice) {
    ### Transform to Interface ###
    trainInp <- filter(allData, Id %in% 1:1460)
    testInp <- filter(allData, Id %in% 1461:2919)
    trainPredictors <- select(trainInp, -Id)
    testPredictors <- select(testInp, -Id)
    
    #trainPredictors <- as.matrix(trainPredictors)
    #testPredictors <- as.matrix(testPredictors)
    
    tr <- list('id' = trainInp$Id, 'y' = log(trainSalesprice), 'predictors' = trainPredictors)
    tst <- list('id' = testInp$Id, 'y' = NULL, 'predictors' = testPredictors)
    l0FeatureSet <- list('train' = tr, 'test' = tst)
    return(l0FeatureSet)
}

preprocL0 <- function(train, test) {
    dr <- rbind(select(train, -SalePrice), test)
    
    # Value Mappings
    lotShapeMap <- data.frame('level' = c('Reg', 'IR1', 'IR2', 'IR3'), 
                              'value' = c(5, 6, 8, 7), 
                              stringsAsFactors = FALSE)
    landSlopeMap <- data.frame('level' = c('Gtl', 'Mod', 'Sev'), 
                               'value' = c(3, 4, 4), 
                               stringsAsFactors = FALSE)
    exterQualMap <- data.frame('level' = c('Ex', 'Gd', 'TA', 'Fa', 'Po', NA), 
                               'value' = c(10, 7, 4, 2, 3, 0), 
                               stringsAsFactors = FALSE)
    exterQualMap <- data.frame('level' = c('Ex', 'Gd', 'TA', 'Fa', 'Po', NA), 
                               'value' = c(10, 7, 4, 2, 3, 0), 
                               stringsAsFactors = FALSE)
    exterCondMap <- data.frame('level' = c('Ex', 'Gd', 'TA', 'Fa', 'Po', NA), 
                               'value' = c(10, 9, 9, 5, 5, 0), 
                               stringsAsFactors = FALSE)
    bsmtQualMap <- data.frame('level' = c('Ex', 'Gd', 'TA', 'Fa', 'Po', NA), 
                              'value' = c(10, 7, 5, 4, 2, 2), 
                              stringsAsFactors = FALSE)
    bsmtCondMap <- data.frame('level' = c('Ex', 'Gd', 'TA', 'Fa', 'Po', NA), 
                              'value' = c(10, 7, 6, 4, 1, 0), 
                              stringsAsFactors = FALSE)
    bsmtExposureMap <- data.frame('level' = c('Gd', 'Av', 'Mn', 'No', NA), 
                                  'value' = c(10, 8, 7, 5, 0), 
                                  stringsAsFactors = FALSE)
    bsmtFinType1Map <- data.frame('level' = c('GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', NA), 
                                  'value' = c(10, 7, 6, 6, 6, 7, 3), 
                                  stringsAsFactors = FALSE)
    bsmtFinType2Map <- data.frame('level' = c('GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', NA), 
                                  'value' = c(10, 7, 6, 6, 6, 7, 3), 
                                  stringsAsFactors = FALSE)
    heatingQCMap <- data.frame('level' = c('Ex', 'Gd', 'TA', 'Fa', 'Po', NA), 
                               'value' = c(10, 7, 6, 5, 4, 0), 
                               stringsAsFactors = FALSE)
    kitchenQualMap <- data.frame('level' = c('Ex', 'Gd', 'TA', 'Fa', 'Po', NA), 
                                 'value' = c(10, 6, 4, 3, 1, 0), 
                                 stringsAsFactors = FALSE)
    functionalMap <- data.frame('level' = c('Typ', 'Min1', 'Min2', 'Mod', 'Maj1', 'Maj2', 'Sev', 'Sal'), 
                                'value' = c(12, 11, 10, 8, 6, 5, 3, 1), 
                                stringsAsFactors = FALSE)
    fireplaceQuMap <- data.frame('level' = c('Ex', 'Gd', 'TA', 'Fa', 'Po', NA), 
                                 'value' = c(10, 6, 5, 4, 1, 0), 
                                 stringsAsFactors = FALSE)
    garageFinishMap <- data.frame('level' = c('Fin', 'RFn', 'Unf', NA), 
                                  'value' = c(10, 8, 5, 0), 
                                  stringsAsFactors = FALSE)
    garageQualMap <- data.frame('level' = c('Ex', 'Gd', 'TA', 'Fa', 'Po', NA), 
                                'value' = c(10, 9, 8, 3, 1, 0), 
                                stringsAsFactors = FALSE)
    garageCondMap <- data.frame('level' = c('Ex', 'Gd', 'TA', 'Fa', 'Po', NA), 
                                'value' = c(10, 9, 8, 3, 1, 0), 
                                stringsAsFactors = FALSE)
    pavedDriveMap <- data.frame('level' = c('Y', 'P', 'N'), 
                                'value' = c(10, 8, 6), 
                                stringsAsFactors = FALSE)
    poolQCMap <- data.frame('level' = c('Ex', 'Gd', 'TA', 'Fa', 'Po', NA), 
                            'value' = c(10, 8, 7, 4, 2, 0), 
                            stringsAsFactors = FALSE)
    fenceMap <- data.frame('level' = c('GdPrv', 'MnPrv', 'GdWo', 'MnWw', NA), 
                           'value' = c(10, 8, 8, 7, 0), 
                           stringsAsFactors = FALSE)
    
    m <- mutate(dr, 
                MSSubClass = as.factor(MSSubClass),
                MSZoning = as.factor(ifelse(is.na(MSZoning), majorFactorValue(MSZoning), MSZoning)),
                Street = as.factor(Street),
                Alley = as.factor(ifelse(is.na(Alley), "NoAccess", Alley)),
                LotShape = mapFactorValues(LotShape, lotShapeMap),
                LandContour = as.factor(LandContour),
                Utilities = as.factor(ifelse(is.na(Utilities), majorFactorValue(Utilities), Utilities)),
                LotConfig = as.factor(LotConfig),
                LandSlope = mapFactorValues(LandSlope, landSlopeMap),
                Neighborhood = as.factor(Neighborhood),
                Condition1 = as.factor(Condition1),
                Condition2 = as.factor(Condition2),
                BldgType = as.factor(BldgType),
                HouseStyle = as.factor(HouseStyle),
                RoofStyle = as.factor(RoofStyle),
                RoofMatl = as.factor(RoofMatl),
                Exterior1st = as.factor(ifelse(is.na(Exterior1st), majorFactorValue(Exterior1st), Exterior1st)),
                Exterior2nd = as.factor(ifelse(is.na(Exterior2nd), majorFactorValue(Exterior2nd), Exterior2nd)),
                MasVnrType = as.factor(ifelse(is.na(MasVnrType), "None", MasVnrType)),
                ExterQual = mapFactorValues(ExterQual, exterQualMap),
                ExterCond = mapFactorValues(ExterCond, exterCondMap),
                Foundation = as.factor(Foundation),
                BsmtQual = mapFactorValues(BsmtQual, bsmtQualMap),
                BsmtCond = mapFactorValues(BsmtCond, bsmtCondMap),
                BsmtExposure = mapFactorValues(BsmtExposure, bsmtExposureMap),
                BsmtFinType1 = mapFactorValues(BsmtFinType1, bsmtFinType1Map),
                BsmtFinType2 = mapFactorValues(BsmtFinType2, bsmtFinType2Map),
                Heating = as.factor(Heating),
                HeatingQC = mapFactorValues(HeatingQC, heatingQCMap),
                CentralAir = as.factor(CentralAir),
                Electrical = as.factor(ifelse(is.na(Electrical), majorFactorValue(Electrical), Electrical)),
                KitchenQual = mapFactorValues(KitchenQual, kitchenQualMap),
                Functional = mapFactorValues(Functional, functionalMap),
                FireplaceQu = mapFactorValues(FireplaceQu, fireplaceQuMap),
                GarageType = as.factor(ifelse(is.na(GarageType), "None", GarageType)),
                GarageFinish = mapFactorValues(GarageFinish, garageFinishMap),
                GarageQual = mapFactorValues(GarageQual, garageQualMap),
                GarageCond = mapFactorValues(GarageCond, garageCondMap),
                PavedDrive = mapFactorValues(PavedDrive, pavedDriveMap),
                PoolQC = mapFactorValues(PoolQC, poolQCMap),
                Fence = mapFactorValues(Fence, fenceMap),
                MiscFeature = as.factor(ifelse(is.na(MiscFeature), "None", MiscFeature)),
                SaleType = as.factor(ifelse(is.na(SaleType), "None", SaleType)),
                SaleCondition = as.factor(ifelse(is.na(SaleCondition), "None", SaleCondition))
    )
    
    ### Imputation ###
    i <- mutate(m,
                LotFrontage = ifelse(is.na(LotFrontage), median(LotFrontage, na.rm = TRUE), LotFrontage),
                MasVnrArea = ifelse(is.na(MasVnrArea), median(MasVnrArea, na.rm = TRUE), MasVnrArea),
                BsmtFinSF1 = ifelse(is.na(BsmtFinSF1), median(BsmtFinSF1, na.rm = TRUE), BsmtFinSF1),
                BsmtFinSF2 = ifelse(is.na(BsmtFinSF2), median(BsmtFinSF2, na.rm = TRUE), BsmtFinSF2),
                BsmtUnfSF = ifelse(is.na(BsmtUnfSF), median(BsmtUnfSF, na.rm = TRUE), BsmtUnfSF),
                TotalBsmtSF = ifelse(is.na(TotalBsmtSF), median(TotalBsmtSF, na.rm = TRUE), TotalBsmtSF),
                BsmtFullBath = ifelse(is.na(BsmtFullBath), median(BsmtFullBath, na.rm = TRUE), BsmtFullBath),
                BsmtHalfBath = ifelse(is.na(BsmtHalfBath), median(BsmtHalfBath, na.rm = TRUE), BsmtHalfBath),
                FullBath = ifelse(is.na(FullBath), median(FullBath, na.rm = TRUE), FullBath),
                HalfBath = ifelse(is.na(HalfBath), median(HalfBath, na.rm = TRUE), HalfBath),
                Functional = ifelse(is.na(Functional), median(Functional, na.rm = TRUE), Functional),
                GarageYrBlt = ifelse(is.na(GarageYrBlt), median(GarageYrBlt, na.rm = TRUE), GarageYrBlt),
                GarageCars = ifelse(is.na(GarageCars), median(GarageCars, na.rm = TRUE), GarageCars),
                GarageArea = ifelse(is.na(GarageArea), median(GarageArea, na.rm = TRUE), GarageArea)
    )
    return(i)
    
}

mapFactorValues <- function(x, mapping) {
    xframe <- data.frame('level' = x, stringsAsFactors = FALSE)
    values <- left_join(xframe, mapping, by = 'level')[['value']]
    return(values)
}

majorFactorValue <- function(x) {
    table <- table(x)
    major <- names(table[table == max(table)])[1]
    return(major)
}