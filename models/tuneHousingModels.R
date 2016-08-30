tuneModelFacade <- function(x, y, trControl,
                             tuneGrid, preProcess = NULL, algo = 'xgboost', seed = 16450) {
    loginfo(paste('Start tuning', algo, '- model...'))
    if(algo == 'xgboost') {
        tuneResult <- train(x = x, y = y, method = 'xgbTree', preProcess = preProcess, 
                            trControl = trControl, tuneGrid = tuneGrid, na.action = na.pass)
    } else {
        stop('Wrong Model Digger')
    }
    loginfo('End tuning xgBoost model')
    return(tuneResult)
}
