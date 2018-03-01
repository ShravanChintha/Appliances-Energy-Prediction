#Reading data from CSV file downloaded from machine learning repository
#file can be downloaded from link: https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction

library(readr)
energydata = read_csv("energydata_complete.csv")

#removing date and random variables as models related to time are not applied here
View(energydata)
energydata=energydata[,-c(1,28,29)]

#dividing data into training and test datasets
#data is divided into 80% training data and 20% test data
library(caret)
library(lattice)
library(ggplot2)
library(e1071)

set.seed(1)
train_index = createDataPartition(energydata$Appliances,p=0.8,list=FALSE)

traindata = energydata[train_index,]
testdata = energydata[-train_index,]

#Analyzing data using pair wise plots
#Plots are plotted separately as size of the plot with 26 variables is more than accepted by R
library(psych)
pairs.panels(traindata[,c(1,2,3,4,5,6,7)])
pairs.panels(traindata[,c(1,8,9,10,11,12,13)])
pairs.panels(traindata[,c(1,14,15,16,17,18,19)])
pairs.panels(traindata[,c(1,20,21,22,23,24,25,26)])

#Data preprocessing
hist(traindata$RH_3,xlab="RH_3",col = "cyan",main="Histogram of RH_3")

#applying BoxCox transformation, scaling and centering methods to reduce the skewness

traindata=as.data.frame(traindata)
traindataPre=preProcess(traindata,method = c("BoxCox","center","scale"))
traindatapro=predict(traindataPre,traindata)

hist(traindatapro$RH_3,xlab="RH_3",col = "cyan",main="Histogram of RH_3 after transformation")

#plotting correlation plots and removing highly correlated variables
library(corrplot)
traindatacorr=cor(traindata)
corrplot(traindatacorr,order="hclust",tl.cex = .75)

#removing variables with correlation morethan 0.8
highCorr <- findCorrelation(traindatacorr, cutoff = .8)
filteredTraindata = traindata[, -highCorr]
ncol(filteredTraindata)

#Correlation plot of the remaining variables
corrplot(cor(filteredTraindata),order="hclust")

#Applying modelling techniques
#Linear Regression model:

#Resampling data using 10 fold Cross validation
ctrl <- trainControl(method = "cv", number = 10, repeats=5)

filteredTraindataX=filteredTraindata[-1]
testdataX=testdata[-1]


set.seed(100)
lmFit <- train(x = filteredTraindataX, y = filteredTraindata$Appliances,
                method = "lm", trControl = ctrl)

lmFit

lmPred = predict(lmFit, newdata=testdataX)
lmPR = postResample(pred=lmPred, obs=testdata$Appliances)

lmPR

rmses = c(lmPR[1])
r2s = c(lmPR[2])
methods = c("Linear Regression")

#KNN model:

knnModel = train(x=filteredTraindataX, y=filteredTraindata$Appliances, method="knn",
                 preProc=c("center","scale"),tuneLength=10)

knnModel

plot(knnModel$results$k, knnModel$results$RMSE, type="o",
     xlab="# neighbors",ylab="RMSE", main="KNNs for Energy Data")

knnPred = predict(knnModel, newdata=testdataX)

knnPR = postResample(pred=knnPred, obs=testdata$Appliances)

knnPR

rmses = c(rmses,knnPR[1])
r2s = c(r2s,knnPR[2])
methods = c(methods,"KNN")

#MARS method:
marsGrid = expand.grid(.degree=1:2, .nprune=2:38)
set.seed(100)
marsModel = train(x=filteredTraindataX, y=filteredTraindata$Appliances, 
                  method="earth", preProc=c("center", "scale"), tuneGrid=marsGrid)

marsModel

varImp(marsModel)

marsPred = predict(marsModel, testdataX)
marsPR = postResample(pred=marsPred, obs=testdata$Appliances)

marsPR

rmses = c(rmses,marsPR[1])
r2s = c(r2s,marsPR[2])
methods = c(methods,"MARS")

#Bagging tree
library(ipred)

BaggTree= bagging( filteredTraindata$Appliances ~ ., data= filteredTraindata)

Bagg_yHat = predict(BaggTree, testdataX)

BaggPR = postResample(pred=Bagg_yHat, obs=testdata$Appliances)
BaggPR

rmses = c(rmses,BaggPR[1])
r2s = c(r2s,BaggPR[2])
methods = c(methods,"Bagging Trees")

#Random Forest method
library(partykit)
library(randomForest)

rfModel = randomForest( filteredTraindata$Appliances ~ ., data=filteredTraindata, 
                        ntree=200, importance= TRUE )

rf_yHat = predict(rfModel,testdata)

rfPR = postResample(pred=rf_yHat, obs=testdata$Appliances)
rfPR

rmses = c(rmses,rfPR[1])
r2s = c(r2s,rfPR[2])
methods = c(methods,"Random Forest")

# the best RMSE and R2 values are produced by Random forest model.
