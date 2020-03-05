---
title: "Machine Learning Assignment"
author: "Jose Noriega"
date: "3/3/2020"
output: html_document
---

## Summary

The goal of this project is to predict the variable "classe",in the data set: "Weight Lifting Exercise", which is a sample of the manner users exercises. We are going to present how we built our models, select few to make our prediction, how we made cross validation and expectations. Finally we are going to use our final prediction model to predict 20 different test cases.
The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har.

## Exploratory Data and Upload necessary libraries

Staring with setting the working directory,loading libraries and the upload of the data training and testing

```{r, message=FALSE, warning=FALSE}
setwd("~/R/PracticalMachineLearning")
library(lattice);library(ggplot2);library(bitops);library(rpart);library(rpart.plot)
library(rattle);library(caret);library(randomForest);library(ggplot2)
library(AppliedPredictiveModeling);library(dplyr)

Training<-read.csv("pml-training.csv",strip.white = TRUE,na.strings = c("NA",""))
Testing<-read.csv("pml-testing.csv",strip.white = TRUE,na.strings = c("NA",""))
dim(Training)
dim(Testing)
```
As Training and Testing data have the same variables (160), we are going to clean the missing data.

## Cleaning the Data

```{r}
features<-names(Testing[,colSums(is.na(Testing))==0])[8:59]
Training<-Training[,c(features,"classe")]
Testing<-Testing[,c(features,"problem_id")]
dim(Training)
dim(Testing)
```

## Partitioning the data

```{r}
set.seed(1234)
inTrain<-createDataPartition(y=Training$classe,p=0.6,list = FALSE)
training<-Training[inTrain,]
testing<-Training[-inTrain,]
dim(training)
dim(testing)
```

# Analysis different Models

## Random Forest Model
```{r}
set.seed(1234)
RandomForrest<-randomForest(classe ~ .,training, ntree=100)
RandomForrest

RandomForrestPred<-predict(RandomForrest,newdata = testing,type = "class")
confusionMatrix(RandomForrestPred,testing$classe)
```

## Decision Tree Model

```{r}
modFit<-rpart(classe~.,training,method="class")
fancyRpartPlot(modFit,cex=0.4)

prediction<-predict(modFit, testing, type = "class")
confusionMatrix(prediction,testing$classe)
```

## Generalized BoostedRegression Model
```{r}
Ctrlmodgb<-trainControl(method = "repeatedcv",number = 5,repeats = 2)
modgbm<-train(classe~., data = training,method="gbm",trControl=Ctrlmodgb,verbose=FALSE)
predictgbm<-predict(modgbm,newdata=testing)
confusionMatrix(predictgbm,testing$classe)
```

## Conclusion

The prediction accuracy resulting for the three model are:
Desicion Tree Model: 73,20 %
Generalized Boosted Model:95.92%
Random Forrest Model: 99,30%

Considering that Random Forrest Model provided more accuracy results, we consider it for our final prediction of 20 different test cases

```{r, results=FALSE}
RandomForrestPred<-predict(RandomForrest,newdata = testing,type = "class")
RandomForrestPred
```
```{r}
# [1] B A B A A E D B A A B C B A E E A B B B
# Levels: A B C D E
```
 






