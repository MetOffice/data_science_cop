# Load in required packages
library(MASS)
library(mgcv)
library(tidyverse)
library(mgcViz)
library(data.table)

# Set your working directory 
setwd('~/Documents/DataScienceCoP')

# Load in demand and weather data frame
load('Data.RData')

# Look at dataframe
head(Data)

# Plot water demand (we have scaled this to the % of average demand so it can be shared)
plot(Data$Date,Data$WaterDemand,type="l",xlab="Date",ylab="Scaled Water Demand")

# Explore relationships in the data
# Paired plot
#pairs(Data,pch=19,cex=0.5)
  
# Focus on relationship between covariates and water demand
par(mfrow=c(2,5),mar=c(4,4,4,4))
plot(Data$MaxTemp,Data$WaterDemand,xlab="Max Temp",ylab="Scaled Water Demand",pch=19)
plot(Data$SunHours,Data$WaterDemand,xlab="Sun Hours",ylab="Scaled Water Demand",pch=19)
plot(Data$Rainfall,Data$WaterDemand,xlab="Rainfall",ylab="Scaled Water Demand",pch=19)
plot(Data$Lag_MaxTemp,Data$WaterDemand,xlab="Lag Max Temp",ylab="Scaled Water Demand",pch=19)
plot(Data$TempIndex,Data$WaterDemand,xlab="Temp Index",ylab="Scaled Water Demand",pch=19)
plot(Data$SoilMoist,Data$WaterDemand,xlab="Soil Moisture Deficit",ylab="Scaled Water Demand",pch=19)
plot(Data$Month,Data$WaterDemand,xlab="Month",ylab="Scaled Water Demand",pch=19)
plot(Data$DayofYear,Data$WaterDemand,xlab="Day of Year",ylab="Scaled Water Demand",pch=19)
plot(Data$DayofWeek,Data$WaterDemand,xlab="Day of Week",ylab="Scaled Water Demand",pch=19)
plot(Data$TimeOverall_Month,Data$WaterDemand,xlab="Time overall (by month)",ylab="Scaled Water Demand",pch=19)


# MODELLING

# Split data up randomly into training and testing data sets, 80:20 proportion respectively 
train_index <- sample(1:nrow(Data), 0.8 * nrow(Data))
test_index <- setdiff(1:nrow(Data), train_index)
training.data <- Data[train_index,]
test.data <- Data[test_index,]


# Model 1: Just weather

# fit the model using the 'gam' function. For more info type ?gam
model1 <- gam(WaterDemand ~
             +s(MaxTemp)
             +s(SunHours)
             +s(Rainfall)
             +s(Lag_MaxTemp)
             +s(TempIndex)
             +s(SoilMoist),
             data = training.data)

# look at model summary:
summary(model1)

# plot resulting splines representing the relationship between the weather variables and water demand
plot(model1,pages=1,residuals=TRUE,all.terms=TRUE,shade=TRUE,shade.col=2,ylab='y')

# plot model diagnostics. For more info type ?gam.check
par(mfrow=c(2,2))
gam.check(model1)

# predict from the model using the test data
pred <- predict.gam(model1,test.data)
# plot comparison of truth and predicted 
par(mfrow=c(1,1),mar=c(4,4,4,4))
plot(test.data$WaterDemand,pred,xlab='True Water Demand (Test Data)',ylab='Predicted Water Demand',pch=19,xlim=c(0.7,1.4),ylim=c(0.7,1.4))
abline(0,1,col='blue',lwd=2)


# Model 2: Add in temporal covariates 

# fit the model using the 'gam' function. For more info type ?gam
model2 <- gam(WaterDemand ~
              +s(MaxTemp)
              +s(SunHours)
              +s(Rainfall)
              +s(Lag_MaxTemp)
              +s(TempIndex)
              +s(SoilMoist)
              +s(DayofYear,bs="cr",k=50)
              +s(DayofWeek,k=7)
              +s(TimeOverall_Month,k=20),
              data = training.data)

# look at model summary:
summary(model2)

# plot resulting splines representing the relationship between the weather variables and water demand
plot(model2,pages=1,residuals=TRUE,all.terms=TRUE,shade=TRUE,shade.col=2,ylab='y')

# plot model diagnostics. For more info type ?gam.check
par(mfrow=c(2,2))
gam.check(model2)

# predict from the model using the test data
pred <- predict.gam(model2,test.data)
# plot comparison of truth and predicted 
par(mfrow=c(1,1),mar=c(4,4,4,4))
plot(test.data$WaterDemand,pred,xlab='True Water Demand (Test Data)',ylab='Predicted Water Demand',pch=19,xlim=c(0.7,1.4),ylim=c(0.7,1.4))
abline(0,1,col='blue',lwd=2)


# Model 3: Add in interaction term

# fit the model using the 'gam' function. For more info type ?gam
model3 <- gam(WaterDemand ~
              +s(MaxTemp)
              +s(SunHours)
              +s(Rainfall)
              +s(Lag_MaxTemp)
              +s(TempIndex)
              +s(SoilMoist)
              +s(DayofYear,bs="cr",k=50)
              +s(DayofWeek,k=7)
              +s(TimeOverall_Month,k=20)
              +ti(Rainfall,Month),         # <- interaction between Rainfall and month index 
              data = training.data)

# look at model summary:
summary(model3)

# plot resulting splines representing the relationship between the weather variables and water demand
plot(model3,pages=1,residuals=TRUE,all.terms=TRUE,shade=TRUE,shade.col=2)
# plot just the interaction part for closer look
mod3 <- getViz(model3)
plot(mod3,residuals=TRUE,all.terms=TRUE,shade=TRUE,shade.col=2,select=10)


# plot model diagnostics. For more info type ?gam.check
par(mfrow=c(2,2))
gam.check(model3)

# predict from the model using the test data
pred <- predict.gam(model3,test.data)
# plot comparison of truth and predicted 
par(mfrow=c(1,1),mar=c(5,5,5,5))
plot(test.data$WaterDemand,pred,xlab='True Water Demand (Test Data)',ylab='Predicted Water Demand',pch=19,xlim=c(0.7,1.4),ylim=c(0.7,1.4))
abline(0,1,col='blue',lwd=2)


# Split out non-weather part of model to check it makes sense

pred<-data.table(predict.gam(model3,test.data,type="terms"))
pw<-select(pred,-'s(DayofYear)',-'s(TimeOverall_Month)',-'s(DayofWeek)')
#base = non weather component of model
pred$predbase<-pred$'s(DayofYear)'+pred$'s(TimeOverall_Month)'+pred$'s(DayofWeek)'+model3$coefficients[1]
#weather component
pred$predweather=rowSums(pw)

#weather component can't be negative
minpw<-min(pred$predweather)
pred$predweather<-pred$predweather-minpw
pred$predbase<-pred$predbase+minpw
pred$predDI<-pred$predbase+pred$predweather

#compare modelled and observed demand
pred$Demand<-test.data$WaterDemand
pred$error<-pred$predDI-pred$Demand

par(mfrow=c(1,1))

plot(test.data$Date,pred$predbase+pred$predweather,type="l",col="blue",ylim=c(0.5,1.5),xlab=("Date"),ylab=("Scaled Demand"))
lines(test.data$Date,test.data$WaterDemand,col="black")
lines(test.data$Date,pred$predbase,col="red")
legend("topleft",c("Modelled demand","Observed demand","Base demand"),cex=.8,lty=c(1,1,1),col=c("blue", "black","red"))


# base model for just 1 year - see dip in demand in summer and at xmas - people on holiday so usage drops
plot(test.data$Date,pred$predbase,type="l",col="red",ylim=c(0.8,1.05),xlab=("Date"),ylab=("Scaled Demand"),xlim=as.Date(c("2011-01-01","2011-12-31")))
legend("topleft",c("Base demand"),cex=.8,lty=c(1),col=c("red"))

  
