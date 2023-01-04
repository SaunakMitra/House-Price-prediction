#removing all list of elements stored before in the environment
rm(list=ls(all=TRUE))

#Getting the working directory to store the dataset (in csv format)
getwd()

#Importing the dataset
House_data<-read.csv(file="USA_Housing.csv",header = T,sep = ",")

#Calling the libraries which will be used for this project
library(ggplot2)
library(dplyr)
library(psych)
library(GGally)
library(cowplot)
library(corrplot)
library(RColorBrewer)
library(reshape)
library(data.table)
library(caret)
library(superml)
library(Metrics)
library(lmtest)

#Details about the dataset and modification
head(House_data)
sum(is.na(House_data))                   #no na value present in the dataset
str(House_data)     #here we can see the "address" column contains character values which is not supposed to be useful while working with this dataset so we will remove that column
House_data<-House_data[,-7]
head(House_data)
str(House_data)
psych::describe(House_data)

#Exploratory Data Analysis(EDA helps us to visualize the data from where we can get a idea about the data ,i.e how the variables are related and in what relation,the distribution of the variables etc.)
ggplot(melt(House_data), aes(factor(variable), value)) + geom_boxplot() + facet_wrap(~variable, scale="free")+theme_dark()           #boxplot of all variables present in the data


ggpairs( House_data,  upper = list(continuous = "density", combo = "box_no_facet"),lower = list(continuous = "points", combo = "dot_no_facet"))


plot1<-ggplot(House_data, aes(x=Avg..Area.Income, y=Price)) +
  geom_point(size=0.4,shape=4)+theme_dark()
plot2<-ggplot(House_data, aes(x=Avg..Area.House.Age, y=Price)) +
  geom_point(size=0.4,shape=5)+theme_dark()
plot3<-ggplot(House_data, aes(x=Avg..Area.Number.of.Rooms, y=Price)) +
  geom_point(size=0.4,shape=6)+theme_dark()
plot4<-ggplot(House_data, aes(x=Avg..Area.Number.of.Bedrooms, y=Price)) +
  geom_point(size=0.4,shape=7)+theme_dark()
plot5<-ggplot(House_data, aes(x=Area.Population, y=Price)) +
  geom_point(size=0.4,shape=8)+theme_dark()
plot_grid(plot1,plot2,plot3,plot4,plot5)

corrplot(cor(House_data),number.digits = 4, method="color",order="hclust",
         col=brewer.pal(n=8, name="RdYlBu"),addCoef.col = "black",tl.col="black", tl.srt=45)            #correlation matrix of "House-data"

ggplot(data=House_data, aes(Price)) +
  geom_histogram(aes(y =..density..), fill = "#D95F02") +         #Checking distribution of price in "House_data"
  geom_density()+ggtitle("Distribution of Housing Price")



#Train and test subsets
X<-as.data.frame(scale(House_data[,(1:5)]))
Y<-as.data.frame(House_data[,6])

index <- createDataPartition(House_data$Price, p = .70, list = FALSE)           #70% of random observation in train subset
train <- House_data[index, ]                            #Train dataset
test <- House_data[-index, ]                            #Test dataset
dim(train) # dimension/shape of train dataset

print(head(train))

dim(test)  # dimension/shape of test dataset

print(head(test))

#Building model

lmModel <- lm(Price ~ . , data = train)                # Taining model
# Printing the model object
print(lmModel)

#Validating Regression Coefficients and Models

summary(lmModel)              #if multiple R squared and adjusted R squared value differs to much thata indicates multicollinearity in the dataset


AIC(lmModel)                  #Akaike’s information criterion(overfit)
BIC(lmModel)                  #Bayesian information criterion(underfit)


names(lmModel)                # Checking model object for actual and predicted values 


rmse(actual = train$Price, predicted = lmModel$fitted.values)              #root mean squared error


# Histogram to check the distribution of errors
hist(lmModel$residuals, col = "light blue",freq = F,main = "Distribution of errors")
lines(density(lmModel$residuals),type = "l")                           #the distribution of error follows the normal distribution so the assumption is validate

plot(lmModel)                             #ploting npp plot and scatter plot to check normality and heteroscadasticity


vif_func(House_data[, 1:5])                    #Checking multicollinearity with the help of variation inflation factor test 


#Checking auto serial correlation

dwtest(lmModel)                  #We got a value of 2.013 which suggests that there is no auto serial correlation.

#Predicting Dependent Variable in Test Dataset
test$PreditedPrice <- predict(lmModel, test)              # Predicting Price in test dataset
head(test[ , c("Price", "PreditedPrice")])

#R-Squared Value for the test dataset

actual <- test$Price
preds <- test$PreditedPrice
rss <- sum((preds - actual) ^ 2)                      #Residual sum of squares
tss <- sum((actual - mean(actual)) ^ 2)               #Total sum of squares
ess<-tss-rss;ess                                      #Explained sum of squares
rsq <- 1 - rss/tss;rsq                                    #Coefficient of determination R^2

#In the test dataset, we got an accuracy of  0.911184 and a training data set, we got an accuracy of   0.9207,.