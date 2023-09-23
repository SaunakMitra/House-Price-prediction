#removing all list of elements stored before in the environment
rm(list=ls(all=TRUE))

#Getting the working directory to store the dataset (in csv format)
getwd()

#Importing the dataset
House_data<-read.csv(file="C:\\Users\\HP\\Desktop\\House price prediction\\Dataset\\USA_Housing.csv",header = T,sep = ",")

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
ggcorr(train, label = T, hjust = 1, layout.exp = 3)
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


install.packages("fmsb")
vif_func<-function(in_frame,thresh=10,trace=T,...){
  
  library(fmsb)
  
  if(any(!'data.frame' %in% class(in_frame))) in_frame<-data.frame(in_frame)
  
  #get initial vif value for all comparisons of variables
  vif_init<-NULL
  var_names <- names(in_frame)
  for(val in var_names){
    regressors <- var_names[-which(var_names == val)]
    form <- paste(regressors, collapse = '+')
    form_in <- formula(paste(val, '~', form))
    vif_init<-rbind(vif_init, c(val, VIF(lm(form_in, data = in_frame, ...))))
  }
  vif_max<-max(as.numeric(vif_init[,2]), na.rm = TRUE)
  
  if(vif_max < thresh){
    if(trace==T){ #print output of each iteration
      prmatrix(vif_init,collab=c('var','vif'),rowlab=rep('',nrow(vif_init)),quote=F)
      cat('\n')
      cat(paste('All variables have VIF < ', thresh,', max VIF ',round(vif_max,2), sep=''),'\n\n')
    }
    return(var_names)
  }
  else{
    
    in_dat<-in_frame
    
    #backwards selection of explanatory variables, stops when all VIF values are below 'thresh'
    while(vif_max >= thresh){
      
      vif_vals<-NULL
      var_names <- names(in_dat)
      
      for(val in var_names){
        regressors <- var_names[-which(var_names == val)]
        form <- paste(regressors, collapse = '+')
        form_in <- formula(paste(val, '~', form))
        vif_add<-VIF(lm(form_in, data = in_dat, ...))
        vif_vals<-rbind(vif_vals,c(val,vif_add))
      }
      max_row<-which(vif_vals[,2] == max(as.numeric(vif_vals[,2]), na.rm = TRUE))[1]
      
      vif_max<-as.numeric(vif_vals[max_row,2])
      
      if(vif_max<thresh) break
      
      if(trace==T){ #print output of each iteration
        prmatrix(vif_vals,collab=c('var','vif'),rowlab=rep('',nrow(vif_vals)),quote=F)
        cat('\n')
        cat('removed: ',vif_vals[max_row,1],vif_max,'\n\n')
        flush.console()
      }
      
      in_dat<-in_dat[,!names(in_dat) %in% vif_vals[max_row,1]]
      
    }
    
    return(names(in_dat))
    
  }
  
}
vif_func(House_data[, 1:5])                    #Checking multicollinearity with the help of variation inflation factor test 


#Checking auto serial correlation

dwtest(lmModel)                  #We got a value of 2.013 which suggests that there is no auto serial correlation.

#Predicting Dependent Variable in Test Dataset
test$PreditedPrice <- predict(lmModel, test)              # Predicting Price in test dataset

head(test[ , c("Price", "PreditedPrice")])

#R-Squared Value for the test dataset

actual <- test$Price;actual
preds <- test$PreditedPrice;preds
rss <- sum((preds - actual) ^ 2);rss                      #Residual sum of squares
tss <- sum((actual - mean(actual)) ^ 2);tss               #Total sum of squares
ess<-tss-rss;ess                                      #Explained sum of squares
rsq <- 1 - rss/tss;rsq                                    #Coefficient of determination R^2

#In the test dataset, we got an accuracy of  0.911184 and a training data set, we got an accuracy of   0.9207,.
shapiro.test(x =lmModel$residuals)
library(lmtest)

bptest(lmModel)
plot(lmModel$fitted.values,lmModel$residuals)
abline(h = 0, col = "red")
 
  ggplot(mapping = aes(x = preds, y =actual)) +
  geom_point(size = 1.6, color = "blue") +
  # Overlay a regression line
  geom_smooth(method = "lm", se = F, color = 'red') +
  ggtitle("Housing price Predictions") +
  xlab("Actual price") +
  ylab("Predicted price") +
  theme(plot.title = element_text(hjust = 0.5))
  
  #Train and test subsets
new_X<-as.data.frame(scale(House_data[,c(1,2,3,5)]));new_X
new_Y<-as.data.frame(House_data[,6]);new_Y
new_House_data<-House_data[,c(1,2,3,5,6)];new_House_data


index <- createDataPartition(new_House_data$Price, p = .70, list = FALSE)           #70% of random observation in train subset
new_train <- new_House_data[index, ]                            #Train dataset
new_test <- new_House_data[-index, ]                            #Test dataset
dim(new_train) # dimension/shape of train dataset

print(head(new_train))

dim(new_test)  # dimension/shape of test dataset

print(head(new_test))
ggcorr(new_train, label = T, hjust = 1, layout.exp = 3)
#Building model

lmModel1 <- lm(Price ~ . , data =new_train)                # Taining model

# Printing the model object
print(lmModel1)

#Validating Regression Coefficients and Models

summary(lmModel1)              #if multiple R squared and adjusted R squared value differs to much thata indicates multicollinearity in the dataset


AIC(lmModel1)                  #Akaike’s information criterion(overfit)
BIC(lmModel1)                  #Bayesian information criterion(underfit)


names(lmModel1)                # Checking model object for actual and predicted values 

library(Metrics)
rmse(as.numeric(new_train$Price),as.numeric(lmModel1$fitted.values))              #root mean squared error
str(as.numeric(lmModel1$fitted.values))

# Histogram to check the distribution of errors
hist(lmModel1$residuals, col = "light blue",freq = F,main = "Distribution of errors")
lines(density(lmModel1$residuals),type = "l")                           #the distribution of error follows the normal distribution so the assumption is validate

plot(lmModel1)                             #ploting npp plot and scatter plot to check normality and heteroscadasticity

#Checking auto serial correlation

dwtest(lmModel1)                  #We got a value of 2.013 which suggests that there is no auto serial correlation.

#Predicting Dependent Variable in Test Dataset
new_test$PreditedPrice <- predict(lmModel1,new_test)              # Predicting Price in test dataset

head(new_test[ , c("Price", "PreditedPrice")])

#R-Squared Value for the test dataset

actual1 <-new_test$Price;actual1
preds1 <-new_test$PreditedPrice;preds1
rss1 <- sum((preds1 - actual1) ^ 2);rss1                   #Residual sum of squares
tss1<- sum((actual1 - mean(actual1)) ^ 2);tss1               #Total sum of squares
ess1<-tss1-rss1;ess1                                      #Explained sum of squares
rsq1<- 1 - rss1/tss1;rsq1                                  #Coefficient of determination R^2

ggplot(mapping = aes(x = preds1, y =actual1)) +
  geom_point(size = 1.6, color = "blue") +
  # Overlay a regression line
  geom_smooth(method = "lm", se = F, color = 'red') +
  ggtitle("Housing price Predictions") +
  xlab("Actual price") +
  ylab("Predicted price") +
  theme(plot.title = element_text(hjust = 0.5))
