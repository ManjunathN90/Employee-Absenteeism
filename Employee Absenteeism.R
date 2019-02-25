rm(list=ls(all=T))

getwd()

#Load Libraries
x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information",
      "MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees','readxl')

#install.packages('readxl')
lapply(x, require, character.only = TRUE)
rm(x)

## Read the Data
df = read.csv("Absent_to_work.csv", header = T, na.strings = c(" ", "", "NA"))

#Check number of unique values in each column
for(i in 1:ncol(df))
{
  print(colnames(df[i]))
  print(length(unique(df[,i])))
}


######## Missing Values #############
missing_val = data.frame(apply(df,2,function(x){sum(is.na(x))}))
missing_val$Columns = row.names(missing_val)
names(missing_val)[1] =  "Missing_percentage"
missing_val$Missing_percentage = (missing_val$Missing_percentage/nrow(df)) * 100
missing_val = missing_val[order(-missing_val$Missing_percentage),]
row.names(missing_val) = NULL
missing_val = missing_val[,c(2,1)]

#Convert it to dataframe
df = as.data.frame(df)


# kNN Imputation
df = knnImputation(df, k=3)



######### Outlier Analysis ######################

cnames = c('Transportation.expense', 'Distance.from.Residence.to.Work','Service.time', 'Age', 'Work.load.Average.day'
            ,'Hit.target','Weight','Height','Body.mass.index','Absenteeism.time.in.hours')



for (i in 1:length(cnames))
{
  assign(paste0("gn",i), ggplot(aes_string(y = (cnames[i]), x = "Absenteeism.time.in.hours"), data = subset(df))+
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
                        outlier.size=1, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(y=cnames[i],x="Absenteeism.time.in.hours")+
           ggtitle(paste("Box plot for",cnames[i])))
}


## Plotting plots together
gridExtra::grid.arrange(gn1,gn2,ncol=2)
gridExtra::grid.arrange(gn3,gn4,ncol=2)
gridExtra::grid.arrange(gn5,gn6,ncol=2)
gridExtra::grid.arrange(gn7,gn8,ncol=2)
gridExtra::grid.arrange(gn9,gn10,ncol=2)



for(i in cnames)
{
  val = df[,i][df[,i] %in% boxplot.stats(df[,i])$out]
  df[,i][df[,i] %in% val] = NA
  
}
df = knnImputation(df, k = 3)



############## Feature Selection ##########################


corrgram(df[,cnames], order = F, upper.panel = panel.pie, text.panel = panel.txt, main = "Correlation Plot")

##Dimensionality Reduction
df = subset(df, select = -c(Body.mass.index ))


############## Feature Scaling ###############################

for(i in colnames(df))
{
  print(i)
  df[,i] = (df[,i] - min(df[,i]))/ (max(df[,i] - min(df[,i])))
}


############# Model Development ##################################

train_index = sample(1:nrow(df), 0.8 * nrow(df))
train = df[train_index,]
test = df[-train_index,]


#### r part regression - Decision Tree
fit_DT = rpart(Absenteeism.time.in.hours ~., data = train, method = "anova")

#predict new case
predictions = predict(fit, test[,-20])
summary(fit_DT)


rmse = function(error)
{
  sqrt(mean(error^2))
}

mae = function(error)
{
  mean(abs(error))
}


error = test$Absenteeism.time.in.hours - predictions
rmse(error)
mae(error)

### Decision Tree 

#RMSE = 0.171
#MAE = 0.118




#### Random Forest

#Develop Model on training data
fit_RF = randomForest(Absenteeism.time.in.hours~., data = train)

#Lets predict for training data
pred_RF= predict(fit_RF, test[,-20])

error = test$Absenteeism.time.in.hours - pred_RF
rmse(error)
mae(error)


### Random Forest 

#RMSE = 0.162
#MAE = 0.115


#### Linear Regression

#Develop Model on training data
fit_LR = lm(Absenteeism.time.in.hours ~ ., data = train)

#Lets predict for training data
pred_LR = predict(fit_LR, test[,-20])



error = test$Absenteeism.time.in.hours - pred_LR
rmse(error)
mae(error)


### Linear Regression 

#RMSE = 0.175
#MAE = 0.127



