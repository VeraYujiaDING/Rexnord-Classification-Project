#LOGISTIC REGRESSION
#import data
df <- read.csv("Rexnord data.csv")
head(df)
str(df)
library(class)
#convert data types as needed
df$Date <- as.factor(df$Date)
df$Prod <- as.factor(df$Prod)
df$InqOrProj <- as.factor(df$InqOrProj)
df$ProjType <- as.factor(df$ProjType)
df$Type <- as.factor(df$Type)
df$Ind <- as.factor(df$Ind)
df$IndSub <- as.factor(df$IndSub)
df$Replacement <- as.factor(df$Replacement)

df$Date <- as.integer(df$Date)
df$Prod <- as.integer(df$Prod)
df$InqOrProj <- as.integer(df$InqOrProj)
df$ProjType <- as.integer(df$ProjType)
df$Type <- as.integer(df$Type)
df$Ind <- as.integer(df$Ind)
df$IndSub <- as.integer(df$IndSub)
df$Replacement <- as.integer(df$Replacement)
df$LeadTime <- as.integer(df$LeadTime)
df$Outcome[df$Outcome=='Won']<- 1
df$Outcome[df$Outcome=='Lost']<- 0
df$Outcome <- as.integer(df$Outcome)

set.seed(100)

# split train and test dataset
ntotal<-nrow(df)
ntrain <- floor(0.6*ntotal)
nvalid <- floor(0.8*ntotal)-ntrain
df_train<-df[1:ntrain,]
df_valid<-df[(ntrain+1):(ntrain+nvalid),]
df_test<-df[(ntrain+nvalid+1):ntotal,]

#check if some products missing in train/test set
table(df_train$Prod)
table(df_test$Prod) #missing: 12, 20
#drop the two products (Ocelot, Zebra) from both datasets
trainset <- subset(df_train,Prod!=12 & Prod!=20)
validset <- subset(df_valid,Prod!=12 & Prod!=20)
testset <- subset(df_test,Prod!=12 & Prod!=20)

# Training the model, and drop the weak predictors
model_lr <- glm(Outcome ~., family = binomial(), trainset)
summary(model_lr) #AIC: 5127.4
model_lr1 <-glm(Outcome ~ ProjType+Replacement+LeadTime, family = binomial(), trainset)
summary(model_lr1) #AIC: 5125
anova(model_lr1, model_lr, test="Chisq") 
#The insignificant chi-square value (p=0.09) suggests that the reduced model with 3 predictors fits as well as the full model with 11 predictors, 
#reinforcing our belief that variables apart from ProjType, Replacement, and LeadTime don’t add significantly to the prediction above and beyond the other variables in the equation. 
#Therefore, we can base your interpretations on the simpler model.


# Now let's generate predictions
testset$predict<-ifelse(predict(model_lr1,testset, type="response")>0.19, 1, 0)
confusion_final  <- table(testset$predict, testset$Outcome)
confusion_final
#change the cutoff value
overall_accuracy_final <- 1 - (confusion_final[1,2]+confusion_final[2,1])/sum(confusion_final)
overall_accuracy_final 
sensitivity_final <- confusion_final[2,2]/sum(confusion_final[,2])
sensitivity_final 
specificity_final <- confusion_final[1,1]/sum(confusion_final[,1])
specificity_final 

# Now play around with the cutoff value to see if you get something you like better
#change custoff value, and compare overall accuracy, sensitivity and specificity 
#0.22: 0.8301698, 0,          0.9993987
#0.21: 0.8041958, 0.02064897, 0.9639206
#0.2:  0.7877123, 0.05899705, 0.9362598 
#0.19: 0.7652348, 0.1032448,  0.9001804 #pick as cutoff value


#KNN
data<-read.csv("Rexnord data.csv")
library(class)

#we need to convert the data to a factor and then an integer
#if you only convert to integer, all values will be null
#if you only convert to factor, we can not run the knn_predict function
data$Prod <- as.factor(data$Prod)
data$ProjType <- as.factor(data$ProjType)
data$Type <- as.factor(data$Type)
data$Ind <- as.factor(data$Ind)
data$IndSub <- as.factor(data$IndSub)
data$Replacement <- as.factor(data$Replacement)
data$Date <- as.factor(data$Date)
data$InqOrProj <- as.factor(data$InqOrProj)

data$Prod <- as.integer(data$Prod)
data$ProjType <- as.integer(data$ProjType)
data$Type <- as.integer(data$Type)
data$Ind <- as.integer(data$Ind)
data$IndSub <- as.integer(data$IndSub)
data$Replacement <- as.integer(data$Replacement)
data$Date <- as.integer(data$Date)
data$LeadTime <- as.integer(data$LeadTime)
data$InqOrProj <- as.integer(data$InqOrProj)

datax<-data[,1:11]

ntotal<-nrow(data)
ntrain <- floor(0.6*ntotal)
nvalid <- floor(0.8*ntotal)-ntrain

data_train<-datax[1:ntrain,]
data_valid<-datax[(ntrain+1):(ntrain+nvalid),]
data_test<-datax[(ntrain+nvalid+1):ntotal,]

true_train<-data$Outcome[1:ntrain]
true_valid<-data$Outcome[(ntrain+1):(ntrain+nvalid)]
true_test<-data$Outcome[(ntrain+nvalid+1):ntotal]

knn_predict<-knn(data_train,data_valid,true_train,k=6)
knnconfusion <- table(knn_predict, true_valid)
knnconfusion
acc <- 1 - (knnconfusion[1,2]+knnconfusion[2,1])/sum(knnconfusion)
acc #81.28 accuracy k = 6

sensitivity_KNN <- knnconfusion[2,2]/sum(knnconfusion[,2])
specificity_KNN <- knnconfusion[1,1]/sum(knnconfusion[,1])
sensitivity_KNN #3.75
specificity_KNN #96.02

scaledata <- data
scaledata$Prod<-scale(data$Prod)
scaledata$InqOrProj<-scale(data$InqOrProj)
scaledata$ProjType<-scale(data$ProjType)
scaledata$Ind<-scale(data$Ind)
scaledata$IndSub<-scale(data$IndSub)
scaledata$Replacement<-scale(data$Replacement)
scaledata$NetValue<-scale(data$NetValue)
scaledata$Date<-scale(data$Date)
scaledata$LeadTime<-scale(data$LeadTime)
scaledata$CustNum<-scale(data$CustNum)

scaledatax<-scaledata[,1:11]

scaledata_train<-scaledatax[1:ntrain,]
scaledata_valid<-scaledatax[(ntrain+1):(ntrain+nvalid),]
scaledata_test<-scaledatax[(ntrain+nvalid+1):ntotal,]

scaletrue_train<-scaledata$Outcome[1:ntrain]
scaletrue_valid<-scaledata$Outcome[(ntrain+1):(ntrain+nvalid)]
scaletrue_test<-scaledata$Outcome[(ntrain+nvalid+1):ntotal]

scaleknn_predict<-knn(scaledata_train,scaledata_valid,scaletrue_train,k=6)
scaleknnconfusion <- table(scaleknn_predict, scaletrue_valid)
scaleknnconfusion
scaleacc <- 1 - (scaleknnconfusion[1,2]+scaleknnconfusion[2,1])/sum(scaleknnconfusion)
scaleacc #81.18 accuracy k = 6

scalesensitivity_KNN <- scaleknnconfusion[2,2]/sum(scaleknnconfusion[,2])
scalespecificity_KNN <- scaleknnconfusion[1,1]/sum(scaleknnconfusion[,1])
scalesensitivity_KNN #8.75
scalespecificity_KNN #95.72

#Association Rules

rm(list=ls())
#import data
df <- read.csv("Rexnord data.csv")
head(df)
str(df)

#A.break buckets of NetValue 
df$NetValueGroup <- 
  ifelse(df$NetValue<2000, 1, 
         ifelse(df$NetValue>=2000 & df$NetValue<5000 ,2,
                ifelse(df$NetValue>=5000 & df$NetValue<10000,3,
                       ifelse(df$NetValue>=10000 & df$NetValue<50000,4,
                              ifelse(df$NetValue>=50000 & df$NetValue<100000,5,6)))))
df$NetValueGroup

#A.break buckets of Leadtime
df$LeadTime <- as.integer(df$LeadTime)
df$LeadTimeGroup <- 
  ifelse(df$LeadTime<2,1,
         ifelse(df$LeadTime>=2 & df$LeadTime<6,2,
                ifelse(df$LeadTime>=6 & df$LeadTime<9,3,
                       ifelse(df$LeadTime>=9 & df$LeadTime<13,4,
                              ifelse(df$LeadTime>=13 & df$LeadTime<17,5,
                                     ifelse(df$LeadTime>=17 & df$LeadTime<21,6,
                                            ifelse(df$LeadTime>=21 & df$LeadTime<26,7,8)))))))


#convert data types to factor variables
df$Date <- as.factor(df$Date)
df$Prod <- as.factor(df$Prod)
df$InqOrProj <- as.factor(df$InqOrProj)
df$ProjType <- as.factor(df$ProjType)
df$Type <- as.factor(df$Type)
df$Ind <- as.factor(df$Ind)
df$IndSub <- as.factor(df$IndSub)
df$Replacement <- as.factor(df$Replacement)
df$LeadTimeGroup <- as.factor(df$LeadTimeGroup)
df$NetValueGroup <- as.factor(df$NetValueGroup)
df$Outcome <- as.factor(df$Outcome)
#select column
df_prep <- df[c(1:7, 12:14)]
str(df_prep)
# Association rules will use the arules package
#install.packages("arules")
library(arules)

# In order to run the association rules algorithm, the data frame of factors needs to be converted to 
# a list of transactions. 

df_trans<-as(df_prep, "transactions")

# The following line actually runs the algorithm to generate the rules
# minlen is the minimum "length" of a rule allowed
# A single antecedent implying a single consequent results in a length of 2
# The other parameters give minimum levels required for the different metrics
# maxlen is the total number of variables can link together
dfrules<-apriori(df_trans, parameter=list(supp=0.05, conf=0.05, minlen=2, maxlen=10))

# We can get a summary, including the number of rules generated, with the following statement

summary(dfrules)

# There are many many rules - probably too many to manage
# Let's subset them a bit to look at only interesting ones
outcome_rules<-subset(dfrules, subset = rhs %in% "Outcome=Won" & lift > 1)
summary(outcome_rules)

#d) outcome=lost (repeat part c)
outcome_rules_d<-subset(dfrules, subset = rhs %in% "Outcome=Lost" & lift > 1.05)
summary(outcome_rules_d)
#57 rules for >1.05

#e) sorting
outcome_rules<-sort(outcome_rules, decreasing=TRUE, by="lift")
inspect(outcome_rules)
#5.54% makes up first bucket
#only 18.5% probability of consequent given antecedent
#18.5% of distributors with nvg=4 have outcome=won
#first 7 rules have same values/probabilities
#1.17 lift means 1.17 times more likely to have outcome=won given the bucket
outcome_rules_d<-sort(outcome_rules_d, decreasing=TRUE, by="lift")
inspect(outcome_rules_d)
#5.56% makes up first bucket
#93.9% probability of consequent given antecedent (useful info)
#93.9% of inquiry, distributor, nvg=6 have outcome=lost
#first 4 rules have same values/probabilities
#rules 5-7 have similar probabilities, but off by a few decimals
#1.12 lift means 1.12 times more likely to have outcome=lost given the bucket


