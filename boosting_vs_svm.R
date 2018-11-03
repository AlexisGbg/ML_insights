# DATA PREPARATION ----
setwd("/Users/gbg/Documents/Projets/GitHub/boosting_vs_svm")
data <- read.csv("data/adult.data", header = F)
colnames(data) <- c("age", 
                    "workclass", 
                    "fnlwgt", 
                    "education", 
                    "education_num", 
                    "marital_status", 
                    "occupation", 
                    "relationship", 
                    "race", 
                    "sex", 
                    "capital_gain", 
                    "capital_loss", 
                    "hours_per_week", 
                    "native_contry", 
                    "income")


hist(data$hours_per_week)
