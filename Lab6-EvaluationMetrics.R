# *****************************************************************************
# Lab 5: Resampling Methods ----
#
# Course Code: BBT4206
# Course Name: Business Intelligence II
# Semester Duration: 21st August 2023 to 28th November 2023
#
# Lecturer: Allan Omondi
# Contact: aomondi [at] strathmore.edu
#
# Note: The lecture contains both theory and practice. This file forms part of
#       the practice. It has required lab work submissions that are graded for
#       coursework marks.
#
# License: GNU GPL-3.0-or-later
# See LICENSE file for licensing information.
# *****************************************************************************

# **[OPTIONAL] Initialization: Install and use renv ----
# The R Environment ("renv") package helps you create reproducible environments
# for your R projects. This is helpful when working in teams because it makes
# your R projects more isolated, portable and reproducible.

# Further reading:
#   Summary: https://rstudio.github.io/renv/
#   More detailed article: https://rstudio.github.io/renv/articles/renv.html

# "renv" It can be installed as follows:
# if (!is.element("renv", installed.packages()[, 1])) {
# install.packages("renv", dependencies = TRUE,
# repos = "https://cloud.r-project.org") # nolint
# }
# require("renv") # nolint

# Once installed, you can then use renv::init() to initialize renv in a new
# project.

# The prompt received after executing renv::init() is as shown below:
# This project already has a lockfile. What would you like to do?

# 1: Restore the project from the lockfile.
# 2: Discard the lockfile and re-initialize the project.
# 3: Activate the project without snapshotting or installing any packages.
# 4: Abort project initialization.

# Select option 1 to restore the project from the lockfile
# renv::init() # nolint

# This will set up a project library, containing all the packages you are
# currently using. The packages (and all the metadata needed to reinstall
# them) are recorded into a lockfile, renv.lock, and a .Rprofile ensures that
# the library is used every time you open the project.

# Consider a library as the location where packages are stored.
# Execute the following command to list all the libraries available in your
# computer:
.libPaths()

# One of the libraries should be a folder inside the project if you are using
# renv

# Then execute the following command to see which packages are available in
# each library:
lapply(.libPaths(), list.files)

# This can also be configured using the RStudio GUI when you click the project
# file, e.g., "BBT4206-R.Rproj" in the case of this project. Then
# navigate to the "Environments" tab and select "Use renv with this project".

# As you continue to work on your project, you can install and upgrade
# packages, using either:
# install.packages() and update.packages or
# renv::install() and renv::update()

# You can also clean up a project by removing unused packages using the
# following command: renv::clean()

# After you have confirmed that your code works as expected, use
# renv::snapshot(), AT THE END, to record the packages and their
# sources in the lockfile.

# Later, if you need to share your code with someone else or run your code on
# a new machine, your collaborator (or you) can call renv::restore() to
# reinstall the specific package versions recorded in the lockfile.

# [OPTIONAL]
# Execute the following code to reinstall the specific package versions
# recorded in the lockfile (restart R after executing the command):
# renv::restore() # nolint

# [OPTIONAL]
# If you get several errors setting up renv and you prefer not to use it, then
# you can deactivate it using the following command (restart R after executing
# the command):
# renv::deactivate() # nolint

# If renv::restore() did not install the "languageserver" package (required to
# use R for VS Code), then it can be installed manually as follows (restart R
# after executing the command):

if (require("languageserver")) {
  require("languageserver")
} else {
  install.packages("languageserver", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

# Introduction ----
# Resampling methods are techniques that can be used to improve the performance
# and reliability of machine learning algorithms. They work by creating
# multiple training sets from the original training set. The model is then
# trained on each training set, and the results are averaged. This helps to
# reduce overfitting and improve the model's generalization performance.

# Resampling methods include:
## Splitting the dataset into train and test sets ----
## Bootstrapping (sampling with replacement) ----
## Basic k-fold cross validation ----
## Repeated cross validation ----
## Leave One Out Cross-Validation (LOOCV) ----

# STEP 1. Install and Load the Required Packages ----
## caret ----
if (require("caret")) {
  require("caret")
} else {
  install.packages("caret", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## klaR ----
if (require("klaR")) {
  require("klaR")
} else {
  install.packages("klaR", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## e1071 ----
if (require("e1071")) {
  require("e1071")
} else {
  install.packages("e1071", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## readr ----
if (require("readr")) {
  require("readr")
} else {
  install.packages("readr", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## LiblineaR ----
if (require("LiblineaR")) {
  require("LiblineaR")
} else {
  install.packages("LiblineaR", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## naivebayes ----
if (require("naivebayes")) {
  require("naivebayes")
} else {
  install.packages("naivebayes", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

# DATASET 1 (Splitting the dataset): Dow Jones Index ----
stock_ror_dataset <- read_csv(
  "data/transforms/dow_jones_index.csv",
  col_types = cols(
    stock = col_factor(
      levels = c(
        "AA",
        "AXP",
        "BA",
        "BAC",
        "CAT",
        "CSCO",
        "CVX",
        "DD",
        "DIS",
        "GE",
        "HD",
        "HPQ",
        "IBM",
        "INTC",
        "JNJ",
        "JPM",
        "KRFT",
        "KO",
        "MCD",
        "MMM",
        "MRK",
        "MSFT",
        "PFE",
        "PG",
        "T",
        "TRV",
        "UTX",
        "VZ",
        "WMT",
        "XOM"
      )
    ),
    date = col_date(format = "%m/%d/%Y")
  )
)
summary(stock_ror_dataset)

# The str() function is used to compactly display the structure (variables
# and data types) of the dataset
str(stock_ror_dataset)

## 1. Split the dataset ====
# Define a 75:25 train:test data split of the dataset.
# That is, 75% of the original data will be used to train the model and
# 25% of the original data will be used to test the model.
train_index <- createDataPartition(stock_ror_dataset$stock,
                                   p = 0.75,
                                   list = FALSE)
stock_ror_dataset_train <- stock_ror_dataset[train_index, ]
stock_ror_dataset_test <- stock_ror_dataset[-train_index, ]

## 2. Train a Naive Bayes classifier using the training dataset ----

### 2.a. OPTION 1: naiveBayes() function in the e1071 package ----
# The "naiveBayes()" function (case sensitive) in the "e1071" package
# is less sensitive to missing values hence all the features (variables
# /attributes) are considered as independent variables that have an effect on
# the dependent variable (stock).

stock_ror_dataset_model_nb_e1071 <- # nolint
  e1071::naiveBayes(stock ~ quarter + date + open + high + low + close +
                      volume + percent_change_price +
                      percent_change_volume_over_last_wk +
                      previous_weeks_volume + next_weeks_open +
                      next_weeks_close + percent_change_next_weeks_price +
                      days_to_next_dividend + percent_return_next_dividend,
                    data = stock_ror_dataset_train)

# The above code can also be written as follows to show a case where all the
# variables are being considered (stock ~ .):
stock_ror_dataset_model_nb <-
  e1071::naiveBayes(stock ~ .,
                    data = stock_ror_dataset_train)

### 2.b. OPTION 2: naiveBayes() function in the caret package ====
# The second option uses the caret::train() function in the caret package to
# train a Naive Bayes classifier but without the attributes that have missing
# values.
stock_ror_dataset_model_nb_caret <- # nolint
  caret::train(stock ~ ., data =
               stock_ror_dataset_train[, c("quarter", "date", "open",
                                           "high", "low", "close",
                                           "volume",
                                           "percent_change_price",
                                           "next_weeks_open",
                                           "next_weeks_close",
                                           "percent_change_next_weeks_price",
                                           "days_to_next_dividend",
                                           "percent_return_next_dividend",
                                           "stock")],
               method = "naive_bayes")

## 3. Test the trained model using the testing dataset ----
### 3.a. Test the trained e1071 Naive Bayes model using the testing dataset ----
predictions_nb_e1071 <-
  predict(stock_ror_dataset_model_nb_e1071,
          stock_ror_dataset_test[, c("quarter", "date", "open", "high",
                                     "low", "close", "volume",
                                     "percent_change_price",
                                     "percent_change_volume_over_last_wk",
                                     "previous_weeks_volume", "next_weeks_open",
                                     "next_weeks_close",
                                     "percent_change_next_weeks_price",
                                     "days_to_next_dividend",
                                     "percent_return_next_dividend")])

### 3.b. Test the trained caret Naive Bayes model using the testing dataset ----
predictions_nb_caret <-
  predict(stock_ror_dataset_model_nb_caret,
          stock_ror_dataset_test[, c("quarter", "date", "open", "high",
                                     "low", "close", "volume",
                                     "percent_change_price", "next_weeks_open",
                                     "next_weeks_close",
                                     "percent_change_next_weeks_price",
                                     "days_to_next_dividend",
                                     "percent_return_next_dividend")])

## 4. View the Results ----
### 4.a. e1071 Naive Bayes model and test results using a confusion matrix ----
# Please watch the following video first: https://youtu.be/Kdsp6soqA7o
print(predictions_nb_e1071)
caret::confusionMatrix(predictions_nb_e1071,
                       stock_ror_dataset_test[, c("quarter", "date", "open",
                                                  "high", "low", "close",
                                                  "volume",
                                                  "percent_change_price",
                                                  "percent_change_volume_over_last_wk", # nolint
                                                  "previous_weeks_volume",
                                                  "next_weeks_open",
                                                  "next_weeks_close",
                                                  "percent_change_next_weeks_price", # nolint
                                                  "days_to_next_dividend",
                                                  "percent_return_next_dividend", # nolint
                                                  "stock")]$stock)
plot(table(predictions_nb_e1071,
           stock_ror_dataset_test[, c("quarter", "date", "open", "high", "low",
                                      "close", "volume", "percent_change_price",
                                      "percent_change_volume_over_last_wk",
                                      "previous_weeks_volume",
                                      "next_weeks_open", "next_weeks_close",
                                      "percent_change_next_weeks_price",
                                      "days_to_next_dividend",
                                      "percent_return_next_dividend",
                                      "stock")]$stock))

### 4.b. caret Naive Bayes model and test results using a confusion matrix ----
print(stock_ror_dataset_model_nb_caret)
caret::confusionMatrix(predictions_nb_caret,
                       stock_ror_dataset_test[, c("quarter", "date", "open",
                                                  "high", "low", "close",
                                                  "volume",
                                                  "percent_change_price",
                                                  "percent_change_volume_over_last_wk", # nolint
                                                  "previous_weeks_volume",
                                                  "next_weeks_open",
                                                  "next_weeks_close",
                                                  "percent_change_next_weeks_price", # nolint
                                                  "days_to_next_dividend",
                                                  "percent_return_next_dividend", # nolint
                                                  "stock")]$stock)
plot(table(predictions_nb_caret,
           stock_ror_dataset_test[, c("quarter", "date", "open", "high", "low",
                                      "close", "volume", "percent_change_price",
                                      "percent_change_volume_over_last_wk",
                                      "previous_weeks_volume",
                                      "next_weeks_open", "next_weeks_close",
                                      "percent_change_next_weeks_price",
                                      "days_to_next_dividend",
                                      "percent_return_next_dividend",
                                      "stock")]$stock))

# DATASET 2 (Splitting the dataset): Default of credit card clients ----
defaulter_dataset <-
  readr::read_csv(
    "data/default of credit card clients.csv",
    col_types = cols(
      SEX = col_factor(levels = c("1", "2")),
      EDUCATION = col_factor(levels = c("0", "1", "2", "3", "4", "5", "6")),
      MARRIAGE = col_factor(levels = c("0", "1", "2", "3")),
      `default payment next month` = col_factor(levels = c("1", "0")),
      `default payment next month` = col_factor(levels = c("1", "0"))
    ),
    skip = 1
  )
summary(defaulter_dataset)
str(defaulter_dataset)

## 1. Split the dataset ----
# Define an 80:20 train:test split ratio of the dataset
# (80% of the original data will be used to train the model and 20% of the
# original data will be used to test the model).
train_index <- createDataPartition(defaulter_dataset$`default payment next month`, # nolint
                                   p = 0.80, list = FALSE)
defaulter_dataset_train <- defaulter_dataset[train_index, ]
defaulter_dataset_test <- defaulter_dataset[-train_index, ]

## 2. Train a Naive Bayes classifier using the training dataset ----

### 2.a. OPTION 1: "NaiveBayes()" function in the "klaR" package ----
defaulter_dataset_model_nb_klaR <- # nolint
  klaR::NaiveBayes(`default payment next month` ~ .,
                   data = defaulter_dataset_train)

### 2.b. OPTION 2: "naiveBayes()" function in the e1071 package ----
defaulter_dataset_model_nb_e1071 <- # nolint
  e1071::naiveBayes(`default payment next month` ~ .,
                    data = defaulter_dataset_train)

## 3. Test the trained Naive Bayes model using the testing dataset ----
predictions_nb_e1071 <-
  predict(defaulter_dataset_model_nb_e1071,
          defaulter_dataset_test[, 1:25])

## 4. View the Results ----
### 4.a. e1071 Naive Bayes model and test results using a confusion matrix ----
print(defaulter_dataset_model_nb_e1071)
caret::confusionMatrix(predictions_nb_e1071,
                       defaulter_dataset_test$`default payment next month`)
# The confusion matrix can also be viewed graphically,
# although with less information.
plot(table(predictions_nb_e1071,
           defaulter_dataset_test$`default payment next month`))

# DATASET 3 (Bootstrapping): Daily Demand Forecasting Orders Data Set =====
demand_forecasting_dataset <-
  readr::read_delim(
    "data/Daily_Demand_Forecasting_Orders.csv",
    delim = ";",
    escape_double = FALSE,
    col_types = cols(
      `Week of the month (first week, second, third, fourth or fifth week` =
        col_factor(levels = c("1", "2", "3", "4", "5")),
      `Day of the week (Monday to Friday)` =
        col_factor(levels = c("2", "3", "4", "5", "6"))
    ),
    trim_ws = TRUE
  )
summary(demand_forecasting_dataset)
str(demand_forecasting_dataset)

## 1. Split the dataset ----
demand_forecasting_dataset_cor <- cor(demand_forecasting_dataset[, 3:13])
View(demand_forecasting_dataset_cor)
# Define a 75:25 train:test data split ratio of the dataset
# (75% of the original data will be used to train the model and 25% of the
# original data will be used to test the model)
train_index <-
  createDataPartition(demand_forecasting_dataset$`Target (Total orders)`,
                      p = 0.75, list = FALSE)
demand_forecasting_dataset_train <- demand_forecasting_dataset[train_index, ] # nolint
demand_forecasting_dataset_test <- demand_forecasting_dataset[-train_index, ] # nolint

## 2. Train a linear regression model (for regression) ----

### 2.a. Bootstrapping train control ----
# The "train control" allows you to specify that bootstrapping (sampling with
# replacement) can be used and also the number of times (repetitions or reps)
# the sampling with replacement should be done. The code below specifies
# bootstrapping with 500 reps. (common values for reps are thousands or tens of
# thousands depending on the hardware resources available).

# This increases the size of the training dataset from 48 observations to
# approximately 48 x 500 = 24,000 observations for training the model.
train_control <- trainControl(method = "boot", number = 500)

demand_forecasting_dataset_model_lm <- # nolint
  caret::train(`Target (Total orders)` ~
                 `Non-urgent order` + `Urgent order` +
                   `Order type A` + `Order type B` +
                   `Order type C` + `Fiscal sector orders` +
                   `Orders from the traffic controller sector` +
                   `Banking orders (1)` + `Banking orders (2)` +
                   `Banking orders (3)`,
               data = demand_forecasting_dataset_train,
               trControl = train_control,
               na.action = na.omit, method = "lm", metric = "RMSE")

## 3. Test the trained linear regression model using the testing dataset ----
predictions_lm <- predict(demand_forecasting_dataset_model_lm,
                          demand_forecasting_dataset_test[, 1:13])

## 4. View the RMSE and the predicted values for the 12 observations ----
print(demand_forecasting_dataset_model_lm)
print(predictions_lm)

## 5. Use the model to make a prediction on unseen new data ----
# New data for each of the 12 variables (independent variables) that determine
# the dependent variable can also be specified as follows in a data frame:
new_data <-
  data.frame(`Week of the month (first week, second, third, fourth or fifth week` = c(1), # nolint
             `Day of the week (Monday to Friday)` = c(2),
             `Non-urgent order` = c(151.06),
             `Urgent order` = c(132.11), `Order type A` = c(52.11),
             `Order type B` = c(109.23),
             `Order type C` = c(160.11), `Fiscal sector orders` = c(7.832),
             `Orders from the traffic controller sector` = c(52112),
             `Banking orders (1)` = c(20130), `Banking orders (2)` = c(94788),
             `Banking orders (3)` = c(12610), check.names = FALSE)

# The variables that are factors (categorical) in the training dataset must
# also be defined as factors in the new data
new_data$`Week of the month (first week, second, third, fourth or fifth week` <-
  as.factor(new_data$`Week of the month (first week, second, third, fourth or fifth week`) # nolint

new_data$`Day of the week (Monday to Friday)` <-
  as.factor(new_data$`Day of the week (Monday to Friday)`)

# We now use the model to predict the output based on the unseen new data:
predictions_lm_new_data <-
  predict(demand_forecasting_dataset_model_lm, new_data)

# The output below refers to the total orders:
print(predictions_lm_new_data)

# DATASET 4 (CV, Repeated CV, and LOOCV): Iranian Churn Dataset ----
churn_dateset <- read_csv(
  "data/Customer Churn.csv",
  col_types = cols(
    Complains = col_factor(levels = c("0",
                                      "1")),
    `Age Group` = col_factor(levels = c("1",
                                        "2", "3", "4", "5")),
    `Tariff Plan` = col_factor(levels = c("1",
                                          "2")),
    Status = col_factor(levels = c("1",
                                   "2")),
    Churn = col_factor(levels = c("0",
                                  "1"))
  )
)
summary(churn_dateset)
str(churn_dateset)

## 1. Split the dataset ====
# define a 75:25 train:test split of the dataset
train_index <- createDataPartition(churn_dateset$`Customer Value`,
                                   p = 0.75, list = FALSE)
churn_dateset_train <- churn_dateset[train_index, ]
churn_dateset_test <- churn_dateset[-train_index, ]

## 2. Regression: Linear Model ----
### 2.a. 10-fold cross validation ----

# Please watch the following video first: https://youtu.be/fSytzGwwBVw
# The train control allows you to specify that k-fold cross validation
# can be used as well as the number of folds (common folds are 5-fold and
# 10-fold cross validation).

# The k-fold cross-validation method involves splitting the dataset (training
# dataset) into k-subsets. Each subset is held-out (withheld) while the model is
# trained on all other subsets. This process is repeated until the accuracy/RMSE
# is determined for each instance in the dataset, and an overall accuracy/RMSE
# estimate is provided.

train_control <- trainControl(method = "cv", number = 10)

churn_dateset_model_lm <-
  caret::train(`Customer Value` ~ .,
               data = churn_dateset_train,
               trControl = train_control, na.action = na.omit,
               method = "lm", metric = "RMSE")

### 2.b. Test the trained linear model using the testing dataset ----
predictions_lm <- predict(churn_dateset_model_lm, churn_dateset_test[, -13])

### 2.c. View the RMSE and the predicted values ====
print(churn_dateset_model_lm)
print(predictions_lm)

## 3. Classification: LDA with k-fold Cross Validation ----

### 3.a. LDA classifier based on a 5-fold cross validation ----
# We train a Linear Discriminant Analysis (LDA) classifier based on a 5-fold
# cross validation train control but this time, using the churn variable for
# classification, not the customer value variable for regression.
train_control <- trainControl(method = "cv", number = 5)

churn_dateset_model_lda <-
  caret::train(`Churn` ~ ., data = churn_dateset_train,
               trControl = train_control, na.action = na.omit, method = "lda2",
               metric = "Accuracy")

### 3.b. Test the trained LDA model using the testing dataset ----
predictions_lda <- predict(churn_dateset_model_lda,
                           churn_dateset_test[, 1:13])

### 3.c. View the summary of the model and view the confusion matrix ----
print(churn_dateset_model_lda)
caret::confusionMatrix(predictions_lda, churn_dateset_test$Churn)

## 4. Classification: Naive Bayes with Repeated k-fold Cross Validation ----
### 4.a. Train an e1071::naive Bayes classifier based on the churn variable ----
churn_dateset_model_nb <-
  e1071::naiveBayes(`Churn` ~ ., data = churn_dateset_train)

### 4.b. Test the trained naive Bayes classifier using the testing dataset ----
predictions_nb_e1071 <-
  predict(churn_dateset_model_nb, churn_dateset_test[, 1:14])

### 4.c. View a summary of the naive Bayes model and the confusion matrix ----
print(churn_dateset_model_nb)
caret::confusionMatrix(predictions_nb_e1071, churn_dateset_test$Churn)

## 5. Classification: SVM with Repeated k-fold Cross Validation ----
### 5.a. SVM Classifier using 5-fold cross validation with 3 reps ----
# We train a Support Vector Machine (for classification) using "Churn" variable
# in the training dataset based on a repeated 5-fold cross validation train
# control with 3 reps.

# The repeated k-fold cross-validation method involves repeating the number of
# times the dataset is split into k-subsets. The final model accuracy/RMSE is
# taken as the mean from the number of repeats.

train_control <- trainControl(method = "repeatedcv", number = 5, repeats = 3)

churn_dateset_model_svm <-
  caret::train(`Churn` ~ ., data = churn_dateset_train,
               trControl = train_control, na.action = na.omit,
               method = "svmLinearWeights2", metric = "Accuracy")

### 5.b. Test the trained SVM model using the testing dataset ----
predictions_svm <- predict(churn_dateset_model_svm, churn_dateset_test[, 1:13])

### 5.c. View a summary of the model and view the confusion matrix ----
print(churn_dateset_model_svm)
caret::confusionMatrix(predictions_svm, churn_dateset_test$Churn)

## 6. Classification: Naive Bayes with Leave One Out Cross Validation ----
# In Leave One Out Cross-Validation (LOOCV), a data instance is left out and a
# model constructed on all other data instances in the training set. This is
# repeated for all data instances.

### 6.a. Train a Naive Bayes classifier based on an LOOCV ----
train_control <- trainControl(method = "LOOCV")

churn_dateset_model_nb_loocv <-
  caret::train(`Churn` ~ ., data = churn_dateset_train,
               trControl = train_control, na.action = na.omit,
               method = "naive_bayes", metric = "Accuracy")

### 6.b. Test the trained model using the testing dataset ====
predictions_nb_loocv <-
  predict(churn_dateset_model_nb_loocv, churn_dateset_test[, 1:14])

### 6.c. View the confusion matrix ====
print(churn_dateset_model_nb_loocv)
caret::confusionMatrix(predictions_nb_loocv, churn_dateset_test$Churn)

# [OPTIONAL] **Deinitialization: Create a snapshot of the R environment ----
# Lastly, as a follow-up to the initialization step, record the packages
# installed and their sources in the lockfile so that other team-members can
# use renv::restore() to re-install the same package version in their local
# machine during their initialization step.
# renv::snapshot() # nolint

# References ----
## Brown, M. (2014). Dow Jones index (Version 1) [Dataset]. University of California, Irvine (UCI) Machine Learning Repository. https://doi.org/10.24432/C5788V # nolint ----

## Ferreira, R., Martiniano, A., Ferreira, A., Ferreira, A., & Sassi, R. (2017). Daily demand forecasting orders (Version 1) [Dataset]. University of California, Irvine (UCI) Machine Learning Repository. https://doi.org/10.24432/C5BC8T # nolint ----

## Iranian churn dataset (Version 1). (2020). [Dataset]. University of California, Irvine (UCI) Machine Learning Repository. https://doi.org/10.24432/C5JW3Z # nolint ----

## National Institute of Diabetes and Digestive and Kidney Diseases. (1999). Pima Indians Diabetes Dataset [Dataset]. UCI Machine Learning Repository. https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database # nolint ----

## Yeh, I.-C. (2016). Default of credit card clients (Version 1) [Dataset]. University of California, Irvine (UCI) Machine Learning Repository. https://doi.org/10.24432/C55S3H # nolint ----

# **Required Lab Work Submission** ----
## Part A ----
# Create a new file called
# "Lab5-Submission-ResamplingMethods.R".
# Provide all the code you have used to perform all the resampling methods we
# have gone through in this lab on the. This should be done on the
# "Pima Indians Diabetes" dataset provided in the "mlbench" package.

## Part B ----
# Upload *the link* to your
# "Lab5-Submission-ResamplingMethods.R" hosted
# on Github (do not upload the .R file itself) through the submission link
# provided on eLearning.

## Part C ----
# Create a markdown file called "Lab-Submission-Markdown.Rmd"
# and place it inside the folder called "markdown". Use R Studio to ensure the
# .Rmd file is based on the "GitHub Document (Markdown)" template when it is
# being created.

# Refer to the following file in Lab 1 for an example of a .Rmd file based on
# the "GitHub Document (Markdown)" template:
#     https://github.com/course-files/BBT4206-R-Lab1of15-LoadingDatasets/blob/main/markdown/BIProject-Template.Rmd # nolint

# Include Line 1 to 14 of BIProject-Template.Rmd in your .Rmd file to make it
# displayable on GitHub when rendered into its .md version

# It should have code chunks that explain all the steps performed on the
# dataset.

## Part D ----
# Render the .Rmd (R markdown) file into its .md (markdown) version by using
# knitR in RStudio.

# You need to download and install "pandoc" to render the R markdown.
# Pandoc is a file converter that can be used to convert the following files:
#   https://pandoc.org/diagram.svgz?v=20230831075849

# Documentation:
#   https://pandoc.org/installing.html and
#   https://github.com/REditorSupport/vscode-R/wiki/R-Markdown

# By default, Rmd files are open as Markdown documents. To enable R Markdown
# features, you need to associate *.Rmd files with rmd language.
# Add an entry Item "*.Rmd" and Value "rmd" in the VS Code settings,
# "File Association" option.

# Documentation of knitR: https://www.rdocumentation.org/packages/knitr/

# Upload *the link* to "Lab-Submission-Markdown.md" (not .Rmd)
# markdown file hosted on Github (do not upload the .Rmd or .md markdown files)
# through the submission link provided on eLearning.