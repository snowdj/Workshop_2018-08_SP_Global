# MACHINE LEARNING ----

# Objectives:
#   Data preparation
#   Show simple base R methods: Logistic Regression
#   Learn performance methods: H2O GLM, GBM, RF
#   Learn Automated ML

# Estimated time: 2 hours



# 1.0 LIBRARIES ----
library(tidyverse)
library(tidyquant)
library(h2o)
library(recipes)
library(rsample)
library(yardstick)


# 2.0 DATA ----
application_train_raw_tbl <- read_csv("00_data/application_train.csv")

application_train_raw_tbl


# 3.0 SPLIT DATA ----

# Resource: https://tidymodels.github.io/rsample/

set.seed(1234)
split_obj <- initial_split(application_train_raw_tbl, prop = 0.15)

training(split_obj) # 15% of Data
testing(split_obj)  # 85% of Data


# 4.0 PREPROCESSING ----

# 4.1 Handle Categorical ----

num2factor_names <- training(split_obj) %>%
    select_if(is.numeric) %>%
    map_df(~ unique(.) %>% length()) %>%
    gather() %>%
    arrange(value) %>%
    filter(value <= 6) %>%
    pull(key)

num2factor_names

string2factor_names <- training(split_obj) %>%
    select_if(is.character) %>%
    names()

string2factor_names


# 4.2 Recipes ----

# Resource: https://tidymodels.github.io/recipes/

rec_obj <- recipe(TARGET ~ ., data = training(split_obj)) %>%
    step_num2factor(num2factor_names) %>%
    step_string2factor(string2factor_names) %>%
    step_meanimpute(all_numeric()) %>%
    step_modeimpute(all_nominal()) %>%
    prep(stringsAsFactors = FALSE)


train_tbl <- bake(rec_obj, training(split_obj))
test_tbl  <- bake(rec_obj, testing(split_obj))



# 5.0 MODELING -----

# 5.1 Logistic Regression ----

model_glm <- glm(TARGET ~ ., data = train_tbl, family = "binomial") # error

train_tbl %>% 
    select_if(is.factor) %>%
    map_df(~ levels(.) %>% length()) %>%
    gather() %>%
    arrange(value)

model_glm <- glm(TARGET ~ ., 
                 data = train_tbl %>% select(-c(FLAG_MOBIL, FLAG_DOCUMENT_12)), 
                 family = "binomial")

predict_glm <- predict(model_glm, 
                       newdata = test_tbl %>% select(-c(FLAG_MOBIL, FLAG_DOCUMENT_12)),
                       type = "response")

tibble(actual  = test_tbl$TARGET,
       predict = predict_glm) %>%
    mutate(predict_class = ifelse(predict > 0.5, 1, 0) %>% as.factor()) %>%
    roc_auc(actual, predict)
# [1] 0.7396841

rm(model_glm)



# 5.2 H2O Models ----

h2o.init()

train_h2o <- as.h2o(train_tbl)
test_h2o  <- as.h2o(test_tbl)

y <- "TARGET"
x <- setdiff(names(train_h2o), y)

# 5.2.2 GLM (Elastic Net) ----

h2o_glm <- h2o.glm(
    x = x,
    y = y,
    training_frame = train_h2o,
    validation_frame = test_h2o,
    nfolds = 5,
    seed = 1234,
    
    # GLM
    family = "binomial"
    
)

h2o.performance(h2o_glm, valid = TRUE) %>%
    h2o.auc()
# [1] 0.7416916

h2o.saveModel(h2o_glm, "03_machine_learning/models/")

h2o.loadModel("03_machine_learning/models/GLM_model_R_1536254907503_1")

h2o_glm@allparameters

# 5.2.3 GBM ----

# Resource: https://blog.h2o.ai/2016/06/h2o-gbm-tuning-tutorial-for-r/

h2o_gbm <- h2o.gbm(
    x = x,
    y = y,
    training_frame = train_h2o,
    validation_frame = test_h2o,
    nfolds = 5,
    seed = 1234,
    
    # GBM
    ntrees = 50,
    max_depth = 5,
    learn_rate = 0.1
)

h2o.performance(h2o_gbm, valid = TRUE) %>%
    h2o.auc()
# [1] 0.7411573

h2o_gbm@allparameters

# 5.2.4 Random Forest ----

h2o_rf <- h2o.randomForest(
    x = x,
    y = y,
    training_frame = train_h2o,
    validation_frame = test_h2o,
    nfolds = 5,
    seed = 1234,
    
    # RF
    ntrees = 50,
    max_depth = 5,
    balance_classes = TRUE
    
)

h2o.performance(h2o_rf, valid = TRUE) %>%
    h2o.auc()
# [1] 0.728683

h2o_rf@allparameters



# 5.3 H2O AutoML ----

# 5.3.1 h2o.automl() ----

# H2O Docs: http://docs.h2o.ai

start <- Sys.time()
automl_results <- h2o.automl(
    x = x,
    y = y,
    training_frame   = train_h2o,
    validation_frame = test_h2o,
    max_runtime_secs = 900
)
Sys.time() - start

# Time difference of 16.5572 mins

# BREAK -----

automl_results@leaderboard %>%
    as.tibble()

# 5.3.2 Getting Models ----

h2o_01_se  <- h2o.getModel("StackedEnsemble_BestOfFamily_0_AutoML_20180905_071103")
h2o_03_glm <- h2o.getModel("GLM_grid_0_AutoML_20180905_071103_model_0")
h2o_04_gbm <- h2o.getModel("GBM_grid_0_AutoML_20180905_071103_model_13")


# 5.3.3 Saving & Loading ----

h2o.saveModel(h2o_04_gbm, "00_models")

h2o.loadModel("03_machine_learning/models/StackedEnsemble_AllModels_0_AutoML_20180904_113915")

# 5.3.4 Performance Metrics -----

performance_h2o <- h2o.performance(h2o_02_se, valid = TRUE)

performance_h2o %>%
    h2o.auc()
# [1] 0.7476822

# 5.3.5 Making Predictions -----

prediction_h2o <- h2o.predict(h2o_01_se, newdata = test_h2o)

prediction_h2o %>%
    as.tibble()



