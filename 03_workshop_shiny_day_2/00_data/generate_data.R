# GENERATE DATA ----

# Objectives:
#   Make predictions on data for application
#   Store data to minimize runtime computations

# Estimated time: 20 min



# 1.0 LIBRARIES ----
library(tidyverse)
library(tidyquant)
library(h2o)
library(recipes)
library(rsample)
library(lime)

# 2.0 DATA ----
application_train_raw_tbl <- read_csv("../02_workshop_h2o_day_1/00_data/application_train.csv")

application_train_raw_tbl

# 3.0 SPLIT DATA ----

# Resource: https://tidymodels.github.io/rsample/

set.seed(1234)
split_obj <- initial_split(application_train_raw_tbl, prop = 0.15)

training(split_obj)
testing(split_obj)

rm(application_train_raw_tbl)

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



saveRDS(train_tbl, file = "00_data/train_tbl.rds")
saveRDS(test_tbl[1:200,], file = "00_data/test_tbl.rds")

# 5.0 PREDICTIONS ----

h2o.init()

h2o_model <- h2o.loadModel("../02_workshop_h2o_day_1/03_machine_learning/models/StackedEnsemble_AllModels_0_AutoML_20180904_113915")


predictions_h2o <- h2o.predict(h2o_model, newdata = as.h2o(test_tbl))

predictions_tbl <- predictions_h2o %>%
    as.tibble() %>%
    bind_cols(
        test_tbl
    )

saveRDS(predictions_tbl, file = "00_data/predictions_tbl.rds")





