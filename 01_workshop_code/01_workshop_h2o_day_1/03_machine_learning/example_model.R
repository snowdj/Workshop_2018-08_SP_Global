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

temp_path <- h2o.saveModel(h2o_glm, "03_machine_learning/models/")

temp_path