# PERFORMANCE ----

# Objectives:
#   Single Explanation
#   Multiple Explanations

# Estimated time: 30 min



# 1.0 LIBRARIES ----
library(tidyverse)
library(tidyquant)
library(h2o)
library(recipes)
library(rsample)
library(lime)

# 2.0 DATA ----
application_train_raw_tbl <- read_csv("00_data/application_train.csv")

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


# 5.0 MODEL ----

h2o.init()

h2o_01_se <- h2o.loadModel("03_machine_learning/models/StackedEnsemble_AllModels_0_AutoML_20180904_113915")
h2o_01_se


# 6.0 LIME ----

predictions_tbl <- h2o_01_se %>% 
    h2o.predict(newdata = as.h2o(test_tbl[1:10,])) %>%
    as.tibble() %>%
    bind_cols(
        test_tbl %>%
            select(TARGET, SK_ID_CURR) %>%
            slice(1:10)
    )

predictions_tbl

# 6.1 Single Explanation ----

explainer <- train_tbl %>%
    select(-TARGET) %>%
    lime(
        model           = h2o_01_se,
        bin_continuous  = TRUE,
        n_bins          = 4,
        quantile_bins   = TRUE
    )

explanation <- test_tbl %>%
    slice(1) %>%
    select(-TARGET) %>%
    lime::explain(
        explainer = explainer,
        n_labels   = 1,
        n_features = 8,
        n_permutations = 10000,
        kernel_width   = 1,
        feature_select = "forward_selection"
    )

explanation %>%
    as.tibble() %>%
    glimpse()

plot_features(explanation)



# 6.2 Multiple Explanations ----

explanations <- test_tbl %>%
    slice(1:10) %>%
    select(-TARGET) %>%
    lime::explain(
        explainer  = explainer,
        n_labels   = 1,
        n_features = 8,
        n_permutations = 5000,
        kernel_width   = 1,
        feature_select = "forward_selection"
    )

explanations %>%
    as.tibble()

plot_features(explanations, ncol = 5)

plot_explanations(explanations)
