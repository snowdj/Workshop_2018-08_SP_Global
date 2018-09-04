# PERFORMANCE ----

# Objectives:
#   Walk through Single Model Performance
#   Extend to Multiple Models For Comparison 

# Estimated time: 45 min



# 1.0 LIBRARIES ----
library(tidyverse)
library(tidyquant)
library(h2o)
library(recipes)
library(rsample)
library(fs)
library(cowplot)
library(plotly)


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


# 5.0 PERFORMANCE - SINGLE MODEL ----

h2o.init()

h2o_01_se <- h2o.loadModel("03_machine_learning/models/StackedEnsemble_BestOfFamily_0_AutoML_20180827_160455")


metrics_01_se_tbl <- h2o_01_se %>%
    h2o.performance(valid = T) %>%
    h2o.metric() %>%
    as.tibble()

metrics_01_se_tbl %>% glimpse()

# 6.1 ROC Plot

metrics_01_se_tbl %>%
    ggplot(aes(fpr, tpr)) +
    geom_point() +
    ggtitle("ROC Plot")

# 6.2 Precision Vs Recall Plot

metrics_01_se_tbl %>%
    ggplot(aes(recall, precision)) +
    geom_point() + 
    ggtitle("Precision VS Recall")


# 6.3 Gain / Lift Plot

gain_lift_01_se_tbl <- h2o_01_se %>%
    h2o.performance(valid = T) %>%
    h2o.gainsLift() %>%
    as.tibble()

gain_lift_01_se_tbl

gain_lift_01_se_tbl %>%
    ggplot(aes(cumulative_data_fraction, cumulative_lift)) +
    geom_point() + 
    geom_line() +
    ggtitle("Cumulative Lift Plot")


# 6.0 PERFORMANCE - MULTIPLE MODELS ----

load_performance_metrics <- function(path) {
    
    model_h2o <- h2o.loadModel(path)
    perf_h2o  <- h2o.performance(model_h2o, valid = T) 
    
    perf_h2o %>%
        h2o.metric() %>%
        as.tibble() %>%
        mutate(auc = h2o.auc(perf_h2o)) %>%
        select(tpr, fpr, auc, precision, recall)
    
}

safe_load_performance_metrics <- possibly(load_performance_metrics, NA)

model_metrics_tbl <- dir_info("03_machine_learning/models/") %>%
    filter(size > 500) %>%
    select(path) %>%
    mutate(metrics = map(path, safe_load_performance_metrics)) %>%
    unnest() %>%
    mutate(
        path = str_split(path, pattern = "/", simplify = T)[,3] %>% 
            as_factor() %>%
            fct_reorder(auc) %>%
            fct_rev(),
        auc  = auc %>% round(3) %>% as.character() %>% as_factor()
    )

model_metrics_tbl


g <- model_metrics_tbl %>%
    ggplot(aes(fpr, tpr, color = path, linetype = auc)) +
    geom_line(size = 1) +
    theme_tq() +
    scale_color_tq() +
    ggtitle("ROC Plot") 
    
ggplotly(g)
