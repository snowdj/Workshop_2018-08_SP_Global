# EXPLORATORY DATA ANALYSIS: PART 2 ----

# Objectives:
#   Effient Exploration, introduction to recipes package, advanced dplyr/ggplot2
#   Optional - Working with directory / many files
#   Preprocessing For Exploration: Goal, Strategy, Execute
#   Recipes Package - Resource
#   Efficient Exploration: Correlation Funnel
#   What are the most important features?

# Estimated time: 
#   1.5 hours (w/optional training)
#   1 hour (w/o optional training)

# 1.0 SETUP ----

# 1.1 Libraries ----

library(tidyverse)
library(tidyquant)
library(fs) # Optional      
library(recipes)


# 1.2 Directory (Optional) ----

# Optional Piece: See if Moody wants us to spend 30 min on directory / files

dir_info_tbl <- dir_info("00_data/") 


dir_info_tbl %>%
    arrange(desc(size))

dir_info_tbl %>%
    glimpse()

dir_info_tbl %>%
    select(path, size) %>%
    arrange(desc(size))

dir_info_tbl %>%
    summarise(total_size = sum(size))

dir_info_tbl %>%
    filter(str_detect(path, "application_")) %>%
    select(path, size)




# 1.3 Data (Optional) ----

# application_train_raw_tbl <- read_csv("00_data/application_train.csv")

# 1.3.1 Working with a bunch of files ----

directory_data <- dir_info_tbl %>%
    filter(str_detect(path, ".csv")) %>%
    select(path, size) %>%
    mutate(data = map(path, read_csv))

directory_data

directory_data %>%
    filter(str_detect(path, "application_train")) %>%
    select(data) %>%
    unnest()

"00_data/application_test.csv" %>% 
    str_split("/|\\.",simplify = T) %>% 
    .[,2]

directory_data <- directory_data %>%
    mutate(name = str_split(path, "/|\\.", simplify = T) %>% .[,2]) %>%
    select(name, size, data)

directory_data

# 1.3.2 Creating a data pointer -----

directory_data %>%
    filter(name == "application_train") %>%
    select(data) %>%
    unnest()

get_data <- function(data_name) {
    directory_data %>%
        filter(name == data_name) %>%
        select(data) %>%
        unnest()
}

get_data("application_train")



get_application_train <- function() {
    get_data("application_train")
}

get_bureau <- function() {
    get_data("bureau")
}

get_application_train() %>%
    glimpse()

get_bureau() %>%
    glimpse()

directory_data



# 2.0 DATA EXPLORATION -----

# Data Exploration Strategy PowerPoint / Visualization

application_train_raw_tbl <- get_application_train()

application_train_raw_tbl

rm(directory_data)



feature_description_tbl <- read_csv("00_data/HomeCredit_columns_description.csv")

feature_description_tbl

feature_description_tbl %>%
    filter(str_detect(Row, "FLAG"))

# External Source
# Relevant Features from Correlation Funnel

# 2.1 Missing Data ----

missing_tbl <- application_train_raw_tbl %>%
    summarize_all(.funs = funs(sum(is.na(.)) / length(.))) %>%
    gather() %>%
    arrange(desc(value))

missing_tbl

missing_tbl %>%
    filter(value > 0) %>%
    mutate(key = as_factor(key) %>% fct_rev()) %>%
    ggplot(aes(x = value, y = key)) +
    geom_point(color = palette_light()[[1]]) +
    expand_limits(x = c(0, 1)) +
    scale_x_continuous(labels = scales::percent) +
    labs(title = "Percentage Missing", subtitle = "application_train.csv") +
    theme_tq()



# 2.2 Categorical Data ----

application_train_raw_tbl %>%
    select_if(is.character) %>%
    map_df(~ unique(.) %>% length()) %>%
    gather() %>%
    arrange(value)

application_train_raw_tbl %>%
    select_if(is.numeric) %>%
    map_df(~ unique(.) %>% length()) %>%
    gather() %>%
    arrange(value) %>%
    View("numeric")

application_train_raw_tbl %>%
    count(AMT_REQ_CREDIT_BUREAU_HOUR)


num2factor_names <- application_train_raw_tbl %>%
    select_if(is.numeric) %>%
    map_df(~ unique(.) %>% length()) %>%
    gather() %>%
    arrange(value) %>%
    filter(value <= 6) %>%
    pull(key)

num2factor_names

string2factor_names <- application_train_raw_tbl %>%
    select_if(is.character) %>%
    names()

string2factor_names




# 3.0 PREPROCESSING ----

# 3.1 Recipes ----

# Show Resources: https://tidymodels.github.io/recipes/

rec_obj <- recipe(TARGET ~ ., data = application_train_raw_tbl) %>%
    step_num2factor(num2factor_names) %>%
    step_string2factor(string2factor_names) %>%
    step_meanimpute(all_numeric()) %>%
    step_modeimpute(all_nominal()) %>%
    step_discretize(all_numeric(), options = list(min_unique = 1)) %>%
    step_dummy(all_nominal(), one_hot = TRUE) %>%
    prep(stringsAsFactors = FALSE)

tidy(rec_obj)

tidy(rec_obj, number = 5) %>%
    filter(str_detect(terms, "AMT_CREDIT"))

app_train_bake_tbl <- bake(rec_obj, application_train_raw_tbl)

app_train_bake_tbl %>%
    glimpse()

# 3.2 Correlation Analysis ----

# dplyr, ggplot2 tutorial

app_train_cor <- app_train_bake_tbl %>%
    cor(y = .$TARGET_X1) 

min_cor_value <- 0.04

cor_tbl <- app_train_cor %>%
    as.data.frame() %>%
    rownames_to_column(var = "features") %>%
    as.tibble() %>%
    rename(value = V1) %>%
    filter(abs(value) >= min_cor_value) %>%
    arrange(features) %>%
    mutate(features = stringi::stri_replace_last_regex(features, "_", "__")) %>%
    separate(features, into = c("feature", "level"), sep = "__", remove = T) %>%
    mutate(direction = ifelse(value > 0, "positive", "negative")) %>%
    arrange(desc(abs(value))) %>%
    mutate(feature = as_factor(feature) %>% fct_rev()) %>%
    mutate(feature_value = as.numeric(feature)) %>%
    filter(feature != "TARGET") 

cor_tbl

cor_tbl %>%
    ggplot(aes(value, feature, color = direction)) +
    geom_point() +
    geom_text(aes(label = level), check_overlap = T, nudge_y = 0.3) +
    geom_vline(xintercept = 0, color = palette_light()[[1]], linetype = 2) +
    theme_tq() +
    scale_color_tq() +
    labs(
        title = "Which Features Support vs Contradict Default?",
        subtitle = "Correlation Funnel: Cohort Feature Analysis",
        x = "Correlation", y = "High Correlation Features"
    )

# 3.3 Confirm Your Suspicions: EXT_SOURCE? ----

tidy(rec_obj)

tidy(rec_obj, number = 5) %>%
    filter(str_detect(terms, "EXT_SOURCE_2"))



feature_description_tbl %>%
    filter(str_detect(Row, "EXT_SOURCE")) %>%
    glimpse()

# Credit Agencies: https://www.credit.com/credit-reports/credit-reporting-agencies/
    

# 3.4 Question Your Data: Days Birth Bin 4? ----

tidy(rec_obj, number = 5) %>%
    filter(str_detect(terms, "DAYS_BIRTH"))

feature_description_tbl %>%
    filter(str_detect(Row, "DAYS_BIRTH")) %>%
    glimpse()
