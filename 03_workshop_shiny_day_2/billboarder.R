# BILLBOARDER VIZUALIZATION ----

# Objective:
#   Show how to develop a custom javascript LIME chart using billboarder

# 1.0 Libraries ----
library(shiny)
library(flexdashboard)
library(billboarder)
library(tidyverse)
library(lime)
library(h2o)

# 2.0 Data ----
predictions_tbl <- readRDS("00_data/predictions_tbl.rds")
train_tbl <- readRDS("00_data/train_tbl.rds")
test_tbl <- readRDS("00_data/test_tbl.rds")

# 3.0 H2O Model ----
h2o.init()
h2o_model <- h2o.loadModel("01_model/StackedEnsemble_BestOfFamily_0_AutoML_20180827_160455")


# 4.0 Visualization ----

selected_applicant_id <- predictions_tbl$SK_ID_CURR[[2]]
# selected_applicant_id <- input$applicant_id

# LIME
explainer <- train_tbl %>%
    select(-TARGET) %>%
    lime(
        model           = h2o_model,
        bin_continuous  = TRUE,
        n_bins          = 4,
        quantile_bins   = TRUE
    )

explanation <- test_tbl %>%
    filter(SK_ID_CURR == selected_applicant_id) %>%
    select(-TARGET) %>%
    lime::explain(
        explainer = explainer,
        n_labels   = 1,
        n_features = 8,
        n_permutations = 5000,
        kernel_width   = 1,
        feature_select = "forward_selection"
    )

plot_features(explanation)

# Labels
label      <- unique(explanation$label)
label_prob <- unique(explanation$label_prob)
model_r2   <- unique(explanation$model_r2)

tibble(
    `Class Prediction` = toupper(label),
    `Class Probability` = round(label_prob, 2),
    `Model R2` = round(model_r2, 2)
) %>%
    gather(key = "Metric", value = "Value")

# Data Transformation
explanation_transformed_tbl <- explanation %>%
    as.tibble() %>%
    select(feature:feature_desc) %>%
    mutate(direction = ifelse(feature_weight > 0, "Supports", "Contradicts")) %>%
    mutate(direction = factor(direction, levels = c("Supports", "Contradicts"))) %>%
    mutate(feature_desc = as_factor(feature_desc) %>% fct_reorder(abs(feature_weight))) %>%
    arrange(desc(feature_desc))

explanation_transformed_tbl

# ggplot viz
explanation_transformed_tbl %>%
    ggplot(aes(feature_desc, feature_weight, fill = direction)) +
    geom_col() +
    coord_flip()

# billboarder
# Resources: https://cran.r-project.org/web/packages/billboarder/index.html
#   Intro: https://cran.r-project.org/web/packages/billboarder/vignettes/billboarder-intro.html
#   Mapping: https://cran.r-project.org/web/packages/billboarder/vignettes/billboarder-mapping.html
#   Options:https://cran.r-project.org/web/packages/billboarder/vignettes/billboarder-options.html

billboarder(data = explanation_transformed_tbl) %>%
    bb_aes(x = feature_desc, y = feature_weight, group = direction) %>%
    bb_barchart(rotated = TRUE, stacked = TRUE) %>%
    bb_y_grid(show = TRUE) %>%
    bb_colors_manual(
        Supports    = "#2c3e50",
        Contradicts = "#e31a1c"
    ) %>%
    bb_tooltip(format = list(
        value = htmlwidgets::JS("function(x) {return parseFloat(x).toFixed(3)}")
    )) 

tidyquant::palette_light()   

