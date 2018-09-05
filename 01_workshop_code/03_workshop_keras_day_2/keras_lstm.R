# KERAS LSTM TUTORIAL ----

# Objectives:
#   Advanced Tutorial: Show what is possible with deep learning
#   Time series visualization techniques
#   Backtesting techniques for sampling time series
#   Working with Keras
#   Scaling Models from one to many
#   Predicting future values

# Estimated time: 4 hr


# 1.0 LIBRARIES ---- 

# Core Tidyverse
library(tidyverse)
library(glue)
library(forcats)

# Time Series
library(timetk)
library(tidyquant)
library(tibbletime)

# Visualization
library(cowplot)

# Preprocessing
library(recipes)

# Sampling / Accuracy
library(rsample)
library(yardstick) 

# Modeling
library(keras)


# 2.0 DATA ----
sunspot_tbl <- datasets::sunspot.month %>%
    tk_tbl() %>%
    mutate(index = as_date(index))

sunspot_tbl

sunspot_tbl %>%
    ggplot(aes(index, value)) +
    geom_point(color = palette_light()[[1]], alpha = 0.5) +
    theme_tq() +
    labs(
        title    = "Sunspots Dataset",
        subtitle = "From 1749 to 2013 (Full Data Set)"
    )

sunspot_tbl %>%
    as_tbl_time(index = index) %>%
    filter_time("start" ~ "1880") %>%
    ggplot(aes(index, value)) +
    geom_point(color = palette_light()[[1]], alpha = 0.5) +
    theme_tq() +
    labs(
        title    = "Sunspots Dataset",
        subtitle = "From 1749 to 1880"
    )

sunspot_tbl %>%
    pull(value) %>%
    acf(lag.max = 12*50)


# 3.0 BACKTESTING STRATEGY ----

periods_train <- 12 * 50
periods_test  <- 12 * 10
skip_span     <- 12 * 20

rolling_origin_resamples <- rolling_origin(
    sunspot_tbl,
    initial    = periods_train,
    assess     = periods_test,
    cumulative = FALSE,
    skip       = skip_span
)

rolling_origin_resamples

# Plotting function for a single split
plot_split <- function(split, expand_y_axis = TRUE, alpha = 1, size = 1, base_size = 14) {
    
    # Manipulate data
    train_tbl <- training(split) %>%
        add_column(key = "training") 
    
    test_tbl  <- testing(split) %>%
        add_column(key = "testing") 
    
    data_manipulated <- bind_rows(train_tbl, test_tbl) %>%
        as_tbl_time(index = index) %>%
        mutate(key = fct_relevel(key, "training", "testing"))
    
    # Collect attributes
    train_time_summary <- train_tbl %>%
        tk_index() %>%
        tk_get_timeseries_summary()
    
    test_time_summary <- test_tbl %>%
        tk_index() %>%
        tk_get_timeseries_summary()
    
    # Visualize
    g <- data_manipulated %>%
        ggplot(aes(x = index, y = value, color = key)) +
        geom_line(size = size, alpha = alpha) +
        theme_tq(base_size = base_size) +
        scale_color_tq() +
        labs(
            title    = glue("Split: {split$id}"),
            subtitle = glue("{train_time_summary$start} to {test_time_summary$end}"),
            y = "", x = ""
        ) +
        theme(legend.position = "none") 
    
    if (expand_y_axis) {
        
        sun_spots_time_summary <- sunspot_tbl %>% 
            tk_index() %>% 
            tk_get_timeseries_summary()
        
        g <- g +
            scale_x_date(limits = c(sun_spots_time_summary$start, 
                                    sun_spots_time_summary$end))
    }
    
    return(g)
}

rolling_origin_resamples$splits[[1]] %>%
    plot_split(expand_y_axis = TRUE) +
    theme(legend.position = "bottom")

# Plotting function that scales to all splits 
plot_sampling_plan <- function(sampling_tbl, expand_y_axis = TRUE, 
                               ncol = 3, alpha = 1, size = 1, base_size = 14, 
                               title = "Sampling Plan") {
    
    # Map plot_split() to sampling_tbl
    sampling_tbl_with_plots <- sampling_tbl %>%
        mutate(gg_plots = map(splits, plot_split, 
                              expand_y_axis = expand_y_axis,
                              alpha = alpha, base_size = base_size))
    
    # Make plots with cowplot
    plot_list <- sampling_tbl_with_plots$gg_plots 
    
    p_temp <- plot_list[[1]] + theme(legend.position = "bottom")
    legend <- get_legend(p_temp)
    
    p_body  <- plot_grid(plotlist = plot_list, ncol = ncol)
    
    p_title <- ggdraw() + 
        draw_label(title, size = 18, fontface = "bold", colour = palette_light()[[1]])
    
    g <- plot_grid(p_title, p_body, legend, ncol = 1, rel_heights = c(0.05, 1, 0.05))
    
    return(g)
    
}

rolling_origin_resamples %>%
    plot_sampling_plan(expand_y_axis = TRUE, ncol = 3, alpha = 1, size = 1, base_size = 10, 
                       title = "Backtesting Strategy: Rolling Origin Sampling Plan")



# 4.0 KERAS LSTM - SINGLE MODEL ----

# 4.1 Get Splits ----
split    <- rolling_origin_resamples$splits[[11]]
split_id <- rolling_origin_resamples$id[[11]]

split

# 4.2 Prepare Data ----
df_trn_tbl <- training(split)
df_tst_tbl <- testing(split)

rec_obj <- recipe(value ~ ., df_trn_tbl) %>%
    step_sqrt(value) %>%
    step_center(value) %>%
    step_scale(value) %>%
    prep()

df_trn_processed_tbl <- bake(rec_obj, newdata = df_trn_tbl)
df_tst_processed_tbl <- bake(rec_obj, newdata = df_tst_tbl)

df_processed_tbl <- bind_rows(
    df_trn_processed_tbl %>% add_column(key = "training"),
    df_tst_processed_tbl %>% add_column(key = "testing")
) %>% 
    as_tbl_time(index = index)

df_processed_tbl

# 4.3 Getting the transformation history ----
tidy(rec_obj)

scale_history <- tidy(rec_obj, number = 3) %>% pull(value)
center_history <- tidy(rec_obj, number = 2) %>% pull(value)

scale_history
center_history

# 4.4 LSTM Plan ----

# Model inputs
lag_setting  <- 120 # = nrow(df_tst)
batch_size   <- 40
train_length <- 440
tsteps       <- 1
epochs       <- 50

# 4.5 Prepare For LSTM (Matrix / Array) ----

# Training Set
lag_trn_tbl <- df_processed_tbl %>%
    mutate(value_lag = lag(value, n = lag_setting)) %>%
    filter(!is.na(value_lag)) %>%
    filter(key == "training") %>%
    tail(train_length)

x_trn_vec <- lag_trn_tbl$value_lag
x_trn_arr <- array(data = x_trn_vec, dim = c(length(x_trn_vec), 1, 1))

y_trn_vec <- lag_trn_tbl$value
y_trn_arr <- array(data = y_trn_vec, dim = c(length(y_trn_vec), 1))

# Testing Set
lag_tst_tbl <- df_processed_tbl %>%
    mutate(
        value_lag = lag(value, n = lag_setting)
    ) %>%
    filter(!is.na(value_lag)) %>%
    filter(key == "testing")

x_tst_vec <- lag_tst_tbl$value_lag
x_tst_arr <- array(data = x_tst_vec, dim = c(length(x_tst_vec), 1, 1))

y_tst_vec <- lag_tst_tbl$value
y_tst_arr <- array(data = y_tst_vec, dim = c(length(y_tst_vec), 1))

# 4.6 Building LSTM Model ----

model <- keras_model_sequential()

model %>%
    
    layer_lstm(units            = 50, 
               input_shape      = c(tsteps, 1), 
               batch_size       = batch_size,
               return_sequences = TRUE, 
               stateful         = TRUE) %>% 
    
    layer_lstm(units            = 50, 
               return_sequences = FALSE, 
               stateful         = TRUE) %>% 
    
    layer_dense(units = 1)

model %>% 
    compile(loss = 'mae', optimizer = 'adam')

model

# 4.7 Fitting LSTM Model ----

for (i in 1:epochs) {
    
    history <- model %>% 
        fit(
            x          = x_trn_arr, 
            y          = y_trn_arr, 
            batch_size = batch_size,
            epochs     = 1, 
            verbose    = 1, 
            shuffle    = FALSE
            )
    
    model %>% reset_states()
    cat("Epoch: ", i)
    
}

# TODO Add plotting function ----


# 4.8 Making Predictions ----

# Make Predictions
pred_out <- model %>% 
    predict(x_tst_arr, batch_size = batch_size) %>%
    .[,1] 

# Retransform values
pred_tbl <- tibble(
    index   = lag_tst_tbl$index,
    value   = (pred_out * scale_history + center_history)^2
) 

# Combine actual data with predictions
tbl_1 <- df_trn_tbl %>%
    add_column(key = "in_sample")

tbl_2 <- df_tst_tbl %>%
    add_column(key = "out_sample")

tbl_3 <- pred_tbl %>%
    add_column(key = "prediction")

# Create time_bind_rows() to solve dplyr issue
time_bind_rows <- function(data_1, data_2, index) {
    index_expr <- enquo(index)
    bind_rows(data_1, data_2) %>%
        as_tbl_time(index = !! index_expr)
}

ret <- list(tbl_1, tbl_2, tbl_3) %>%
    reduce(time_bind_rows, index = index) %>%
    arrange(key, index) %>%
    mutate(key = as_factor(key))

ret

# 4.9 Assessing LSTM Performance ----

calc_rmse <- function(prediction_tbl) {
    
    rmse_calculation <- function(data) {
        data %>%
            filter(key != "in_sample") %>%
            spread(key = key, value = value) %>%
            select(-index) %>%
            filter(!is.na(prediction)) %>%
            rename(
                truth    = out_sample,
                estimate = prediction
            ) %>%
            rmse(truth, estimate)
    }
    
    safe_rmse <- possibly(rmse_calculation, otherwise = NA)
    
    safe_rmse(prediction_tbl)
    
}

calc_rmse(ret)

# 4.10 Visualizing The Prediction ----

# Setup single plot function
plot_prediction <- function(data, id, alpha = 1, size = 2, base_size = 14) {
    
    rmse_val <- calc_rmse(data)
    
    g <- data %>%
        ggplot(aes(index, value, color = key)) +
        geom_point(alpha = alpha, size = size) + 
        theme_tq(base_size = base_size) +
        scale_color_tq() +
        theme(legend.position = "none") +
        labs(
            title = glue("{id}, RMSE: {round(rmse_val, digits = 1)}"),
            x = "", y = ""
        )
    
    return(g)
}

ret %>% 
    plot_prediction(id = split_id, alpha = 0.65) +
    theme(legend.position = "bottom")


# 5.0 KERAS LSTM - 11 MODELS ----

# 5.1 Keras Prediction Function ----

predict_keras_lstm <- function(split, epochs = 300, ...) {
    
    lstm_prediction <- function(split, epochs, ...) {
        
        # 4.1 Get Splits ----
        df_trn_tbl <- training(split)
        df_tst_tbl <- testing(split)
        
        # 4.2 Prepare Data ----
        rec_obj <- recipe(value ~ ., df_trn_tbl) %>%
            step_sqrt(value) %>%
            step_center(value) %>%
            step_scale(value) %>%
            prep()
        
        df_trn_processed_tbl <- bake(rec_obj, newdata = df_trn_tbl)
        df_tst_processed_tbl <- bake(rec_obj, newdata = df_tst_tbl)
        
        df_processed_tbl <- bind_rows(
            df_trn_processed_tbl %>% add_column(key = "training"),
            df_tst_processed_tbl %>% add_column(key = "testing")
        ) %>% 
            as_tbl_time(index = index)
        
        
        # 4.3 Getting the transformation history ----
        scale_history <- tidy(rec_obj, number = 3) %>% pull(value)
        center_history <- tidy(rec_obj, number = 2) %>% pull(value)
        
        # 4.4 LSTM Plan ----
        
        # Model inputs
        lag_setting  <- 120 # = nrow(df_tst)
        batch_size   <- 40
        train_length <- 440
        tsteps       <- 1
        epochs       <- epochs
        
        # 4.5 Prepare For LSTM (Matrix / Array) ----
        
        # Training Set
        lag_trn_tbl <- df_processed_tbl %>%
            mutate(value_lag = lag(value, n = lag_setting)) %>%
            filter(!is.na(value_lag)) %>%
            filter(key == "training") %>%
            tail(train_length)
        
        x_trn_vec <- lag_trn_tbl$value_lag
        x_trn_arr <- array(data = x_trn_vec, dim = c(length(x_trn_vec), 1, 1))
        
        y_trn_vec <- lag_trn_tbl$value
        y_trn_arr <- array(data = y_trn_vec, dim = c(length(y_trn_vec), 1))
        
        # Testing Set
        lag_tst_tbl <- df_processed_tbl %>%
            mutate(
                value_lag = lag(value, n = lag_setting)
            ) %>%
            filter(!is.na(value_lag)) %>%
            filter(key == "testing")
        
        x_tst_vec <- lag_tst_tbl$value_lag
        x_tst_arr <- array(data = x_tst_vec, dim = c(length(x_tst_vec), 1, 1))
        
        y_tst_vec <- lag_tst_tbl$value
        y_tst_arr <- array(data = y_tst_vec, dim = c(length(y_tst_vec), 1))
        
        # 4.6 Building LSTM Model ----
        
        model <- keras_model_sequential()
        
        model %>%
            layer_lstm(units            = 50, 
                       input_shape      = c(tsteps, 1), 
                       batch_size       = batch_size,
                       return_sequences = TRUE, 
                       stateful         = TRUE) %>% 
            layer_lstm(units            = 50, 
                       return_sequences = FALSE, 
                       stateful         = TRUE) %>% 
            layer_dense(units = 1)
        
        model %>% 
            compile(loss = 'mae', optimizer = 'adam')
        
        # 4.7 Fitting LSTM Model ----
        
        for (i in 1:epochs) {
            model %>% fit(x          = x_trn_arr, 
                          y          = y_trn_arr, 
                          batch_size = batch_size,
                          epochs     = 1, 
                          verbose    = 1, 
                          shuffle    = FALSE)
            
            model %>% reset_states()
            cat("Epoch: ", i)
            
        }
        
        # 4.8 Making Predictions ----
        
        # Make Predictions
        pred_out <- model %>% 
            predict(x_tst_arr, batch_size = batch_size) %>%
            .[,1] 
        
        # Retransform values
        pred_tbl <- tibble(
            index   = lag_tst_tbl$index,
            value   = (pred_out * scale_history + center_history)^2
        ) 
        
        # Combine actual data with predictions
        tbl_1 <- df_trn_tbl %>%
            add_column(key = "in_sample")
        
        tbl_2 <- df_tst_tbl %>%
            add_column(key = "out_sample")
        
        tbl_3 <- pred_tbl %>%
            add_column(key = "prediction")
        
        # Create time_bind_rows() to solve dplyr issue
        time_bind_rows <- function(data_1, data_2, index) {
            index_expr <- enquo(index)
            bind_rows(data_1, data_2) %>%
                as_tbl_time(index = !! index_expr)
        }
        
        ret <- list(tbl_1, tbl_2, tbl_3) %>%
            reduce(time_bind_rows, index = index) %>%
            arrange(key, index) %>%
            mutate(key = as_factor(key))
        
        return(ret)
        
    }
    
    safe_lstm <- possibly(lstm_prediction, otherwise = NA)
    
    safe_lstm(split, epochs, ...)
    
}

predict_keras_lstm(split, epochs = 10)


predict_keras_lstm(NA)

# 5.2 Map Function  11 Samples ----

start <- Sys.time()
sample_predictions_lstm_tbl <- rolling_origin_resamples %>%
    mutate(predict = map(splits, predict_keras_lstm, epochs = 50))
Sys.time() - start
# Time difference of 4.183814 mins

sample_predictions_lstm_tbl

# 5.3 Assessing Performance ----

sample_rmse_tbl <- sample_predictions_lstm_tbl %>%
    mutate(rmse = map_dbl(predict, calc_rmse)) %>%
    select(id, rmse)

sample_rmse_tbl

sample_rmse_tbl %>%
    summarize(
        mean_rmse = mean(rmse),
        sd_rmse   = sd(rmse)
    )

# 5.4 Visualizing Performance ----

# TODO - COLOR OUT OF SAMPLE ----

plot_predictions <- function(sampling_tbl, predictions_col, 
                             ncol = 3, alpha = 1, size = 2, base_size = 14,
                             title = "Backtested Predictions") {
    
    predictions_col_expr <- enquo(predictions_col)
    
    # Map plot_split() to sampling_tbl
    sampling_tbl_with_plots <- sampling_tbl %>%
        mutate(gg_plots = map2(!! predictions_col_expr, id, 
                               .f        = plot_prediction, 
                               alpha     = alpha, 
                               size      = size, 
                               base_size = base_size)) 
    
    # Make plots with cowplot
    plot_list <- sampling_tbl_with_plots$gg_plots 
    
    p_temp <- plot_list[[1]] + theme(legend.position = "bottom")
    legend <- get_legend(p_temp)
    
    p_body  <- plot_grid(plotlist = plot_list, ncol = ncol)
    
    
    
    p_title <- ggdraw() + 
        draw_label(title, size = 18, fontface = "bold", colour = palette_light()[[1]])
    
    g <- plot_grid(p_title, p_body, legend, ncol = 1, rel_heights = c(0.05, 1, 0.05))
    
    return(g)
    
}


sample_predictions_lstm_tbl %>%
    plot_predictions(predictions_col = predict, alpha = 0.5, size = 1, base_size = 10,
                     title = "Keras Stateful LSTM: Backtested Predictions")

# 6.0 Predicting The Next 10 Years ----

predict_keras_lstm_future <- function(data, epochs = 300, ...) {
    
    lstm_prediction <- function(data, epochs, ...) {
        
        # 5.1.2 Data Setup (MODIFIED)
        df <- data
        
        # 5.1.3 Preprocessing
        rec_obj <- recipe(value ~ ., df) %>%
            step_sqrt(value) %>%
            step_center(value) %>%
            step_scale(value) %>%
            prep()
        
        df_processed_tbl <- bake(rec_obj, df)
        
        center_history <- rec_obj$steps[[2]]$means["value"]
        scale_history  <- rec_obj$steps[[3]]$sds["value"]
        
        # 5.1.4 LSTM Plan
        lag_setting  <- 120 # = nrow(df_tst)
        batch_size   <- 40
        train_length <- 440
        tsteps       <- 1
        epochs       <- epochs
        
        # 5.1.5 Train Setup (MODIFIED)
        lag_train_tbl <- df_processed_tbl %>%
            mutate(value_lag = lag(value, n = lag_setting)) %>%
            filter(!is.na(value_lag)) %>%
            tail(train_length)
        
        x_train_vec <- lag_train_tbl$value_lag
        x_train_arr <- array(data = x_train_vec, dim = c(length(x_train_vec), 1, 1))
        
        y_train_vec <- lag_train_tbl$value
        y_train_arr <- array(data = y_train_vec, dim = c(length(y_train_vec), 1))
        
        x_test_vec <- y_train_vec %>% tail(lag_setting)
        x_test_arr <- array(data = x_test_vec, dim = c(length(x_test_vec), 1, 1))
        
        # 5.1.6 LSTM Model
        model <- keras_model_sequential()
        
        model %>%
            layer_lstm(units            = 50, 
                       input_shape      = c(tsteps, 1), 
                       batch_size       = batch_size,
                       return_sequences = TRUE, 
                       stateful         = TRUE) %>% 
            layer_lstm(units            = 50, 
                       return_sequences = FALSE, 
                       stateful         = TRUE) %>% 
            layer_dense(units = 1)
        
        model %>% 
            compile(loss = 'mae', optimizer = 'adam')
        
        # 5.1.7 Fitting LSTM
        for (i in 1:epochs) {
            model %>% fit(x          = x_train_arr, 
                          y          = y_train_arr, 
                          batch_size = batch_size,
                          epochs     = 1, 
                          verbose    = 1, 
                          shuffle    = FALSE)
            
            model %>% reset_states()
            cat("Epoch: ", i)
            
        }
        
        # 5.1.8 Predict and Return Tidy Data (MODIFIED)
        # Make Predictions
        pred_out <- model %>% 
            predict(x_test_arr, batch_size = batch_size) %>%
            .[,1] 
        
        # Make future index using tk_make_future_timeseries()
        idx <- data %>%
            tk_index() %>%
            tk_make_future_timeseries(n_future = lag_setting)
        
        # Retransform values
        pred_tbl <- tibble(
            index   = idx,
            value   = (pred_out * scale_history + center_history)^2
        )
        
        # Combine actual data with predictions
        tbl_1 <- df %>%
            add_column(key = "actual")
        
        tbl_3 <- pred_tbl %>%
            add_column(key = "predict")
        
        # Create time_bind_rows() to solve dplyr issue
        time_bind_rows <- function(data_1, data_2, index) {
            index_expr <- enquo(index)
            bind_rows(data_1, data_2) %>%
                as_tbl_time(index = !! index_expr)
        }
        
        ret <- list(tbl_1, tbl_3) %>%
            reduce(time_bind_rows, index = index) %>%
            arrange(key, index) %>%
            mutate(key = as_factor(key))
        
        return(ret)
        
    }
    
    safe_lstm <- possibly(lstm_prediction, otherwise = NA)
    
    safe_lstm(data, epochs, ...)
    
}

future_sun_spots_tbl <- predict_keras_lstm_future(sunspot_tbl, epochs = 300)

future_sun_spots_tbl %>%
    filter_time("1900" ~ "end") %>%
    plot_prediction(id = NULL, alpha = 0.4, size = 1.5) +
    theme(legend.position = "bottom") +
    ggtitle("Sunspots: Ten Year Forecast", subtitle = "Forecast Horizon: 2013 - 2023")


