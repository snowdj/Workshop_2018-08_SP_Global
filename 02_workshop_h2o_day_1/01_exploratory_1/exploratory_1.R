# EXPLORATORY DATA ANALYSIS: PART 1 ----

# Objectives:
#   Build data wrangling & visualization skills
#   Learn how to size the problem: What is the cost of default?
#   Investigate Cohorts (Groups): Which contract types are more costly?

# Estimated time: 45 min


# 1.0 SETUP ----

# 1.1 Libraries ----

library(tidyverse)
library(tidyquant)



# 1.2 Data ----

application_train_raw_tbl <- read_csv("00_data/application_train.csv")

application_train_raw_tbl %>%
    glimpse()



# 2.0 DATA EXPLORATION: Sizing The Problem -----


# 2.1 How Many People Are Defaulting Vs Not? ----

data_balance_tbl <- application_train_raw_tbl %>%
    select(TARGET) %>%
    group_by(TARGET) %>%
    summarize(n = n()) %>%
    mutate(prop = n / sum(n)) %>%
    mutate(pct_text = paste0(round(prop, 3) * 100, "%"))

data_balance_tbl

data_balance_tbl %>%
    mutate(
        TARGET = as.factor(TARGET) %>%
            fct_recode("Yes" = "1", "No" = "0")
        ) %>%
    ggplot(aes(x = TARGET, y = n)) +
    geom_col(fill = palette_light()[[1]]) +
    geom_label(aes(label = pct_text)) +
    labs(title = "Data Balance", subtitle = "application_train.csv",
         y = "Number of Loan Applicants (n)", x = "Defaulted?") + 
    scale_y_continuous(labels = scales::comma) +
    theme_tq()


# 2.2 How Much Does Default Cost? -----

avg_loan_value <- 15000
avg_loan_rate  <- 0.10
recovery_rate  <- 0.40
# lgd (loss given default = 1 - recovery_rate)
# ead (exposure at default = average_loan_value * nobs)
# ps (probability of survival = 1 - probability of default) 
# positive = ps (survival), negative = pd (default)
# el (expected loss = pd * lgd * ead)
# Add recovery column (normally 40%)

default_cost_tbl <- data_balance_tbl %>%
    add_column(exposure = c(avg_loan_value * avg_loan_rate, -1 * avg_loan_value)) %>%
    mutate(recover_rate = ifelse(exposure < 0, recovery_rate, 0)) %>%
    mutate(ad = exposure * n) %>%
    mutate(al  = ead * (1-recover_rate) * prop) %>%
    mutate(al_text = scales::dollar(el)) %>%
    mutate(direction = ifelse(ead > 0, "ps (survival)", "pd (default)")) %>%
    mutate(direction = as.factor(direction) %>% fct_relevel("ps (survival)", after = 0))

default_cost_tbl

default_cost_tbl %>%
    mutate(TARGET = as.factor(TARGET)) %>%
    ggplot(aes(x = TARGET, y = el)) +
    geom_col(aes(fill = direction)) +
    geom_label(aes(label = el_text)) +
    labs(title = "Cost of Default", subtitle = "application_train.csv") + 
    scale_y_continuous(labels = scales::dollar) +
    scale_fill_tq() +
    theme_tq()


# 2.3 Which Contract Type Is More Costly? -----

application_train_raw_tbl %>%
    select_if(is.character) %>%
    glimpse()

application_train_raw_tbl %>%
    count(NAME_CONTRACT_TYPE)

average_return_tbl <- tibble(
    TARGET = c(0, 1),
    exposure = c(avg_loan_value*avg_loan_rate, -1*avg_loan_value)
)

average_return_tbl

contract_type_tbl <- application_train_raw_tbl %>%
    
    # Count by multiple groups
    select(TARGET, NAME_CONTRACT_TYPE) %>%
    group_by(TARGET, NAME_CONTRACT_TYPE) %>%
    summarize(n = n()) %>%
    ungroup() %>%
    
    # Group by contract type
    group_by(NAME_CONTRACT_TYPE) %>%
    mutate(prop = n / sum(n)) %>%
    mutate(pct_text = paste0(round(prop, 3) * 100, "%")) %>%
    arrange(NAME_CONTRACT_TYPE) %>%
    ungroup() %>%
    
    # Join average_return_tbl
    left_join(average_return_tbl) %>%
    
    # Calculate ROI
    mutate(recover_rate = ifelse(exposure < 0, recovery_rate, 0)) %>%
    mutate(ead = n * exposure * (1 - recover_rate)) %>%
    mutate(ead_text = scales::dollar(ead)) %>%
    mutate(direction = ifelse(ead > 0, "ps (survival)", "pd (default)")) %>%
    mutate(direction = as.factor(direction) %>% fct_relevel("ps (survival)", after = 0))

contract_type_tbl


contract_type_tbl %>%
    filter(TARGET == 1) %>%
    mutate(
        NAME_CONTRACT_TYPE = as.factor(NAME_CONTRACT_TYPE) %>% 
            fct_reorder(ead) %>%
            fct_rev()
        ) %>%
    
    # Plot
    ggplot(aes(x = ead, y = NAME_CONTRACT_TYPE)) +
    # Line
    geom_point(color = palette_light()[[2]], size = 6) +
    geom_segment(aes(yend = NAME_CONTRACT_TYPE), xend = 0, 
                 color = palette_light()[[2]], size = 1.5) +
    # Vline
    geom_vline(xintercept = 0, linetype = 2) +
    # Label
    geom_label(aes(label = paste0(ead_text, "\n", pct_text, " Default")), 
               color = palette_light()[[2]],
               vjust = -0.5) +
    # Aesthetics
    expand_limits(x = c(-400e6, 400e6)) +
    theme_tq() +
    scale_x_continuous(labels = scales::dollar) +
    labs(
        title = "Default Cost By Contract Type",
        x = "Exposure At Default",
        y = "Contract Type"
    )
    
    
    
