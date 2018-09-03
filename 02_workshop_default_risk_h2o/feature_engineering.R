
#Feature Engineering


library(tidyverse)


bureau_raw_tbl <- read_csv("00_data/bureau.csv")


mode <- function(x) {
    length(unique(x))
}


bureau_raw_tbl %>%
    group_by(SK_ID_CURR) %>%
    summarize_at(
        .vars = vars(DAYS_CREDIT, DAYS_CREDIT_ENDDATE),
        .funs = funs(length, mode, mean, sd)
    )


