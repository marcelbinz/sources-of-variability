rm(list = ls())

# load data ---------------------------------------------------------------

library(tidyverse)

source("R/utils.R")

pth_data <- "data/2-armed-bandits/"

n_ids_test <- 1000000 # when large, uses all

no <- 2




# Witte, Thalmann, & Schulz (2025) ----------------------------------------

sigma_epsilon_sq_witte <- 100
mn_witte <- 50


tbl_witte_t1 <- read_csv(str_c(pth_data, "final2armedBanditSession1.csv"))
tbl_witte_t2 <- read_csv(str_c(pth_data, "final2armedBanditSession2.csv"))
tbl_witte_t2$ID <- tbl_witte_t2$ID + 1000

# merge two sessions
tbl_witte_both <- rbind(tbl_witte_t1, tbl_witte_t2)
# make ids unique
tbl_lookup_id <- tibble(ID = unique(tbl_witte_both$ID))
tbl_lookup_id$ID_unique <- row_number(tbl_lookup_id$ID)
tbl_witte_both <- tbl_witte_both %>% left_join(tbl_lookup_id, by="ID") %>% select(-ID) %>% rename(ID = ID_unique)
# add ground-truth innovation variances
tbl_witte_both$sigma_xi_squared <- ifelse(tbl_witte_both$cond %in% c("FS", "FF"), 4, 0)

# run kalman filter
tbl_run <- tbl_witte_both %>% filter(ID <= n_ids_test)
l_tbl_witte <- split(tbl_run, interaction(tbl_run$block, tbl_run$ID))

l_witte_mv <- map(l_tbl_witte, kalman_learning, no = no,  sigma_epsilon_sq = sigma_epsilon_sq_witte, m0 = mn_witte)


# merge again
tbl_witte_mv <- bind_rows(l_witte_mv) %>% relocate(ID, block)
tbl_witte_mv <- tbl_witte_mv %>%
  group_by(ID, block) %>%
  mutate(trial_id = row_number()) %>%
  group_by(ID) %>%
  mutate(sequence_id = row_number()) %>%
  ungroup()



# Gershman (2018) ---------------------------------------------------------

sigma_epsilon_sq_gershman <- 10
mn_gershman <- 0

tbl_gershman_s1 <- read_csv(str_c(pth_data, "data1.csv"))
tbl_gershman_s2 <- read_csv(str_c(pth_data, "data2.csv"))

tbl_gershman_s2$subject <- tbl_gershman_s2$subject + 1000
# merge two sessions
tbl_gershman_both <- rbind(tbl_gershman_s1, tbl_gershman_s2)
# make ids unique
tbl_lookup_id <- tibble(subject = unique(tbl_gershman_both$subject))
tbl_lookup_id$subject_unique <- row_number(tbl_lookup_id$subject)
tbl_gershman_both <- tbl_gershman_both %>% left_join(tbl_lookup_id, by="subject") %>% select(-subject) %>% rename(subject = subject_unique)
# add ground-truth innovation variance
tbl_gershman_both$sigma_xi_squared <- 0

# run kalman filter
tbl_run <- tbl_gershman_both %>% filter(subject <= n_ids_test)
tbl_run <- tbl_run %>% rename(ID = subject, trial_id = trial, chosen = choice)
# change to 0-based indexing
tbl_run$chosen <- tbl_run$chosen - 1
l_tbl_gershman <- split(tbl_run, interaction(tbl_run$block, tbl_run$ID))
l_gershman_mv <- map(l_tbl_gershman, kalman_learning, no = no,  sigma_epsilon_sq = sigma_epsilon_sq_gershman, m0 = mn_gershman)

# merge again
tbl_gershman_mv <- bind_rows(l_gershman_mv) %>% relocate(ID, block)
tbl_gershman_mv <- tbl_gershman_mv %>%
  group_by(ID, block) %>%
  mutate(trial_id = row_number()) %>%
  group_by(ID) %>%
  mutate(sequence_id = row_number()) %>%
  ungroup()





# Fan et al. (2023) -------------------------------------------------------


sigma_epsilon_sq_fan <- 100
mn_fan <- 0

tbl_fan_s1 <- read_csv(str_c(pth_data, "exp1_bandit_task_scale.csv"), col_select = c(sub, block, trial, cond, C, reward))
tbl_fan_s2 <- read_csv(str_c(pth_data, "exp2_bandit_task_scale.csv"), col_select = c(sub, block, trial, cond, C, reward))

tbl_fan_s2$sub <- tbl_fan_s2$sub + 1000
# merge two sessions
tbl_fan_both <- rbind(tbl_fan_s1, tbl_fan_s2)
# make ids unique
tbl_lookup_id <- tibble(sub = unique(tbl_fan_both$sub))
tbl_lookup_id$sub_unique <- row_number(tbl_lookup_id$sub)
tbl_fan_both <- tbl_fan_both %>% left_join(tbl_lookup_id, by="sub") %>% select(-sub) %>% rename(sub = sub_unique)

# add ground-truth innovation variances
# condition 4 was stable/stable
tbl_fan_both <- tbl_fan_both %>% mutate(sigma_xi_squared = ifelse(cond == 4, 0, 4))

# run kalman filter
tbl_run <- tbl_fan_both %>% filter(sub <= n_ids_test)
tbl_run <- tbl_run %>% rename(ID = sub, trial_id = trial, chosen = C)
l_tbl_fan <- split(tbl_run, interaction(tbl_run$block, tbl_run$ID))

# note. checked parallel processing, but takes same amount of time
l_fan_mv <- map(l_tbl_fan, kalman_learning, no = no,  sigma_epsilon_sq = sigma_epsilon_sq_fan, m0 = mn_fan)


# merge again
tbl_fan_mv <- bind_rows(l_fan_mv) %>% relocate(ID, block)
tbl_fan_mv <- tbl_fan_mv %>%
  group_by(ID, block) %>%
  mutate(trial_id = row_number()) %>%
  group_by(ID) %>%
  mutate(sequence_id = row_number()) %>%
  ungroup()


n_before <- length(unique(tbl_witte_both$ID)) + length(unique(tbl_gershman_both$subject)) + length(unique(tbl_fan_both$sub))
tbl_gershman_mv$ID <- tbl_gershman_mv$ID + max(tbl_witte_mv$ID)
tbl_fan_mv$ID <- tbl_fan_mv$ID + max(tbl_gershman_mv$ID)
n_after <- length(unique(tbl_witte_mv$ID)) + length(unique(tbl_gershman_mv$ID)) + length(unique(tbl_fan_mv$ID))

if (!(assertthat::are_equal(n_before, n_after))){stop("dropped some ids, re-check\n")}


# Merge Different Data Sets -----------------------------------------------

tbl_all <- rbind(tbl_witte_mv, tbl_gershman_mv, tbl_fan_mv)

tbl_all <- tbl_all %>% 
  rename(sid = ID, left_m = m_1, right_m = m_2, left_v = v_1, right_v = v_2) %>%
  mutate(sid_unique = sid)


write_csv(tbl_all, "data/2-armed-bandits-merged.csv")
