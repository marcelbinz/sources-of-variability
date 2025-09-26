# 2025-09-25 some insights:
# prediction from average model is better than prediction from individually fit models
# this seems odd, but is because there are many outlier parameters when fitting individually
# when throwing out outlier parameter values, the individual model performs better than the average model
# to check that, you can vary the parameter windsorize_predictions (TRUE vs. FALSE)


rm(list = ls())

library(tidyverse)
library(future)
library(furrr)

n_total_trials <- 195
n_trials_train <- 130

tbl_itc <- read_csv("data/full-data.csv") %>%
  mutate(
    value_difference_rl = right_val - left_val, 
    time_difference_rl = right_time - left_time,
    value_difference_rl_z = scale(value_difference_rl)[, 1],
    time_difference_rl_z = scale(time_difference_rl)[, 1],
    sid_all = str_c(month, sid),
    sid_all = as.numeric(as.character(factor(sid_all, labels=1:length(unique(sid_all))))),
    sid = sid_all
  )

tbl_itc <- tbl_itc %>%
  group_by(sid) %>%
  mutate(trial_id_random = sample(n_total_trials)) %>%
  ungroup()

tbl_itc_train <- tbl_itc %>%
  filter(trial_id_random <= n_trials_train)

tbl_itc_test <- tbl_itc %>%
  filter(trial_id_random > n_trials_train)

assertthat::are_equal((nrow(tbl_itc_train) + nrow(tbl_itc_test)), nrow(tbl_itc))

# note. logistic model on individual data and hierarchical model on all data do not work at all
# in frequentist stats (did not try Bayesian)

upper_and_lower_bounds <- function(par, lo, hi) {
  log(((par - lo) / (hi - lo)) / (1 - (par - lo) / (hi - lo)))
}

upper_and_lower_bounds_revert <- function(par, lo, hi) {
  lo + ((hi - lo) / (1 + exp(-par)))
}

dd_hyperbolic <- function(pars_tf, my_df) {
  pars <- upper_and_lower_bounds_revert(pars_tf, los, his)
  delta <- pars[[1]]
  tau <- pars[[2]]
  v_l <- my_df$left_val * (1 / (1 + delta * my_df$left_time))
  v_r <- my_df$right_val * (1 / (1 + delta * my_df$right_time))
  
  p_r <- 1 / (1 + exp(-(tau * (v_r - v_l))))
  lik <- dbinom(my_df$right_picked, size=nrow(my_df), prob=p_r)
  
  eps <- .Machine$double.eps#1e-100
  one_minus_eps <- 1 - .Machine$double.eps
  lik <- pmin(pmax(lik, eps), one_minus_eps)
  sum_neg_log_lik <- sum(-log(lik))
  return(sum_neg_log_lik)
}

predict_dd_hyperbolic <- function(pars, my_df) {
  delta <- pars[[1]]
  tau <- pars[[2]]
  
  v_l <- my_df$left_val * (1 / (1 + delta * my_df$left_time))
  v_r <- my_df$right_val * (1 / (1 + delta * my_df$right_time))
  
  p_r <- 1 / (1 + exp(-(tau * (v_r - v_l))))
  sample_r <- rbinom(n = nrow(my_df), size = 1, prob = p_r)
  
  l_out <- list(p_r = p_r, sample_r = sample_r)
  return(l_out)
}



los <- c(0.0001, 0.0001)
his <- c(200, 200)


startvals_tf <- upper_and_lower_bounds(c(.015, 10), los, his)
l_tbl_itc <- tbl_itc %>% split(.$sid)
l_tbl_itc_train <- tbl_itc_train %>% split(.$sid) #%>% filter(sid <= 1000) 
l_tbl_itc_test <- tbl_itc_test %>% split(.$sid) #%>% filter(sid <= 1000) 

wrap_dd_fit <- function(my_df) {
  optim(startvals_tf, dd_hyperbolic, my_df = my_df)
}


n_cpus <- availableCores() - 2
plan("multisession", workers = n_cpus)
l_results_id <- future_map(l_tbl_itc_train, wrap_dd_fit)
plan("sequential")



tbl_params_fitted <- map(l_results_id, "par") %>%
  map(function(x) upper_and_lower_bounds_revert(x, los, his)) %>%
  reduce(rbind) %>%
  as.data.frame() %>%
  as_tibble() %>%
  rename(delta = V1, tau = V2)
tbl_params_fitted$sid <- map_dbl(l_tbl_itc_train, function(x) as_vector(x[1, "sid"]))
tbl_params_fitted <- tbl_params_fitted %>%
  mutate(params = pmap(list(delta, tau), ~ list(..1, ..2))) %>%
  arrange(delta) %>%
  mutate(cumprop_delta = row_number(delta)/max(row_number(delta))) %>%
  arrange(tau) %>%
  mutate(cumprop_tau = row_number(tau)/max(row_number(tau)))


windsor_lo <- .025
windsor_hi <- .975
tbl_params_fitted_filtered <- tbl_params_fitted %>%
  filter(between(cumprop_delta, windsor_lo, windsor_hi) & between(cumprop_tau, windsor_lo, windsor_hi))


tbl_params_fitted_filtered %>% mutate(`a) log (delta)` = log(delta), `b) inv_temp` = 1/tau) %>%
  pivot_longer(-c(cumprop_delta, cumprop_tau, params, delta, tau, sid)) %>%
  ggplot(aes(value)) +
  geom_histogram(color = "white", fill = "dodgerblue", bins = 10, aes(value, after_stat(count/sum(count)*2))) +
  facet_wrap(~ name, scales = "free_x") +
  theme_bw() +
  labs(x = "Parameter Value", y = "Proportion")

append_preds <- function(x, y) {
  x$pred_prob_right_picked <- y$p_r
  x$pred_right_picked <- y$sample_r
  return(x)
}


eval_preds <- function(x, sid) {
  mat_result <- table(x[, c("right_picked", "pred_right_picked")]) / nrow(x)
  tbl_result <- as_tibble(mat_result)
  tbl_result <- tbl_result %>% rename(prop = n)
  tbl_result$sid <- sid
  return(tbl_result)
}

## predict on test data

windsorize_predictions <- TRUE

if (windsorize_predictions) {
  l_tbl_itc_test <- l_tbl_itc_test[tbl_params_fitted_filtered$sid]
  l_tbl_itc_train <- l_tbl_itc_train[tbl_params_fitted_filtered$sid]
  tbl_params_fitted <- tbl_params_fitted_filtered
}

l_predict_test <- map2(tbl_params_fitted$params, l_tbl_itc_test, predict_dd_hyperbolic)
l_predict_train <- map2(tbl_params_fitted$params, l_tbl_itc_train, predict_dd_hyperbolic)
l_tbl_itc_test <- map2(l_tbl_itc_test, l_predict_test, append_preds)
l_tbl_itc_train <- map2(l_tbl_itc_train, l_predict_train, append_preds)

tbl_preds_test <- map2(l_tbl_itc_test, 1:length(l_tbl_itc_test), eval_preds) %>% reduce(rbind)
tbl_prop_correct_test <- tbl_preds_test %>% 
  filter(right_picked == pred_right_picked) %>%
  group_by(sid) %>%
  summarize(prop_correct = sum(prop)) %>%
  ungroup()

tbl_prop_correct_test %>%
  mutate(above_chance = prop_correct > .5) %>%
  summarize(better_than_chance = mean(above_chance))

ggplot(tbl_prop_correct_test, aes(prop_correct)) + 
  geom_vline(xintercept = .5, color = "darkred", linetype = "dotdash", alpha = .5, linewidth = 1) +
  geom_histogram(color = "white", fill = "dodgerblue", bins = 50) +
  theme_bw() +
  labs(x = "Proportion Correct", y = "Nr. Participants")

mean(tbl_prop_correct_test$prop_correct)


## also predict on train data as a comparison

tbl_preds_train <- map2(l_tbl_itc_train, 1:length(l_tbl_itc_train), eval_preds) %>% reduce(rbind)
tbl_prop_correct_train <- tbl_preds_train %>% 
  filter(right_picked == pred_right_picked) %>%
  group_by(sid) %>%
  summarize(prop_correct = sum(prop)) %>%
  ungroup()

tbl_prop_correct_train %>%
  mutate(above_chance = prop_correct > .5) %>%
  summarize(better_than_chance = mean(above_chance))

mean(tbl_prop_correct_train$prop_correct)

# 
# # test for some trials (single run)
# l_results <- optim(startvals_tf, dd_hyperbolic, my_df = tbl_test)
# params_fitted <- upper_and_lower_bounds_revert(l_results$par, los, his)
# l_pred <- predict_dd_hyperbolic(params_fitted, tbl_test)
# tbl_test$pred_prob <- l_pred$p_r
# tbl_test$pred_right_picked <- l_pred$sample_r
# table(tbl_test[, c("right_picked", "pred_right_picked")]) / nrow(tbl_test)

select_delta_percentile <- function(thx) {
  tbl_params_fitted_filtered %>% 
    arrange(cumprop_delta) %>% 
    filter(cumprop_delta >= thx) %>%
    summarize(delta[1]) %>% as_vector()
}
perc_of_interest <- c(.05, .10, .25, .5, .75, .9, .95)
delta_showcase <- tibble(
  perc = perc_of_interest,
  delta = map_dbl(perc_of_interest, select_delta_percentile)
)


# compare to joint model
l_results_fixed <- optim(startvals_tf, dd_hyperbolic, my_df = tbl_itc_train)
nll_fixed <- l_results_fixed$value
nll_id <- sum(map_dbl(l_results_id, "value"))

avg_params <- upper_and_lower_bounds_revert(l_results_fixed$par, los, his)
preds_avg <- predict_dd_hyperbolic(avg_params, tbl_itc_test)
tbl_itc_test_avg <- tbl_itc_test
tbl_itc_test_avg$pred_prob_right_picked <- preds_avg$p_r
tbl_itc_test_avg$pred_right_picked <- preds_avg$sample_r

sum(tbl_itc_test_avg$pred_right_picked == tbl_itc_test_avg$right_picked) / nrow(tbl_itc_test_avg)


# 2 params overall
AIC_fixed <- 2 * nll_fixed + 2 * 2
BIC_fixed <- 2 * nll_fixed + log(nrow(tbl_itc)) * 2

# 2 params per participant
AIC_id <- 2 * nll_id + 2 * length(unique(tbl_itc$sid)) * 2
BIC_id <- 2 * nll_id + log(nrow(tbl_itc)) * length(unique(tbl_itc$sid)) * 2

tbl_ics <- tibble(
  model = rep(c("Fixed", "ID"), each = 2),
  ic = rep(c("AIC", "BIC"), 2),
  value = c(AIC_fixed, BIC_fixed, AIC_id, BIC_id)
)

ggplot(tbl_ics, aes(ic, value, group = model)) +
  geom_col(aes(fill = model), position = position_dodge(width = .45), width = .5) +
  theme_bw() +
  labs(x = "", y = "Value", title = "Model Comparison") +
  theme(
    text = element_text(size = 22),
    axis.text.y = element_text(),
    legend.position = "right",
    legend.title = element_blank()
  )

# compare discounting functions

n_days <- 100
t <- seq(0, n_days, by = 2)
delta <- delta_showcase
tbl_showcase_id <- crossing(t, delta) %>%
  mutate(f_discount = 1 / (1 + delta * t)) %>%
  pivot_longer(-c(perc, delta, t)) %>%
  mutate(`log (delta)` = log(delta), perc = factor(perc * 100))


tbl_showcase_fixed <- crossing(t, delta = upper_and_lower_bounds_revert(l_results_fixed$par, los, his)[1]) %>%
  mutate(f_discount = 1 / (1 + delta * t))


dg <- position_dodge(width = 2)
ggplot() +
  # First color scale: perc lines and points
  geom_line(data = tbl_showcase_id, aes(t, value, group = perc, color = perc), position = dg) +
  geom_point(data = tbl_showcase_id, aes(t, value), color = "white", size = 3, position = dg) +
  geom_point(data = tbl_showcase_id, aes(t, value, group = perc, color = perc), size = 1.5, position = dg) +
  scale_color_viridis_d(name = "Percentage") +
  
  # Reset color scale for fixed model
  ggnewscale::new_scale_color() +
  
  # Second color scale: fixed model line
  geom_line(data = tbl_showcase_fixed, aes(t, f_discount, color = "Fixed Model"), linewidth = 2, linetype = "dotdash") +
  scale_color_manual(name = "", values = c("Fixed Model" = "red")) +
  
  theme_bw() +
  labs(x = "Time (Days)", y = "Discount Factor") +
  theme(
    text = element_text(size = 22),
    axis.text.y = element_text(),
    legend.position = "right"
  )
