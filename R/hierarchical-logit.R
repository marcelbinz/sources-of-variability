rm(list = ls())

library(lme4)
library(tidyverse)
library(grid)
library(gridExtra)

tbl_itc <- read_csv("data/itc-processed.csv")

tbl_itc <- tbl_itc %>%
  group_by(sid_unique) %>%
  mutate(trial_id = row_number()) %>%
  ungroup() %>%
  mutate(
    abs_vd_z = scale(absolute_value_diff)[, 1],
    rel_vd_z = scale(relative_value_diff)[, 1],
    abs_td_z = scale(absolute_time_diff)[, 1],
    rel_td_z = scale(relative_time_diff)[, 1]
  )

tbl_itc_train <- tbl_itc %>%
  filter(trial_id <= 130)# %>% filter(sid_unique <= 200)
tbl_itc_test <- tbl_itc %>%
  filter(trial_id > 130)# %>% filter(sid_unique <= 200)



# Run or Load The Models --------------------------------------------------


is_fit <- TRUE # set to FALSE when only loading saved modeling results


if (is_fit) {
  model_full <- glmer(
    later_picked ~ 
      abs_vd_z + rel_vd_z + 
      abs_td_z + rel_td_z +
      (1 + abs_td_z || sid_unique), 
    data = tbl_itc_train,
    family = binomial(link = "logit"),
    control = glmerControl(
      optimizer = "nloptwrap",
      calc.derivs = FALSE
    )
  )
  saveRDS(model_full, file = "models/full-heuristic.rds")
  
  model_time <- glmer(
    later_picked ~ abs_td_z + (1 + abs_td_z || sid_unique), 
    data = tbl_itc_train,
    family = binomial(link = "logit"),
    control = glmerControl(
      optimizer = "nloptwrap",
      calc.derivs = FALSE
    )
  )
  saveRDS(model_time, file = "models/time-heuristic.rds")
  
  
  # whether rel or abs value is taken does not make a difference wrt prediction accuracy
  model_value <- glmer(
    later_picked ~ abs_vd_z + (1 + abs_vd_z || sid_unique), 
    data = tbl_itc_train,
    family = binomial(link = "logit"),
    control = glmerControl(
      optimizer = "nloptwrap",
      calc.derivs = FALSE
    )
  )
  saveRDS(model_value, file = "models/value-heuristic.rds")
  
} else {
  model_full  <- readRDS("models/full-heuristic.rds")
  model_time  <- readRDS("models/time-heuristic.rds")
  model_value <- readRDS("models/value-heuristic.rds")
  
}



# Generate Predictions ----------------------------------------------------


tbl_itc_test$p_hat_full <- predict(model_full, newdata = tbl_itc_test, type="response")
tbl_itc_test$y_pred_full_5 <- as.integer(tbl_itc_test$p_hat_full >= .5)
tbl_itc_test$y_pred_full_1 <- as.integer(tbl_itc_test$p_hat_full >= .1)
tbl_itc_test$y_pred_full_3 <- as.integer(tbl_itc_test$p_hat_full >= .3)
tbl_itc_test$y_pred_full_7 <- as.integer(tbl_itc_test$p_hat_full >= .7)
tbl_itc_test$y_pred_full_9 <- as.integer(tbl_itc_test$p_hat_full >= .9)


tbl_itc_test$loglik_full <- tbl_itc_test$later_picked * log(tbl_itc_test$p_hat_full) + 
  (1 - tbl_itc_test$later_picked) + log(1 - tbl_itc_test$p_hat_full)

tbl_itc_test$p_hat_time <- predict(model_time, newdata = tbl_itc_test, type="response")
tbl_itc_test$y_pred_time_5 <- as.integer(tbl_itc_test$p_hat_time >= .5)
tbl_itc_test$y_pred_time_1 <- as.integer(tbl_itc_test$p_hat_time >= .1)
tbl_itc_test$y_pred_time_3 <- as.integer(tbl_itc_test$p_hat_time >= .3)
tbl_itc_test$y_pred_time_7 <- as.integer(tbl_itc_test$p_hat_time >= .7)
tbl_itc_test$y_pred_time_9 <- as.integer(tbl_itc_test$p_hat_time >= .9)

tbl_itc_test$loglik_time <- tbl_itc_test$later_picked * log(tbl_itc_test$p_hat_time) + 
  (1 - tbl_itc_test$later_picked) + log(1 - tbl_itc_test$p_hat_time)

tbl_itc_test$p_hat_value <- predict(model_value, newdata = tbl_itc_test, type="response")
tbl_itc_test$y_pred_value_5 <- as.integer(tbl_itc_test$p_hat_value >= .5)
tbl_itc_test$y_pred_value_1 <- as.integer(tbl_itc_test$p_hat_value >= .1)
tbl_itc_test$y_pred_value_3 <- as.integer(tbl_itc_test$p_hat_value >= .3)
tbl_itc_test$y_pred_value_7 <- as.integer(tbl_itc_test$p_hat_value >= .7)
tbl_itc_test$y_pred_value_9 <- as.integer(tbl_itc_test$p_hat_value >= .9)

tbl_itc_test$loglik_value <- tbl_itc_test$later_picked * log(tbl_itc_test$p_hat_value) + 
  (1 - tbl_itc_test$later_picked) + log(1 - tbl_itc_test$p_hat_value)


tbl_itc_test <- tbl_itc_test %>% 
  mutate(
    correct_pred_full = y_pred_full_5 == later_picked,
    correct_pred_time = y_pred_time_5 == later_picked,
    correct_pred_value = y_pred_value_5 == later_picked
  )


v_testacc <- map_dbl(tbl_itc_test[, c("correct_pred_full", "correct_pred_time", "correct_pred_value")], mean)
v_testloglik <- map_dbl(tbl_itc_test[, c("loglik_full", "loglik_time", "loglik_value")], sum)

tbl_test <- tibble(
  Model = c("Full", "Time", "Value"),
  Accuracy = v_testacc,
  "Neg. LL" = -round(v_testloglik, 0)
) %>% pivot_longer(-Model, names_to = "Measure")


plt_compare <- ggplot(tbl_test, aes(Model, value, group = Measure)) +
  #geom_hline(yintercept = .5, color = "darkred", linetype = "dotdash", alpha = .5, linewidth = 1) +
  geom_col(width = .5, color = "black", aes(fill = Measure)) +
  geom_label(aes(label = str_remove(str_trim(format(round(value, 3), big.mark = ",", scientific = FALSE)), ".000$"))) +
  facet_wrap(~ Measure, scales = "free_y") +
  scale_y_continuous(expand = expansion(mult = c(0, .1)), labels = scales::label_comma()) +
  scale_x_discrete() +
  theme_bw() +
  scale_fill_brewer(palette = "Set2") +
  theme(
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12),
    strip.background = element_rect(fill = "white", color = "grey"),
    strip.text = element_text(size = 12)
  )


tbl_re <- ranef(model_time) %>%
  as_tibble()
tbl_re$term <- factor(tbl_re$term, labels = c("Intercept", "Time (Absolute)"))

plt_re <- ggplot(tbl_re, aes(condval)) +
  geom_histogram(aes(fill = term), color = "black") +
  facet_wrap(~ term) +
  scale_fill_brewer(palette = "Set2", guide = "none") +
  scale_y_continuous(expand = expansion(add = c(0, 5))) +
  scale_x_continuous() +
  labs(y = "Nr. Participants", x = "Random Effect") +
  theme_bw() +
  theme(
    axis.title = element_text(size = 16),
    axis.text = element_text(size = 15),
    strip.background = element_rect(fill = "white", color = "grey"),
    strip.text = element_text(size = 16)
  )

grid.draw(arrangeGrob(plt_compare, plt_re, nrow = 1, widths = c(.4, .6)))

pdf(file = "figures/itc-hierarchical-heuristics.pdf", 8, 4)
grid.draw(arrangeGrob(plt_compare, plt_re, nrow = 1, widths = c(.3, .7)))
dev.off()

cor(tbl_itc[, c("absolute_value_diff", "absolute_time_diff")])
pl_raw <- ggplot(
  tbl_itc %>% 
    slice_sample(n = 100000) %>% 
    group_by(absolute_value_diff, absolute_time_diff) %>%
    summarize(n = n(), .groups = "drop"),
  aes(absolute_time_diff, absolute_value_diff)) +
  geom_point(aes(size = n)) +
  geom_smooth(method = "lm") +
  scale_y_continuous() +
  scale_x_continuous() +
  labs(y = "Abs. Value Difference", x = "Abs. Time Difference") +
  theme_bw() +
  theme(
    axis.title = element_text(size = 16),
    axis.text = element_text(size = 14),
    strip.background = element_rect(fill = "white", color = "grey"),
    strip.text = element_text(size = 16),
    legend.title = element_blank()
  )

pl_log <- ggplot(
  tbl_itc %>% 
    slice_sample(n = 100000) %>% 
    group_by(absolute_value_diff, absolute_time_diff) %>%
    summarize(n = n(), .groups = "drop"),
  aes(log(absolute_time_diff), log(absolute_value_diff))) +
  geom_point(aes(size = n)) +
  geom_smooth(method = "lm") +
  scale_y_continuous() +
  scale_x_continuous() +
  labs(y = "Abs. Value Difference", x = "Abs. Time Difference") +
  theme_bw() +
  theme(
    axis.title = element_text(size = 16),
    axis.text = element_text(size = 14),
    strip.background = element_rect(fill = "white", color = "grey"),
    strip.text = element_text(size = 16),
    legend.title = element_blank()
  )


grid.draw(arrangeGrob(pl_raw, pl_log, nrow = 1))





# Precision - Recall Analysis ---------------------------------------------

tbl_recall <- tbl_itc_test %>% filter(later_picked == 0) %>%
  summarize(
    recall_full_1 = mean(y_pred_full_1 == later_picked),
    recall_full_3 = mean(y_pred_full_3 == later_picked),
    recall_full_5 = mean(y_pred_full_5 == later_picked),
    recall_full_7 = mean(y_pred_full_7 == later_picked),
    recall_full_9 = mean(y_pred_full_9 == later_picked),
    
    recall_time_1 = mean(y_pred_time_1 == later_picked),
    recall_time_3 = mean(y_pred_time_3 == later_picked),
    recall_time_5 = mean(y_pred_time_5 == later_picked),
    recall_time_7 = mean(y_pred_time_7 == later_picked),
    recall_time_9 = mean(y_pred_time_9 == later_picked),
    
    recall_value_1 = mean(y_pred_value_1 == later_picked),
    recall_value_3 = mean(y_pred_value_3 == later_picked),
    recall_value_5 = mean(y_pred_value_5 == later_picked),
    recall_value_7 = mean(y_pred_value_7 == later_picked),
    recall_value_9 = mean(y_pred_value_9 == later_picked),
  ) %>% mutate(idx = 1) %>% pivot_longer(-idx, values_to = "Recall") %>%
  mutate(
    model = str_remove_all(str_extract(name, "_[a-z]*_"), "_"),
    centile = str_extract(name, "[1-9]")
  )


tbl_fp <- tbl_itc_test %>% filter(later_picked == 1) %>%
  summarize(
    fpr_full_1 = mean(y_pred_full_1 != later_picked),
    fpr_full_3 = mean(y_pred_full_3 != later_picked),
    fpr_full_5 = mean(y_pred_full_5 != later_picked),
    fpr_full_7 = mean(y_pred_full_7 != later_picked),
    fpr_full_9 = mean(y_pred_full_9 != later_picked),
    
    fpr_time_1 = mean(y_pred_time_1 != later_picked),
    fpr_time_3 = mean(y_pred_time_3 != later_picked),
    fpr_time_5 = mean(y_pred_time_5 != later_picked),
    fpr_time_7 = mean(y_pred_time_7 != later_picked),
    fpr_time_9 = mean(y_pred_time_9 != later_picked),
    
    fpr_full_1 = mean(y_pred_value_1 != later_picked),
    fpr_value_3 = mean(y_pred_value_3 != later_picked),
    fpr_value_5 = mean(y_pred_value_5 != later_picked),
    fpr_value_7 = mean(y_pred_value_7 != later_picked),
    fpr_value_9 = mean(y_pred_value_9 != later_picked),
  ) %>% mutate(idx = 1) %>% pivot_longer(-idx, values_to = "FPR") %>%
  mutate(
    model = str_remove_all(str_extract(name, "_[a-z]*_"), "_"),
    centile = str_extract(name, "[1-9]")
  )

precision <- function(ycol) {
  ycol <- sym(ycol)
  tbl_itc_test %>%
    filter({{ycol}} == 0) %>%
    summarize(
      precision = mean({{ycol}} == later_picked)
    ) %>% as_vector()
}

tbl_prep <- crossing(
  prefix = c("y_pred_full_", "y_pred_time_", "y_pred_value_"), 
  centile = seq(1, 9, by = 2)
) %>% mutate(
  name = str_c(prefix, centile)
)


v_precision <- map_dbl(tbl_prep$name, precision)

tbl_precision <- crossing(
  prefix = c("precision_full_", "precision_time_", "precision_value_"), 
  centile = seq(1, 9, by = 2)
) %>% mutate(
  name = str_c(prefix, centile)
) %>% select(-c(prefix)) %>%
  mutate(
    Precision = v_precision,
    model = str_remove_all(str_extract(name, "_[a-z]*_"), "_"),
    centile = as.character(centile)
  )


tbl_metrics <- tbl_recall %>% 
  left_join(tbl_precision, by = c("model", "centile")) %>%
  left_join(tbl_fp, by = c("model", "centile")) %>%
  mutate(
    model = factor(model, labels = c("Full", "Time", "Value"))
  )


plt_roc <- ggplot(tbl_metrics, aes(FPR, Recall, group = model)) +
  geom_abline() +
  geom_line(aes(color = model)) +
  geom_point(color = "white", size = 3) +
  geom_point(aes(color = model), size = 2) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  scale_color_brewer(palette = "Set2", guide = "none") +
  scale_y_continuous(expand = expansion(add = c(0, 0))) +
  scale_x_continuous(expand = expansion(add = c(.01, 0))) +
  labs(y = "TPR", x = "FPR", title = "ROC") +
  theme_bw() +
  theme(
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12),
    strip.background = element_rect(fill = "white", color = "grey"),
    strip.text = element_text(size = 12),
    legend.title = element_blank()
  )


plt_pr <- ggplot(tbl_metrics, aes(Recall, Precision, group = model)) +
  geom_line(aes(color = model)) +
  geom_point(color = "white", size = 3) +
  geom_point(aes(color = model), size = 2) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  scale_color_brewer(palette = "Set2") +
  scale_y_continuous(expand = expansion(add = c(0, 0))) +
  scale_x_continuous(expand = expansion(add = c(0, 0))) +
  labs(y = "Precision", x = "TPR/Recall", title = "Precision-Recall") +
  theme_bw() +
  theme(
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12),
    strip.background = element_rect(fill = "white", color = "grey"),
    strip.text = element_text(size = 12),
    legend.text = element_text(size = 12),
    legend.title = element_blank()
  )


pdf(file = "figures/itc-heuristics-eval.pdf", 7, 3.5)
grid.draw(arrangeGrob(plt_roc, plt_pr, nrow = 1, widths = c(.425, .575)))
dev.off()


# Explore Diminishing Effect of Time --------------------------------------

tbl_itc$earlier_time <- apply(tbl_itc[, c("right_time", "left_time")], 1, min)
tbl_itc$earlier_time_bin <- cut(tbl_itc$earlier_time, c(0, exp(seq(0, 4, by = .5)), 1000), labels = FALSE)
tbl_itc$absolute_value_diff_cut <- cut(tbl_itc$absolute_value_diff, c(.2, .6, 1.1, 2.1, 3, 4, 5, 10, 101, 501, 1001), labels = FALSE)

# problem is that for even tiny value differences of .1$ people tend to prefer the later option when later is 1 day
tbl_itc %>% group_by(absolute_value_diff_cut, earlier_time_bin) %>%
  summarize(prop_later_picked = mean(later_picked), .groups = "drop") %>%
  ggplot(aes(earlier_time_bin, prop_later_picked, group = absolute_value_diff_cut)) +
  geom_line(aes(color = absolute_value_diff_cut))

