rm(list = ls())

library(tidyverse)
library(grid)
library(gridExtra)


tbl_accuracy <- read_csv("wandb/wandb_export_2026-02-19T17_18_32.282+01_00.csv")
tbl_accuracy_long <- tbl_accuracy %>%
  select(epoch, ends_with("dev/acc_epoch")) %>%
  pivot_longer(-epoch) %>%
  mutate(
    Condition = str_match(name, "condition=([a-z_]*)_moreshuffle")[,2],
    shuffle = str_match(name, "moreshuffle=([a-z_/-]*)_tf")[,2]
  )


ggplot(tbl_accuracy_long, aes(epoch, value, group = shuffle)) +
  geom_line(aes(color = shuffle)) +
  facet_wrap(~ Condition)


# last 100 epochs
tbl_plt <- tbl_accuracy_long %>%
  filter(epoch > 150) %>%
  group_by(Condition, shuffle) %>%
  summarize(mn_acc = mean(value), sd = sd(value)/sqrt(100), .groups = "drop")


tbl_plt$Condition <- factor(
  tbl_plt$Condition, 
  labels = c("ID-NoHistory", "ID-History", "Shared-History", "Shared-NoHistory")
)

tbl_plt$Condition <- factor(
  tbl_plt$Condition,
  levels = c("ID-NoHistory", "ID-History", "Shared-NoHistory", "Shared-History")
)

tbl_plt$available <- tbl_plt$shuffle
tbl_plt$available <- factor(
  tbl_plt$available, labels = c(
    "Time & Value & Response", "Value & Response",
    "Time & Response", "Response", "Nothing"
  ))

tbl_plt <- tbl_plt %>% mutate(
  History = as.numeric(!str_detect(Condition, "NoHistory")),
  History = factor(History, labels = c("No History", "History"), ordered = TRUE),
  ID = as.numeric(str_detect(Condition, "ID")),
  ID = factor(ID, labels = c("Shared", "ID"), ordered = TRUE)
)


ggplot(tbl_plt %>% filter(shuffle == "nothing"), aes(History, mn_acc, group = ID)) +
  geom_errorbar(aes(
    History, ymin = mn_acc - 2*sd, ymax = mn_acc + 2*sd, color = ID
    ), width = .2, linewidth = .75) +
  geom_point(color = "white", size = 5) +
  geom_point(aes(color = ID), size = 3) +
  scale_color_brewer(palette = "Set2") +
  labs(y = "Test Accuracy") +
  theme_bw() +
  theme(
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12),
    legend.title = element_blank()
  )


tbl_pred_limit <- tbl_plt %>% filter(
  Condition %in% c("ID-NoHistory", "ID-History") &
    available == "Time & Value & Response"
)
tbl_pred_limit <- rbind(tbl_pred_limit, tbl_pred_limit)
tbl_pred_limit$Condition[3] <- "Shared-NoHistory"
tbl_pred_limit$Condition[4] <- "Shared-History"


ggplot(tbl_plt, aes(fct_rev(available), mn_acc, group = fct_rev(available))) +
  geom_col(aes(fill = fct_rev(available)), width = .8) +
  geom_hline(yintercept = .5, color = "red", linetype = "dashed", alpha = .5, linewidth = 1) +
  geom_hline(data = tbl_pred_limit, aes(yintercept = mn_acc), color = "darkgreen", linetype = "dashed", alpha = .7, linewidth = 1) +
  facet_wrap(~ Condition) +
  scale_fill_brewer(palette = "Set2", guide = "none") +
  scale_y_continuous(breaks = seq(.5, 1, by = .1)) +
  coord_flip(ylim = c(.45, 1)) +
  labs(y = "Test Accuracy", x = "Available Variables") +
  theme_bw() +
  theme(
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12),
    strip.background = element_rect(fill = "white", color = "grey"),
    strip.text = element_text(size = 12)
  )


tbl_nohistory <- rbind(
  tbl_plt %>% filter(Condition == "ID-NoHistory"),
  tbl_plt %>% filter(available == "Time & Value & Response" & Condition == "Shared-NoHistory")
) %>% pivot_wider(id_cols = c(Condition, shuffle, History, available), names_from = ID, values_from = mn_acc)
tbl_nohistory$Shared[tbl_nohistory$Condition == "ID-NoHistory"] <- tbl_nohistory$Shared[tbl_nohistory$Condition == "Shared-NoHistory"]
tbl_nohistory <- tbl_nohistory %>% 
  filter(
    Condition == "ID-NoHistory" &
      !(available %in% c("Response", "Nothing"))
  ) %>%
  mutate(gain = (ID/Shared - 1))

ggplot(tbl_nohistory, aes(available, gain)) +
  geom_col(aes(fill = available), width = .5) +
  geom_hline(yintercept = 0, color = "red", alpha = .5, linewidth = 1, linetype = "dotdash") +
  geom_label(aes(label = str_c(round(gain, 3)))) +
  scale_fill_brewer(palette = "Set2", guide = "none") +
  scale_y_continuous(breaks = seq(-.02, .08, by = .02)) +
  scale_x_discrete(labels = function(x) str_wrap(x, width = 8)) +
  labs(y = "Accuracy Gain", x = "Used Variables") +
  theme_bw() +
  theme(
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12),
    axis.text.x = element_text(),
    strip.background = element_rect(fill = "white", color = "grey"),
    strip.text = element_text(size = 12)
  ) +
  coord_cartesian(ylim = c(-.02, .08))
