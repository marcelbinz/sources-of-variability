rm(list = ls())

library(tidyverse)
library(grid)
library(gridExtra)


tbl_loss <- read_csv("wandb/wandb_export_2025-09-22T20_05_45.295+02_00.csv")
tbl_plot <- tbl_loss %>% select(epoch, ends_with("dev/loss_epoch"))

conds <- str_match(colnames(tbl_plot)[2:5], "=([a-zA-Z-_]*)_")[, 2]
conds <- str_replace(conds, "original", "ID-History")
conds <- str_replace(conds, "shared_hist", "Shared-History")
conds <- str_replace(conds, "shared_nohist", "Shared-NoHistory")
conds <- str_replace(conds, "id_nohist", "ID-NoHistory")


colnames(tbl_plot) <- c("Step", conds)


# Create a data frame for the horizontal line
chance_line <- data.frame(y = .5, label = "Chance Performance")

tbl_plot_long <- tbl_plot %>% 
  pivot_longer(-c(Step), names_to = "Condition") %>%
  mutate(
    prob = exp(-value),
    History = as.numeric(!str_detect(Condition, "NoHistory")),
    History = factor(History, labels = c("No History", "History"), ordered = TRUE),
    ID = as.numeric(str_detect(Condition, "ID")),
    ID = factor(ID, labels = c("Shared", "ID"), ordered = TRUE)
  )

stepsize <- 10

plt_prob <- tbl_plot_long %>% group_by(Step, Condition, History, ID) %>%
  summarize(loss_avg = mean(prob, na.rm = TRUE)) %>%
  filter(Step %% stepsize == 0) %>%
  ggplot(aes(Step, loss_avg, group = Condition)) +
  # Add horizontal line with legend label
  geom_hline(
    data = chance_line,
    aes(yintercept = y, linetype = label),
    color = "red", linewidth = 1, alpha = 0.5
  ) +
  # Plot main lines and points
  geom_line(aes(color = History), position = position_dodge(width = .75)) +
  geom_point(color = "white", size = 5, position = position_dodge(width = .75)) +
  geom_point(aes(color = History, shape = ID), size = 3, position = position_dodge(width = .75)) +
  # Customize legend
  scale_linetype_manual(name = "", values = c("Chance Performance" = "dotdash")) +
  theme_bw() +
  theme(
    text = element_text(size = 16),
    axis.title = element_text(size = 18),
    legend.position = "none"
    ) +
  labs(x = "Epoch", y = "Probability (correct)") +
  coord_cartesian(ylim = c(.5, .75))

chance_line_loss <- data.frame(y = .693, label = "Chance Performance")

plt_loss <- tbl_plot_long %>%
  filter(Step %% stepsize == 0) %>%
  ggplot(aes(Step, value, group = Condition)) +
  # Add horizontal line with legend label
  geom_hline(
    data = chance_line_loss,
    aes(yintercept = y, linetype = label),
    color = "red", linewidth = 1, alpha = 0.5
  ) +
  # Plot main lines and points
  geom_line(aes(color = History), position = position_dodge(width = .75)) +
  geom_point(color = "white", size = 5, position = position_dodge(width = .75)) +
  geom_point(aes(color = History, shape = ID), size = 3, position = position_dodge(width = .75)) +
  # Customize legend
  scale_linetype_manual(name = "", values = c("Chance Performance" = "dotdash")) +
  theme_bw() +
  theme(
    text = element_text(size = 16),
    axis.title = element_text(size = 18)
  ) +
  labs(x = "Epoch", y = "Loss (Avg.)") +
  coord_cartesian(ylim = c(.3, .7))

grid.draw(arrangeGrob(plt_prob, plt_loss, nrow = 1, widths = c(.4, .6)))
