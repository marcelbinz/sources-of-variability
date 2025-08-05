library(tidyverse)
library(grid)
library(gridExtra)


tbl_loss <- read_csv("wandb/run-20250805_035422-ioz2d946/wandb_export_2025-08-05T09_09_32.374+02_00.csv")
tbl_plot <- tbl_loss %>% select(Step, ends_with("train/loss_step"))
colnames(tbl_plot) <- c("Step", "Shared-NoHistory", "Shared-History", "ID-NoHistory", "ID-History")

step_size <- round(max(tbl_plot$Step) / 20)
tbl_plot$Step_bin <- floor(tbl_plot$Step / step_size) + 1

# Create a data frame for the horizontal line
chance_line <- data.frame(y = .5, label = "Chance Performance")

tbl_plot_long <- tbl_plot %>% 
  select(-Step) %>%
  pivot_longer(-c(Step_bin), names_to = "Condition") %>%
  mutate(
    prob = exp(-value),
    History = as.numeric(!str_detect(Condition, "NoHistory")),
    History = factor(History, labels = c("No History", "History"), ordered = TRUE),
    ID = as.numeric(str_detect(Condition, "ID")),
    ID = factor(ID, labels = c("Shared", "ID"), ordered = TRUE)
  )

plt_prob <- tbl_plot_long %>% group_by(Step_bin, Condition, History, ID) %>%
  summarize(loss_avg = mean(prob, na.rm = TRUE)) %>%
  ggplot(aes(Step_bin, loss_avg, group = Condition)) +
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
  labs(x = "Steps (binned)", y = "Probability (correct)")

chance_line_loss <- data.frame(y = .693, label = "Chance Performance")

plt_loss <- tbl_plot_long %>% group_by(Step_bin, Condition, History, ID) %>%
  summarize(loss_avg = mean(value, na.rm = TRUE)) %>%
  ggplot(aes(Step_bin, loss_avg, group = Condition)) +
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
  labs(x = "Steps (binned)", y = "Loss (Avg.)")

grid.draw(arrangeGrob(plt_prob, plt_loss, nrow = 1, widths = c(.4, .6)))

