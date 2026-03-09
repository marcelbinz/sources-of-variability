
# Load Packages -----------------------------------------------------------



rm(list = ls())

library(tidyverse)
library(grid)
library(gridExtra)

if (!dir.exists("figures")) {dir.create("figures")}



# Plotting Settings -------------------------------------------------------

## define task settings and which data to select
## note. data_select only required for risky data set, not itc


task <- c("itc", "risky")[2]
if (task == "risky"){
  data_select = 2 # or 2: 1 includes problem repetitions, 2 excludes them
}


task_settings <- switch(
  task,
  "itc" = list(
    pth_conditions = "wandb/wandb_export_itc_conditions.csv",
    pth_masklength = "wandb/wandb_export_itc_masklength.csv",
    pth_variables = "wandb/wandb_export_itc_variables.csv",
    epochthxs = c(201, 250),
    n_epochs = 50,
    indep_vars_labels = c(
      "T&V&R", "V&R",
      "T&R", "R", "Nothing"
    ),
    pl_dir = "figures/itc-conditions.pdf",
    pl_dir_masklength = "figures/itc-masklength.pdf",
    pl_dir_gains = "figures/itc-gains.pdf",
    pl_dir_joint = "figures/itc-jointplot.pdf",
    pl_conditions_labelpos = c(.5, .3)
  ),
  "risky" = list(
    pth_conditions = c(
      "wandb/wandb_export_risky_conditions_peterson-select.csv",
      "wandb/wandb_export_risky_conditions_only-first-problem.csv"
    )[data_select],
    # only ran on peterson select, because no effects (/not enough signal) for first problem
    pth_masklength = "wandb/wandb_export_risky_masklength.csv",
    pth_variables = c(
      "wandb/wandb_export_risky_variables_peterson-select.csv",
      "wandb/wandb_export_risky_variables_only-first-problem.csv"
    )[data_select],
    epochthxs = c(201, 250), 
    n_epochs = 50,
    indep_vars_labels = c(
      "P&V&R", "V&R",
      "P&R", "R", "Nothing"
    ),
    pl_dir = c(
      "figures/risky-peterson-select-conditions.pdf",
      "figures/risky-first-problem-conditions.pdf"
    )[data_select],
    pl_dir_masklength = "figures/risky-peterson-select-masklength.pdf",
    pl_dir_gains = c(
      "figures/risky-peterson-select-gains.pdf",
      "figures/risky-first-problem-gains.pdf"
    )[data_select],
    pl_dir_joint = c(
      "figures/risky-peterson-select-jointplot.pdf",
      "figures/risky-first-problem-jointplot.pdf"
    )[data_select],
    pl_conditions_labelpos = c(.5, .8)
  )
)





# Four Conditions ---------------------------------------------------------


tbl_accuracy <- read_csv(task_settings$pth_conditions)
tbl_accuracy_long <- tbl_accuracy %>%
  select(epoch, ends_with("dev/acc_epoch")) %>%
  pivot_longer(-epoch) %>%
  mutate(
    Condition = str_match(name, "condition=([a-z_]*)_[md]")[,2],
    shuffle = str_match(name, "moreshuffle=([a-z_/-]*)_tf")[,2]
  )


# select epochs to be analyzed
tbl_conditions <- tbl_accuracy_long %>%
  filter(between(epoch, task_settings$epochthxs[1], task_settings$epochthxs[2])) %>%
  group_by(Condition, shuffle) %>%
  summarize(mn_acc = mean(value), sd = sd(value)/sqrt(task_settings$n_epochs), .groups = "drop")

tbl_conditions$Condition <- factor(
  tbl_conditions$Condition,
  levels = c("shared_nohist", "shared_hist", "id_nohist", "original"),
  labels = c("Shared-NoHistory", "Shared-History", "ID-NoHistory", "ID-History")
)

# plot and save

tbl_conditions <- tbl_conditions %>% mutate(
  History = as.numeric(!str_detect(Condition, "NoHistory")),
  History = factor(History, labels = c("No History", "History"), ordered = TRUE),
  ID = as.numeric(str_detect(Condition, "ID")),
  ID = factor(ID, labels = c("Shared", "ID"), ordered = TRUE)
)


pl_conditions <- ggplot(tbl_conditions, aes(History, mn_acc, group = ID)) +
  geom_hline(yintercept = .5, linetype = "dotdash", linewidth = 1, color = "red", alpha = .3) +
  geom_errorbar(aes(
    History, ymin = mn_acc - 2*sd, ymax = mn_acc + 2*sd, color = ID
  ), width = .2, linewidth = .75) +
  geom_line((aes(color = ID))) +
  geom_point(color = "white", size = 5) +
  geom_point(aes(color = ID), size = 3) +
  scale_color_brewer(palette = "Set1") +
  scale_y_continuous(expand = c(0.01, 0)) +
  labs(y = "Test Accuracy") +
  theme_bw() +
  theme(
    axis.title = element_text(size = 20),
    axis.text = element_text(size = 18),
    legend.text = element_text(size = 16),
    legend.title = element_blank(),
    axis.title.x = element_blank(),
    legend.position = "inside",
    legend.position.inside = task_settings$pl_conditions_labelpos
  ) +
  coord_cartesian(ylim = c(.5, 1))



pdf(file = task_settings$pl_dir, 5, 4)
grid.draw(pl_conditions)
dev.off()




# Masklength --------------------------------------------------------------


tbl_constrain <- read_csv(task_settings$pth_masklength)

tbl_constrain <- tbl_constrain %>% select(
  c("epoch",
    (contains("original") | contains("id_nohist")) &
      contains("dev/acc_epoch") & !contains("MIN") & !contains("MAX")
  ))

masklenghts <- str_match(colnames(tbl_constrain), "windowsize=([0-9]+) | (epoch)")[, 2]
masklenghts <- masklenghts[!is.na(masklenghts)]
# last two are all and none
masklenghts <- masklenghts[0:(length(masklenghts)-2)]

#Window=
colnames(tbl_constrain) <- c("Epoch", masklenghts, "All Prev", "No Hist")

tbl_constrain_long <- tbl_constrain %>% pivot_longer(-"Epoch")  %>%
  filter(between(Epoch, task_settings$epochthxs[1], task_settings$epochthxs[2])) 
tbl_constrain_long$name <- factor(
  tbl_constrain_long$name,
  levels = c(sort(as.numeric(masklenghts)), "All Prev", "No Hist")
)

tbl_plt_masklength <- tbl_constrain_long %>%
  group_by(name) %>%
  summarize(accuracy = mean(value, na.rm = TRUE), .groups = "drop")

pl_masklength <- ggplot(tbl_plt_masklength, aes(name, accuracy)) +
  geom_hline(yintercept = .5, linetype = "dotdash", linewidth = 1, color = "red", alpha = .3) +
  geom_line(color = "#1F78B4", linewidth = 1, aes(group = 1)) +
  geom_col(aes(fill = name), width = .5, alpha = .5) +
  geom_point(color = "white", size = 4) +
  geom_point(aes(color = name)) +
  geom_label(aes(y = accuracy - .05, label = round(accuracy, 3))) +
  theme_bw() +
  scale_x_discrete(labels = function(x) str_wrap(x, width = 30)) +
  scale_y_continuous(expand = c(0.01, 0)) +
  labs(x = "Size of Window", y = "Test Accuracy") +
  theme(
    axis.title = element_text(size = 20),
    axis.text = element_text(size = 18),
    title = element_text(size = 15),
    strip.background = element_rect(fill = "white"), 
    axis.title.x = element_blank()
  ) +
  scale_color_manual(values = c(
    '#E0F3F8', '#C2E0EE', '#A6CEE3', '#6AA6CC', '#1F78B4',
    '#083F82', '#041C3A', "#F28E2B"), guide = "none") +
  scale_fill_manual(values = c(
    '#E0F3F8', '#C2E0EE', '#A6CEE3', '#6AA6CC', '#1F78B4',
    '#083F82', '#041C3A', "#F28E2B"), guide = "none") +
  coord_cartesian(ylim = c(.5, 1))

pdf(file = task_settings$pl_dir_masklength, 7.75, 4)
grid.draw(pl_masklength)
dev.off()


# Effect Decomposition ----------------------------------------------------



tbl_accuracy <- read_csv(task_settings$pth_variables)
tbl_accuracy_long <- tbl_accuracy %>%
  select(epoch, ends_with("dev/acc_epoch")) %>%
  pivot_longer(-epoch) %>%
  mutate(
    Condition = str_match(name, "condition=([a-z_]*)_[md]")[,2],
    shuffle = str_match(name, "more_shuffle=([a-z_/-]*)_tf")[,2]
  )

tbl_accuracy_long$Condition <- factor(
  tbl_accuracy_long$Condition,
  levels = c("shared_nohist", "shared_hist", "id_nohist", "original"),
  labels = c("Shared-NoHistory", "Shared-History", "ID-NoHistory", "ID-History")
)

tbl_accuracy_long$available <- tbl_accuracy_long$shuffle
tbl_accuracy_long$available[tbl_accuracy_long$available == "time-val"] <- "val-time"
tbl_accuracy_long$available[tbl_accuracy_long$available == "prob-val-picked_prev"] <- "val-prob-picked_prev"
tbl_accuracy_long$available[tbl_accuracy_long$available == "prob-val"] <- "val-prob"


tbl_accuracy_long$available <- factor(
  tbl_accuracy_long$available, labels = task_settings$indep_vars_labels
)
# plot and save

tbl_accuracy_long <- tbl_accuracy_long %>% mutate(
  History = as.numeric(!str_detect(Condition, "NoHistory")),
  History = factor(History, labels = c("No History", "History"), ordered = TRUE),
  ID = as.numeric(str_detect(Condition, "ID")),
  ID = factor(ID, labels = c("Shared", "ID"), ordered = TRUE)
)

# select epochs to be analyzed
tbl_variables <- tbl_accuracy_long %>%
  filter(between(epoch, task_settings$epochthxs[1], task_settings$epochthxs[2])) %>%
  group_by(Condition, History, ID, available, shuffle) %>%
  summarize(mn_acc = mean(value), sd = sd(value)/sqrt(task_settings$n_epochs), .groups = "drop")



subselect_conditions <- function(tbl_plt, h, i) {
  tbl_plt %>% filter(History == h[1] & ID == i[1]) %>%
    select(-c(Condition, shuffle, History)) %>%
    left_join(
      tbl_plt %>% filter(History == h[2] & ID == i[2]) %>%
        select(-c(Condition, shuffle, History)),
      by = c("available"), suffix = c("_ID", "_Shared")
    ) %>%
    mutate(
      delta_mn = mn_acc_ID - mn_acc_Shared
    ) %>% relocate("available", .before = "mn_acc_ID")
}

tbl_plt_gain_id <- subselect_conditions(tbl_variables, rep("No History", 2), c("ID", "Shared"))
tbl_plt_gain_hist <- subselect_conditions(tbl_variables, c("History", "No History"), rep("Shared", 2))
tbl_plt_gain_both <- subselect_conditions(tbl_variables, c("History", "No History"), c("ID", "Shared"))

tbl_plt_gain_base <- subselect_conditions(tbl_variables, rep("No History", 2), c("Shared", "Shared"))
tbl_plt_gain_base$mn_acc_all <- tbl_plt_gain_base$mn_acc_ID[
  tbl_plt_gain_base$available == "T&V&R" |
    tbl_plt_gain_base$available == "P&V&R"
  ]
tbl_plt_gain_base$mn_acc_nothing <- tbl_plt_gain_base$mn_acc_ID[tbl_plt_gain_base$available == "Nothing"]
tbl_plt_gain_base$prop_all <- tbl_plt_gain_base$mn_acc_ID - tbl_plt_gain_base$mn_acc_nothing

plot_gain <- function(tbl_plt, ttl) {
  ggplot(tbl_plt, aes(available, delta_mn, group = available)) +
    geom_col(aes(fill = available)) +
    geom_label(aes(y = ifelse(delta_mn > 0, delta_mn - .005, delta_mn + .005), label = round(delta_mn, 3))) +
    coord_cartesian(ylim = c(-0.01, .14)) +
    scale_x_discrete(labels = function(x) str_wrap(x, width = 8)) +
    scale_fill_brewer(palette = "Set1", guide = "none") +
    scale_y_continuous(breaks = c(0, .05, .1), minor_breaks = seq(-.02, .14, by = .01)) +
    labs(y = "Mean Difference", title = ttl) +
    theme_bw() +
    theme(
      axis.title = element_text(size = 20),
      axis.text = element_text(size = 16),
      axis.title.x = element_blank(),
      title = element_text(size = 15),
      strip.background = element_rect(fill = "white", color = "grey"),
      strip.text = element_text(size = 12),
      panel.grid.major.y = element_line(size = 1, colour = "grey80"), 
      panel.grid.minor.y = element_line(size = 0.3, colour = "grey80")
    )
}

pl_id_gain <- plot_gain(tbl_plt_gain_id, "ID: Time Invariant")
pl_history_gain <- plot_gain(tbl_plt_gain_hist, "Shared History")
pl_both_gain <- plot_gain(tbl_plt_gain_both, "ID: Time-Based")

pl_base <- ggplot(tbl_plt_gain_base, aes(available, prop_all, group = available)) +
  geom_col(aes(fill = available)) +
  geom_label(aes(y = prop_all + .05, label = round(prop_all, 3))) +
  coord_cartesian(ylim = c(0, .5)) +
  scale_x_discrete(labels = function(x) str_wrap(x, width = 8)) +
  scale_fill_brewer(palette = "Set1", guide = "none") +
  scale_y_continuous(breaks = seq(0, .5, by = .1), minor_breaks = seq(0, .5, by = .05)) +
  labs(y = "Added Accuracy", title = "Baseline") +
  theme_bw() +
  theme(
    axis.title = element_text(size = 20),
    axis.text = element_text(size = 16),
    axis.title.x = element_blank(),
    title = element_text(size = 15),
    strip.background = element_rect(fill = "white", color = "grey"),
    strip.text = element_text(size = 12),
    panel.grid.major.y = element_line(size = 1, colour = "grey80"), 
    panel.grid.minor.y = element_line(size = 0.3, colour = "grey80")
  )

pl_gains <- arrangeGrob(pl_id_gain, pl_history_gain, pl_both_gain, ncol = 3)

pdf(file = task_settings$pl_dir_gains, 15, 4.5)
grid.draw(pl_gains)
dev.off()




# Joint Plot --------------------------------------------------------------

pdf(file = task_settings$pl_dir_joint, 16.5, 8)
grid.draw(arrangeGrob(
  arrangeGrob(pl_conditions, pl_masklength, pl_base, nrow = 1, widths = c(.25, .45, .3)),
  pl_gains,
  nrow = 2
))
dev.off()



