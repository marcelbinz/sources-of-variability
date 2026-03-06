
# Load Packages -----------------------------------------------------------



rm(list = ls())

library(tidyverse)
library(grid)
library(gridExtra)

if (!dir.exists("figures")) {dir.create("figures")}



# Plotting Settings -------------------------------------------------------

## define task settings and which data to select
## note. data_select only required for risky data set, not itc


task <- c("itc", "risky")[1]
if (task == "risky"){
  data_select = 1 # or 2: 1 includes problem repetitions, 2 excludes them
}


# TODO
# for risky data set, load wandb export with all variable combinations on level 2
# i.e., four conditions with each five variable combinations

task_settings <- switch(
  task,
  "itc" = list(
    pth_conditions = "wandb/wandb_export_2026-02-19T17_18_32.282+01_00.csv",
    pth_masklength = "wandb/wandb_export_itc_constrain.csv",
    epochthxs = c(101, 150),
    n_epochs = 50,
    indep_vars_labels = c(
      "Time & Val & Resp", "Val & Resp",
      "Time & Resp", "Resp", "Nothing"
    ),
    pl_dir = "figures/itc-conditions.pdf",
    pl_dir_masklength = "figures/itc-masklength.pdf",
    pl_dir_gains = "figures/itc-gains.pdf"
  ),
  "risky" = list(
    pth_conditions = c(
      "wandb/wandb_export_risky_peterson-select.csv",
      "wandb/wandb_export_risky_only_first_problem.csv"
    )[data_select],
    pth_masklength = "wandb/wandb_export_risky_masklength.csv",
    epochthxs = c(76, 150), 
    n_epochs = 75,
    indep_vars_labels = c(
      "Prob & Val & Resp", "Val & Resp",
      "Prob & Resp", "Resp", "Nothing"
    ),
    pl_dir = c(
      "figures/risky-peterson-select-conditions.pdf",
      "figures/risky-first-problem-conditions.pdf"
    )[data_select],
    pl_dir_masklength = "figures/risky-masklength.pdf",
    pl_dir_gains = "figures/risky-gains.pdf"
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
tbl_plt <- tbl_accuracy_long %>%
  filter(between(epoch, task_settings$epochthxs[1], task_settings$epochthxs[2])) %>%
  group_by(Condition, shuffle) %>%
  summarize(mn_acc = mean(value), sd = sd(value)/sqrt(task_settings$n_epochs), .groups = "drop")

tbl_plt$Condition <- factor(
  tbl_plt$Condition,
  levels = c("shared_nohist", "shared_hist", "id_nohist", "original"),
  labels = c("Shared-NoHistory", "Shared-History", "ID-NoHistory", "ID-History")
)

tbl_plt$available <- tbl_plt$shuffle
tbl_plt$available <- factor(
  tbl_plt$available, labels = task_settings$indep_vars_labels
)


# plot and save

tbl_plt <- tbl_plt %>% mutate(
  History = as.numeric(!str_detect(Condition, "NoHistory")),
  History = factor(History, labels = c("No History", "History"), ordered = TRUE),
  ID = as.numeric(str_detect(Condition, "ID")),
  ID = factor(ID, labels = c("Shared", "ID"), ordered = TRUE)
)


pl_conditions <- ggplot(tbl_plt %>% filter(shuffle == "nothing"), aes(History, mn_acc, group = ID)) +
  geom_hline(yintercept = .5, linetype = "dotdash", linewidth = 1, color = "red", alpha = .3) +
  geom_errorbar(aes(
    History, ymin = mn_acc - 2*sd, ymax = mn_acc + 2*sd, color = ID
  ), width = .2, linewidth = .75) +
  geom_line((aes(color = ID))) +
  geom_point(color = "white", size = 5) +
  geom_point(aes(color = ID), size = 3) +
  scale_color_brewer(palette = "Set1") +
  labs(y = "Test Accuracy") +
  theme_bw() +
  theme(
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12),
    legend.title = element_blank(),
    axis.title.x = element_blank()
  ) +
  coord_cartesian(ylim = c(.5, 1))



pdf(file = task_settings$pl_dir, 5, 4)
grid.draw(pl_conditions)
dev.off()




# Masklength --------------------------------------------------------------


tbl_constrain <- read_csv(task_settings$pth_masklength)

tbl_constrain <- tbl_constrain %>% select(
  c("epoch",
    contains("original") & contains("dev/acc_epoch") & !contains("MIN") & !contains("MAX")
  ))

masklenghts <- str_match(colnames(tbl_constrain), "windowsize=([0-9]+) | (epoch)")[, 2]
masklenghts <- masklenghts[!is.na(masklenghts)]

#Window=
colnames(tbl_constrain) <- c("Epoch", masklenghts, "All Previous")

tbl_constrain_long <- tbl_constrain %>% pivot_longer(-"Epoch")
tbl_constrain_long$name <- factor(
  tbl_constrain_long$name,
  levels = c(sort(as.numeric(masklenghts)), "All Previous")
)

tbl_plt_masklength <- tbl_constrain_long %>%
  group_by(name) %>%
  summarize(accuracy = mean(value), .groups = "drop")

pl_masklength <- ggplot(tbl_plt_masklength, aes(name, accuracy)) +
  geom_hline(yintercept = .5, linetype = "dotdash", linewidth = 1, color = "red", alpha = .3) +
  geom_line(color = "#1F78B4", linewidth = 1, aes(group = 1)) +
  geom_col(aes(fill = name), width = .5, alpha = .5) +
  geom_point(color = "white", size = 4) +
  geom_point(aes(color = name)) +
  geom_label(aes(y = accuracy - .05, label = round(accuracy, 3))) +
  theme_bw() +
  scale_x_discrete(expand = c(0.15, 0)) +
  scale_y_continuous(expand = c(0.01, 0)) +
  labs(x = "Size of Window", y = "Accuracy") +
  theme(
    strip.background = element_rect(fill = "white"), text = element_text(size = 22)
  ) +
  scale_color_manual(values = c("#E0F3F8", "#A6CEE3", "#1F78B4", "#08519C", "#08306B"), guide = "none") +
  scale_fill_manual(values = c("#E0F3F8", "#A6CEE3", "#1F78B4", "#08519C", "#08306B"), guide = "none") +
  coord_cartesian(ylim = c(.5, .9))

pdf(file = task_settings$pl_dir_masklength, 5.75, 3.5)
grid.draw(pl_masklength)
dev.off()


# Effect Decomposition ----------------------------------------------------


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

tbl_plt_gain_id <- subselect_conditions(tbl_plt, rep("No History", 2), c("ID", "Shared"))
tbl_plt_gain_hist <- subselect_conditions(tbl_plt, c("History", "No History"), rep("Shared", 2))
tbl_plt_gain_both <- subselect_conditions(tbl_plt, c("History", "No History"), c("ID", "Shared"))


plot_gain <- function(tbl_plt, ttl) {
  ggplot(tbl_plt, aes(available, delta_mn, group = available)) +
    geom_col(aes(fill = available)) +
    geom_label(aes(y = ifelse(delta_mn > 0, delta_mn - .005, delta_mn + .005), label = round(delta_mn, 3))) +
    coord_cartesian(ylim = c(-0.01, .065)) +
    scale_x_discrete(labels = function(x) str_wrap(x, width = 8)) +
    scale_fill_brewer(palette = "Set1", guide = "none") +
    scale_y_continuous(breaks = c(0, .05), minor_breaks = seq(-.02, .08, by = .01)) +
    labs(y = "Mean Difference", x = "Available Variables", title = ttl) +
    theme_bw() +
    theme(
      axis.title = element_text(size = 14),
      axis.text = element_text(size = 12),
      strip.background = element_rect(fill = "white", color = "grey"),
      strip.text = element_text(size = 12),
      panel.grid.major.y = element_line(size = 1, colour = "grey80"), 
      panel.grid.minor.y = element_line(size = 0.3, colour = "grey80")
    )
}

pl_id_gain <- plot_gain(tbl_plt_gain_id, "Individual Differences")
pl_history_gain <- plot_gain(tbl_plt_gain_hist, "History")
pl_both_gain <- plot_gain(tbl_plt_gain_both, "Both")

pl_gains <- arrangeGrob(pl_id_gain, pl_history_gain, pl_both_gain, ncol = 3)

pdf(file = task_settings$pl_dir_gains, 11, 4.5)
grid.draw(pl_gains)
dev.off()
