
# Load Packages -----------------------------------------------------------



rm(list = ls())

library(tidyverse)
library(grid)
library(gridExtra)

home_grown <- c("R/utils.R")
walk(home_grown, source)

if (!dir.exists("figures")) {dir.create("figures")}



# Plotting Settings -------------------------------------------------------

## define task settings and which data to select
## note. data_select only required for risky data set, not itc


task <- c("itc", "risky", "mm", "2abd")[3]
if (task == "risky"){
  data_select = 1 # 1 or 2: 1 includes problem repetitions, 2 excludes them
}


task_settings <- switch(
  task,
  
  ### ITC ###
  "itc" = list(
    pth_conditions = "wandb/wandb_export_itc_conditions.csv",
    pth_masklength = "wandb/wandb_export_itc_masklength.csv",
    pth_variables = "wandb/wandb_export_itc_variables.csv",
    epochthxs = c(201, 250),
    n_epochs = 50,
    indep_vars_labels_incoming = c(
      "T&V&R", "V&R",
      "T&R", "R", "Nothing"
    ),
    indep_vars_labels_ordered = c(
      "T&V&R", "V&R",
      "T&R", "R", "Nothing"
    ),
    pl_dir = "figures/itc-conditions.pdf",
    pl_dir_masklength = "figures/itc-masklength.pdf",
    pl_dir_gains = "figures/itc-gains.pdf",
    pl_dir_joint = "figures/itc-jointplot.pdf",
    pl_dir_joint2 = "figures/itc-jointplot-2.pdf",
    pl_dir_cond_and_mask = "figures/itc-conditions-and-mask.pdf",
    pl_conditions_labelpos = c(.5, .3)
  ),
  
  
  ### RISKY ###
  "risky" = list(
    pth_conditions = c(
      "wandb/wandb_export_risky_conditions_peterson-select.csv",
      "wandb/wandb_export_risky_conditions_only-first-problem.csv"
    )[data_select],
    # only ran on peterson select, because no effects (/not enough signal) for first problem
    pth_masklength = c(
      "wandb/wandb_export_risky_masklength_peterson-select.csv",
      "wandb/wandb_export_risky_masklength_only-first-problem.csv"
    )[data_select],
    pth_variables = c(
      "wandb/wandb_export_risky_variables_peterson-select.csv",
      "wandb/wandb_export_risky_variables_only-first-problem.csv"
    )[data_select],
    epochthxs = c(201, 250),#c(201, 250), 
    n_epochs = 50,
    indep_vars_labels_incoming = c(
      "P&V&R", "V&R",
      "P&R", "R", "Nothing"
    ),
    indep_vars_labels_ordered = c(
      "P&V&R", "V&R",
      "P&R", "R", "Nothing"
    ),
    pl_dir = c(
      "figures/risky-peterson-select-conditions.pdf",
      "figures/risky-first-problem-conditions.pdf"
    )[data_select],
    pl_dir_masklength = c(
      "figures/risky-peterson-select-masklength.pdf",
      "figures/risky-only-first-problem-masklength.pdf"
    )[data_select],
    pl_dir_gains = c(
      "figures/risky-peterson-select-gains.pdf",
      "figures/risky-first-problem-gains.pdf"
    )[data_select],
    pl_dir_joint = c(
      "figures/risky-peterson-select-jointplot.pdf",
      "figures/risky-first-problem-jointplot.pdf"
    )[data_select],
    pl_dir_joint2 = c(
      "figures/risky-peterson-select-jointplot-2.pdf",
      "figures/risky-first-problem-jointplot-2.pdf"
    )[data_select],
    pl_dir_cond_and_mask = c(
      "figures/risky-peterson-select-conditions-and-mask.pdf",
      "figures/risky-first-problem-conditions-and-mask.pdf"
    )[data_select],
    pl_conditions_labelpos = c(.5, .8)
  ),
  
  
  ### MM ###
  "mm" = list(
    pth_conditions = "wandb/wandb_export_mm_conditions.csv",
    pth_masklength = "wandb/wandb_export_mm_masklength.csv",
    pth_variables = "wandb/wandb_export_mm_culture_age_small.csv",
    pth_culture_age_small = "wandb/wandb_export_mm_culture_age_small.csv",
    pth_culture_age_small_shared = "wandb/wandb_export_mm_culture_age_small_shared.csv",
    pth_culture_age_small_id = "wandb/wandb_export_mm_culture_age_small_id.csv",
    epochthxs = c(201, 250),
    n_epochs = 50,
    indep_vars_labels_incoming = c(
      "Nothing", "A", "C", "A&C"
    ),
    indep_vars_labels_ordered = c(
      "Nothing", "A", "C", "A&C"
    ),
    pl_dir = "figures/mm-conditions.pdf",
    pl_dir_masklength = "figures/mm-masklength.pdf",
    pl_dir_joint = "figures/mm-jointplot.pdf",
    pl_dir_joint2 = "figures/mm-jointplot2.pdf",
    pl_dir_cond_and_mask = "figures/mm-conditions-and-mask.pdf",
    pl_conditions_labelpos = c(.5, .7)
  ),
  
  
  ### 2ABD ###
  "2abd" = list(
    pth_conditions = "wandb/wandb_export_2abd_conditions.csv",
    pth_masklength = "wandb/wandb_export_2abd_masklength.csv",
    pth_variables = "wandb/wandb_export_2abd_variables.csv",
    epochthxs = c(201, 250),
    n_epochs = 50,
    indep_vars_labels_incoming = c("V&R", "R", "Nothing", "M&V&R", "M&R"),
    indep_vars_labels_ordered = c("M&V&R","M&R", "V&R", "R", "Nothing"),
    pl_dir = "figures/2abd-conditions.pdf",
    pl_dir_masklength = "figures/2abd-masklength.pdf",
    pl_dir_gains = "figures/2abd-gains.pdf",
    pl_dir_joint = "figures/2abd-jointplot.pdf",
    pl_dir_joint2 = "figures/2abd-jointplot2.pdf",
    pl_dir_cond_and_mask = "figures/2abd-conditions-and-mask.pdf",
    pl_conditions_labelpos = c(.5, .3)
  )
)



# Four Conditions ---------------------------------------------------------




pl_conditions <- plot_four_conditions(task_settings$pth_conditions, ttl = element_blank(), task_settings)

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
colnames(tbl_constrain) <- c("Epoch", masklenghts, "All", "No Hist")


# note. in mm not all masklenghts converged between 201 and 250
# masklength == 10 overfit to the train data between 201 and 250
# therefore, use a range, in which performance was stable
if (task == "mm") task_settings$epochthxs <- c(161, 210)

tbl_constrain_long <- tbl_constrain %>% pivot_longer(-"Epoch")  %>%
  filter(between(Epoch, task_settings$epochthxs[1], task_settings$epochthxs[2])) 
tbl_constrain_long$name <- factor(
  tbl_constrain_long$name,
  levels = c(sort(as.numeric(masklenghts)), "All", "No Hist")
)

tbl_plt_masklength <- tbl_constrain_long %>%
  group_by(name) %>%
  summarize(accuracy = mean(value, na.rm = TRUE), .groups = "drop")

blue_cols <- c(
  '#E0F3F8', '#C2E0EE', '#A6CEE3', '#6AA6CC', 
  '#1F78B4', '#083F82', '#041C3A'
)[(7-(length(masklenghts))):7]

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
    axis.title = element_text(size = 16),
    axis.text = element_text(size = 14),
    title = element_text(size = 16),
    strip.background = element_rect(fill = "white"), 
    axis.title.x = element_blank()
  ) +
  scale_color_manual(values = c(
    blue_cols, "#F28E2B"), guide = "none") +
  scale_fill_manual(values = c(
    blue_cols, "#F28E2B"), guide = "none") +
  coord_cartesian(ylim = c(.5, 1))

pdf(file = task_settings$pl_dir_masklength, 7.75, 4)
grid.draw(pl_masklength)
dev.off()


# Effect Decomposition ----------------------------------------------------

# effect decomposition not available for MM dataset
if (task != "mm") {
  
  tbl_variables <- prep_tbl_variables(task_settings)
  
  # avg no history / baseline plot
  tbl_plt_gain_base <- subselect_conditions(tbl_variables, rep("No History", 2), c("Shared", "Shared"))
  tbl_plt_gain_base$mn_acc_all <- tbl_plt_gain_base$mn_acc_ID[
    tbl_plt_gain_base$available == "T&V&R" |
      tbl_plt_gain_base$available == "P&V&R" |
      tbl_plt_gain_base$available == "M&V&R"
  ]
  tbl_plt_gain_base$mn_acc_nothing <- tbl_plt_gain_base$mn_acc_ID[tbl_plt_gain_base$available == "Nothing"]
  tbl_plt_gain_base$prop_all <- tbl_plt_gain_base$mn_acc_ID - tbl_plt_gain_base$mn_acc_nothing
  
  # three deltas to baseline
  tbl_plt_gain_id <- subselect_conditions(tbl_variables, rep("No History", 2), c("ID", "Shared"))
  tbl_plt_gain_hist <- subselect_conditions(tbl_variables, c("History", "No History"), rep("Shared", 2))
  tbl_plt_gain_both <- subselect_conditions(tbl_variables, c("History", "No History"), c("ID", "Shared"))
  
  
  plot_gain <- function(tbl_plt, ttl, ymax = .14) {
    ggplot(tbl_plt, aes(available, delta_mn, group = available)) +
      geom_col(aes(fill = available)) +
      geom_label(aes(y = ifelse(delta_mn > 0, delta_mn - .005, delta_mn + .005), label = round(delta_mn, 3))) +
      coord_cartesian(ylim = c(-0.01, ymax)) +
      scale_x_discrete(labels = function(x) str_wrap(x, width = 8)) +
      scale_fill_brewer(palette = "Set1", guide = "none") +
      scale_y_continuous(breaks = c(0, .25, by = .05), minor_breaks = seq(-.02, ymax, by = .01)) +
      labs(y = "Mean Difference", title = ttl) +
      theme_bw() +
      theme(
        axis.title = element_text(size = 16),
        axis.text = element_text(size = 16),
        axis.title.x = element_blank(),
        title = element_text(size = 16),
        strip.background = element_rect(fill = "white", color = "grey"),
        strip.text = element_text(size = 12),
        panel.grid.major.y = element_line(size = 1, colour = "grey80"), 
        panel.grid.minor.y = element_line(size = 0.3, colour = "grey80")
      )
  }
  
  pl_gain_base <- ggplot(tbl_plt_gain_base, aes(available, prop_all, group = available)) +
    geom_col(aes(fill = available)) +
    geom_label(aes(y = prop_all + .05, label = round(prop_all, 3))) +
    coord_cartesian(ylim = c(0, .5)) +
    scale_x_discrete(labels = function(x) str_wrap(x, width = 8)) +
    scale_fill_brewer(palette = "Set1", guide = "none") +
    scale_y_continuous(breaks = seq(0, .5, by = .1), minor_breaks = seq(0, .5, by = .05)) +
    labs(y = "Added Accuracy", title = "Baseline") +
    theme_bw() +
    theme(
      axis.title = element_text(size = 16),
      axis.text = element_text(size = 16),
      axis.title.x = element_blank(),
      title = element_text(size = 16),
      strip.background = element_rect(fill = "white", color = "grey"),
      strip.text = element_text(size = 12),
      panel.grid.major.y = element_line(size = 1, colour = "grey80"), 
      panel.grid.minor.y = element_line(size = 0.3, colour = "grey80")
    )
  
  ymax <- ifelse(task == "2abd", .25, .14)
  pl_id_gain <- plot_gain(tbl_plt_gain_id, "ID: Time Invariant", ymax)
  pl_history_gain <- plot_gain(tbl_plt_gain_hist, "Shared History", ymax)
  pl_both_gain <- plot_gain(tbl_plt_gain_both, "ID: Time-Based", ymax)
  
  
  
  pl_gains <- arrangeGrob(pl_id_gain, pl_history_gain, pl_both_gain, ncol = 3)
  
  pdf(file = task_settings$pl_dir_gains, 15, 4.5)
  grid.draw(pl_gains)
  dev.off()
  
  
  ggplot(
    tbl_variables %>% 
      #filter(Condition == "Shared-NoHistory"),
      filter(Condition == "ID-History"),
    aes(available, mn_acc, group = available)
  ) + 
    geom_hline(yintercept = .5, color = "darkred", alpha = .5, size = 1.5, linetype = "dotdash") +
    geom_col(aes(fill = available), width = .6) +
    scale_fill_brewer(palette = "Set1", guide = "none") +
    coord_cartesian(ylim = c(.5, 1)) +
    scale_y_continuous(breaks = seq(.5, 1, by = .1), minor_breaks = seq(.5, 1, by = .05)) +
    labs(y = "Test Accuracy") +
    theme_bw() +
    theme(
      axis.title = element_text(size = 16),
      axis.text = element_text(size = 16),
      axis.title.x = element_blank(),
      title = element_text(size = 16),
      strip.background = element_rect(fill = "white", color = "grey"),
      strip.text = element_text(size = 12),
      panel.grid.major.y = element_line(size = 1, colour = "grey80"), 
      panel.grid.minor.y = element_line(size = 0.3, colour = "grey80")
    )
  
  pl_variables_original <- plot_variables(tbl_variables, cd = "ID-History", mn_acc, FALSE, "ID & History")
  pl_variables_shared_nohistory <- plot_variables(tbl_variables, cd = "Shared-NoHistory", mn_acc, FALSE, "Shared & No History")
  pl_delta <- plot_variables(
    tbl_plt_gain_both %>% mutate(Condition = "compare"), 
    "compare", delta_mn, TRUE, "Difference", 
    max_y_delta = max(tbl_plt_gain_both$delta_mn),
    min_y_delta = min(tbl_plt_gain_both$delta_mn)
  )
  
}

if (task == "mm") {
  tbl_variables_shared <- prep_tbl_culture(task_settings, "pth_culture_age_small", c(451, 500))
  tbl_variables_shared <- tbl_variables_shared %>%
    mutate(
      available = str_c(ifelse(Age == "Age", "A", ""), ifelse(Culture == "Culture", "C", "")),
      available = ifelse(available == "", "Nothing", available),
      available = ifelse(available == "AC", "A&C", available),
      available = factor(available, levels = c("A&C", "A", "C", "Nothing"), ordered = TRUE),
      ID = "Shared",
      History = "No History",
      Condition = "Shared",
      shuffle = "dummy"
    )
  
  # this has to be replaced by the proper data!
  tbl_variables_id <- tbl_variables_shared %>% mutate(ID = "ID", Condition = "ID")
  
  pl_variables_shared_nohistory <- plot_variables(tbl_variables_shared, "Shared", mn_acc, FALSE, "Shared")
  pl_variables_original <- plot_variables(tbl_variables_id, "ID", mn_acc, FALSE, "ID")
  tbl_variables <- bind_rows(tbl_variables_shared, tbl_variables_id)
  tbl_plt_gain_both <- subselect_conditions(tbl_variables, c("No History", "No History"), c("ID", "Shared")) %>%
    mutate(Condition = "compare")
  
  pl_delta <- plot_variables(
    tbl_plt_gain_both, cd = "compare", delta_mn, TRUE, "Difference", 
    max_y_delta = max(tbl_plt_gain_both$delta_mn),
    min_y_delta = min(tbl_plt_gain_both$delta_mn)
  )
  
}


# Joint Plot --------------------------------------------------------------


if (task != "mm"){
  pdf(file = task_settings$pl_dir_joint, 16.5, 8)
  grid.draw(arrangeGrob(
    arrangeGrob(pl_conditions, pl_masklength, pl_gain_base, nrow = 1, widths = c(.25, .45, .3)),
    pl_gains,
    nrow = 2
  ))
  dev.off()
} else {
  pdf(file = task_settings$pl_dir_joint, 11, 4)
  grid.draw(
    arrangeGrob(pl_conditions, pl_masklength,nrow = 1, widths = c(.35, .65)),
  )
  dev.off()
}


# Joint Plot 2 ------------------------------------------------------------
pdf(file = task_settings$pl_dir_joint2, 12, 6)
grid.draw(arrangeGrob(
  arrangeGrob(pl_conditions, pl_masklength, nrow = 1, widths = c(.35, .65)),
  arrangeGrob(pl_variables_shared_nohistory, pl_variables_original, pl_delta, nrow = 1),
  nrow = 2
))
dev.off()



# Data Set Size Analysis: ITC & MM -----------------------------------------

if (task %in% c("itc", "mm")) {
  if (task == "itc") {
    pths_sizes <- c(
      "wandb/wandb_export_itc_conditions_fewdata.csv",
      "wandb/wandb_export_itc_conditions_meddata.csv",
      "wandb/wandb_export_itc_conditions_muchdata.csv"
    )
    ttls_sizes <- c(
      "Nr. Trials = 14",
      "Nr. Trials = 48",
      "Nr. Trials = 195"
    )
    pth_fig <- "figures/itc-dataset-sizes.pdf"
  } else if (task == "mm") {
    pths_sizes <- c(
      "wandb/wandb_export_mm_conditions_38_traintrials.csv",
      "wandb/wandb_export_mm_conditions_55_traintrials.csv"
    )
    ttls_sizes <- c(
      "Nr. Trials = 48",
      "Nr. Trials = 85"
    )
    pth_fig <- "figures/mm-dataset-sizes.pdf"
  }
  
  
  l_pl_sizes <- map2(pths_sizes, ttls_sizes, plot_four_conditions, task_settings = task_settings)
  
  pls_conditions_sizes <- do.call(arrangeGrob, c(l_pl_sizes, nrow = 1))
  
  
  pdf(file = pth_fig, ifelse(task == "itc", 10, 6.6), 3.5)
  grid.draw(pls_conditions_sizes)
  dev.off()
}



# Just Conditions & Masklength --------------------------------------------

pdf(file = task_settings$pl_dir_cond_and_mask, 8.5, 2.75)
grid.draw(arrangeGrob(pl_conditions, pl_masklength, nrow = 1, widths = c(.35, .65)))
dev.off()





# Work in Progress: Lineplots for Level 2 Shuffling? ----------------------
# 
# tbl_plt_gain_base <- tbl_plt_gain_base %>%
#   filter(available != "R") %>%
#   mutate(
#     "Mean" = factor(str_detect(available, "M"), labels = c("No Mean", "Mean"), ordered = TRUE),
#     "Variance" = factor(str_detect(available, "V"), labels = c("No Variance", "Variance"), ordered = TRUE)
#   )
# 
# 
# ggplot(tbl_plt_gain_base, aes(Mean, mn_acc_ID, group = Variance)) +
#   geom_line(aes(color = Variance)) +
#   geom_point(color = "white", size = 3) +
#   geom_point(aes(color = Variance)) +
#   coord_cartesian(ylim = c(.5, 1)) +
#   scale_x_discrete(labels = function(x) str_wrap(x, width = 8)) +
#   scale_color_brewer(palette = "Set1") +
#   scale_y_continuous(breaks = seq(.5, 1, by = .1), minor_breaks = seq(0, .5, by = .05)) +
#   labs(y = "Test Accuracy", title = "Baseline") +
#   theme_bw() +
#   theme(
#     axis.title = element_text(size = 16),
#     axis.text = element_text(size = 16),
#     axis.title.x = element_blank(),
#     title = element_text(size = 16),
#     strip.background = element_rect(fill = "white", color = "grey"),
#     strip.text = element_text(size = 12),
#     panel.grid.major.y = element_line(size = 1, colour = "grey80"), 
#     panel.grid.minor.y = element_line(size = 0.3, colour = "grey80")
#   )
