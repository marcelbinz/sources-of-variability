
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

for (task in c("itc", "mm")){
  task <- "mm"
  if (task == "risky"){
    data_select = 1 # 1 or 2: 1 includes problem repetitions, 2 excludes them
  }
  
  task_settings <- switch(
    task,
    
    ### ITC ###
    "itc" = list(
      task = task,
      pth_conditions = "wandb/wandb_export_itc_conditions_code.csv",
      pth_masklength = "wandb/wandb_export_itc_masklength_code.csv",
      pth_variables = "wandb/wandb_export_itc_variables_code.csv",
      epochthxs = c(201, 250),
      n_epochs = 50,
      indep_vars_labels_incoming = c(
        "T&M&R", "T&M", "M&R", "M",
        "T&R", "T", "R", "Nothing"
      ),
      indep_vars_labels_ordered = c(
        "T&M&R", "T&M", "M&R", "T&R",
        "T", "M", "R", "Nothing"
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
      task = task,
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
      task = task,
      pth_conditions = "wandb/wandb_export_mm_conditions_code.csv",
      pth_masklength = "wandb/wandb_export_mm_masklength_code.csv",
      pth_variables = "wandb/wandb_export_mm_variables_code.csv",
      pth_culture_age = "wandb/wandb_export_mm_culture_code.csv",
      epochthxs = c(201, 250),
      n_epochs = 50,
      indep_vars_labels_incoming = c(
        "A", "A&C", "A&C&R", "A&R", "C", "C&R", "Nothing", "R"
      ),
      indep_vars_labels_ordered = c(
        "A&C&R", "A&C", "A&R", "C&R", "A", "C", "R","Nothing"
      ),
      pl_dir = "figures/mm-conditions.pdf",
      pl_dir_culture = "figures/mm-cultureseq.pdf",
      pl_dir_masklength = "figures/mm-masklength.pdf",
      pl_dir_joint = "figures/mm-jointplot.pdf",
      pl_dir_joint2 = "figures/mm-jointplot2.pdf",
      pl_dir_cond_and_mask = "figures/mm-conditions-and-mask.pdf",
      pl_conditions_labelpos = c(.5, .7)
    ),
    
    
    ### 2ABD ###
    "2abd" = list(
      task = task,
      pth_conditions = "wandb/wandb_export_2abd_conditions.csv",
      pth_masklength = "wandb/wandb_export_2abd_masklength.csv",
      pth_variables = "wandb/wandb_export_2abd_variables.csv",
      epochthxs = c(201, 250),#c(201, 250),
      n_epochs = 50,
      indep_vars_labels_incoming = c("V&R","V", "R","Nothing","M&V&R","M&V", "M&R", "M"),
      indep_vars_labels_ordered = c("M&V&R","M&V", "M&R","V&R","M", "V", "R","Nothing"),
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
  
  
  
  # level 1 shuffling
  l_conditions <- plot_four_conditions(task_settings$pth_conditions, ttl = element_blank(), task_settings)
  pl_conditions <- l_conditions$pl_conditions
  pdf(file = task_settings$pl_dir, 5, 4)
  grid.draw(pl_conditions)
  dev.off()
  
  # level 1 shuffling with culture sequences
  if (task == "mm") {
    l_conditions_culture <- plot_four_conditions(task_settings$pth_culture_age, ttl = element_blank(), task_settings)
    pl_conditions_culture <- l_conditions_culture$pl_conditions
    pdf(file = task_settings$pl_dir_culture, 5, 4)
    grid.draw(pl_conditions_culture)
    dev.off()
  }
  
  
  
  
  # Masklength --------------------------------------------------------------
  
  
  
  l_masklength <- prep_tbls_masklength(task_settings)
  tbl_plt_masklength <- l_masklength$tbl_plt_masklength
  tbl_masklength_loss <- l_masklength$tbl_masklength_loss
  
  blue_cols <- c(
    '#E0F3F8', '#C2E0EE', '#A6CEE3', '#6AA6CC', 
    '#1F78B4', '#083F82', '#041C3A'
  )[(8-(length(levels(tbl_plt_masklength$masklength))-1)):7]
  
  pl_masklength <- ggplot(tbl_plt_masklength, aes(masklength, accuracy)) +
    geom_hline(yintercept = .5, linetype = "dotdash", linewidth = 1, color = "red", alpha = .3) +
    geom_line(color = "#1F78B4", linewidth = 1, aes(group = 1)) +
    geom_col(aes(fill = masklength), width = .5, alpha = .5) +
    geom_point(color = "white", size = 4) +
    geom_point(aes(color = masklength)) +
    geom_label(aes(y = accuracy - .05, label = round(accuracy, 3))) +
    theme_bw() +
    scale_x_discrete(labels = function(x) str_wrap(x, width = 30)) +
    scale_y_continuous(expand = expansion(add = c(0.02, 0))) +
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
  
  
  tbl_variables <- prep_tbl_variables(task_settings)
  
  # avg no history / baseline plot
  tbl_plt_gain_base <- subselect_conditions(tbl_variables, rep("No History", 2), c("Shared", "Shared"))
  tbl_plt_gain_base$mn_acc_all <- tbl_plt_gain_base$mn_acc_ID[
    tbl_plt_gain_base$available == "T&M&R" |
      tbl_plt_gain_base$available == "P&V&R" |
      tbl_plt_gain_base$available == "M&V&R" |
      tbl_plt_gain_base$available == "A&C&R"
  ]
  tbl_plt_gain_base$mn_acc_nothing <- tbl_plt_gain_base$mn_acc_ID[tbl_plt_gain_base$available == "Nothing"]
  tbl_plt_gain_base$prop_all <- tbl_plt_gain_base$mn_acc_ID - tbl_plt_gain_base$mn_acc_nothing
  
  
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
  
  # three deltas to baseline
  tbl_plt_gain_id <- subselect_conditions(tbl_variables, rep("No History", 2), c("ID", "Shared"))
  tbl_plt_gain_hist <- subselect_conditions(tbl_variables, c("History", "No History"), rep("Shared", 2))
  tbl_plt_gain_both <- subselect_conditions(tbl_variables, c("History", "No History"), c("ID", "Shared"))
  
  
  ymax <- ifelse(task == "2abd", .25, .14)
  pl_id_gain <- plot_gain(tbl_plt_gain_id, "ID: Time Invariant", ymax)
  pl_history_gain <- plot_gain(tbl_plt_gain_hist, "Shared History", ymax)
  pl_both_gain <- plot_gain(tbl_plt_gain_both, "ID: Time-Based", ymax)
  
  pl_gains <- arrangeGrob(pl_id_gain, pl_history_gain, pl_both_gain, ncol = 3)
  
  pdf(file = task_settings$pl_dir_gains, 15, 4.5)
  grid.draw(pl_gains)
  dev.off()
  
  pl_variables_original <- plot_variables(tbl_variables, cd = "ID-History", mn_acc, FALSE, "Original ID & Original History")
  pl_variables_shared_nohistory <- plot_variables(tbl_variables, cd = "Shared-NoHistory", mn_acc, FALSE, "Shuffled ID & Shuffled History")
  pl_delta <- plot_variables(
    tbl_plt_gain_both %>% mutate(Condition = "compare"), 
    "compare", delta_mn, TRUE, "Difference", 
    max_y_delta = max(tbl_plt_gain_both$delta_mn),
    min_y_delta = min(tbl_plt_gain_both$delta_mn)
  )
  
  
  # Joint Plot 1 ------------------------------------------------------------
  
  pdf(file = task_settings$pl_dir_joint, 16.5, 8)
  grid.draw(arrangeGrob(
    arrangeGrob(pl_conditions, pl_masklength, pl_gain_base, nrow = 1, widths = c(.25, .45, .3)),
    pl_gains,
    nrow = 2
  ))
  dev.off()
  
  
  
  # Joint Plot 2 ------------------------------------------------------------
  
  pl_empty <- ggplot() + theme_minimal()
  
  pdf(file = task_settings$pl_dir_joint2, 12, 6)
  grid.draw(arrangeGrob(
    arrangeGrob(pl_empty, pl_conditions, pl_masklength, nrow = 1, widths = c(.25, .3, .45)),
    arrangeGrob(pl_variables_shared_nohistory, pl_variables_original, nrow = 1),
    nrow = 2
  ))
  dev.off()
  
  png(file = str_replace(task_settings$pl_dir_joint2, ".pdf", ".png"), 11.1, 5.55, units="in", res = 300)
  grid.draw(arrangeGrob(
    arrangeGrob(pl_empty, pl_conditions, pl_masklength, nrow = 1, widths = c(.25, .3, .45)),
    arrangeGrob(pl_variables_shared_nohistory, pl_variables_original, nrow = 1),
    nrow = 2
  ))
  dev.off()
  
  
  
  # Data Set Size Analysis: ITC & MM -----------------------------------------
  
  if (task == "itc") {
    pths_sizes <- c(
      "wandb/wandb_export_itc_conditions_fewdata_code.csv",
      "wandb/wandb_export_itc_conditions_meddata_code.csv",
      "wandb/wandb_export_itc_conditions_muchdata_code.csv"
    )
    ttls_sizes <- c(
      "Nr. Trials = 14",
      "Nr. Trials = 48",
      "Nr. Trials = 195"
    )
    pth_fig <- "figures/itc-dataset-sizes.pdf"
    
    
    l_pl_sizes <- map2(pths_sizes, ttls_sizes, plot_four_conditions, task_settings = task_settings)
    
    pls_conditions_sizes <- do.call(arrangeGrob, c(map(l_pl_sizes, 1), nrow = 1))
    
    
    pdf(file = pth_fig, ifelse(task == "itc", 10, 6.6), 3.5)
    grid.draw(pls_conditions_sizes)
    dev.off()
  }
  
  
  
  # Just Conditions & Masklength --------------------------------------------
  
  pdf(file = task_settings$pl_dir_cond_and_mask, 8.5, 2.75)
  grid.draw(arrangeGrob(pl_conditions, pl_masklength, nrow = 1, widths = c(.35, .65)))
  dev.off()
  
  
  
  # Tables with Accuracies and Losses ---------------------------------------
  
  one <- l_conditions$tbl_conditions %>%
    rename(Accuracy = mn_acc, Loss = mn_loss) %>%
    select(-c(Condition, shuffle, sd_acc, sd_loss)) %>%
    relocate(Accuracy, .after = ID) %>% 
    relocate(Loss, .after = Accuracy) %>%
    mutate(Available = "All", Masklength = "All") %>%
    relocate(Available, .after = "ID") %>%
    relocate(Masklength, .after = "Available") %>%
    mutate(Analysis = "Level 1", History = str_replace(History, "\\n", " "))
  
  two <- tbl_masklength_loss %>% 
    rename(Accuracy = mn_acc, Loss = mn_loss, Masklength = masklength) %>%
    mutate(History = "Original History", ID = "Original ID", Available = "All") %>%
    relocate(History, .before = Masklength) %>%
    relocate(ID, .before = Masklength) %>%
    relocate(Available, .before = "Masklength") %>%
    mutate(Analysis = "Masklength")
  
  three <- tbl_variables %>% 
    rename(Accuracy = mn_acc, Loss = mn_loss, Available = available) %>%
    select(-c(Condition, shuffle, se_acc, se_loss))
  three$History <- factor(three$History, labels = c("Shuffled History", "Original History"))
  three$ID <- factor(three$ID, labels = c("Shuffled ID", "Original ID"))
  three$Masklength <- "All"
  three <- three %>% relocate(Masklength, .after = Available) %>% mutate(Analysis = "Level 2")
  
  write_csv(
    bind_rows(one, two, three) %>% 
      mutate(Accuracy = round(Accuracy, 3), Loss = round(Loss, 3)), 
    file = str_c("data/accuracies-and-losses-", task, ".csv")
  )
  
  
}
