
# Load Packages -----------------------------------------------------------



rm(list = ls())

library(tidyverse)
library(grid)
library(gridExtra)

home_grown <- c("R/utils.R")
walk(home_grown, source)


if (!dir.exists("figures")) {dir.create("figures")}


# ITC ---------------------------------------------------------------------


# conditions
task_settings = list(
  pth_conditions = "wandb/wandb_export_itc_conditions.csv",
  pth_masklength = "wandb/wandb_export_itc_masklength.csv",
  pth_variables = "wandb/wandb_export_itc_variables.csv",
  epochthxs = c(201, 250),
  n_epochs = 50,
  indep_vars_labels = c(
    "T&V&R", "V&R",
    "T&R", "R", "Nothing"
  )
)

plt_conditions <- plot_four_conditions(task_settings$pth_conditions, element_blank(), task_settings)


# masklenghts

l_constrain <- prep_tbl_masklength(task_settings, "ITC")
tbl_constrain_long <- l_constrain$tbl_constrain_long
masklengths <- l_constrain$masklengths

tbl_plt_masklength <- tbl_constrain_long %>%
  group_by(name) %>%
  summarize(accuracy = mean(value, na.rm = TRUE), .groups = "drop")

blue_cols <- c(
  '#E0F3F8', '#C2E0EE', '#A6CEE3', '#6AA6CC', 
  '#1F78B4', '#083F82', '#041C3A'
)[(7-(length(masklengths))):7]

plt_masklength <- ggplot(tbl_plt_masklength, aes(name, accuracy)) +
  geom_hline(yintercept = .5, linetype = "dotdash", linewidth = 1, color = "red", alpha = .3) +
  geom_line(color = "#1F78B4", linewidth = 1, aes(group = 1)) +
  geom_col(aes(fill = name), width = .5, alpha = .5) +
  geom_point(color = "white", size = 4) +
  geom_point(aes(color = name)) +
  geom_label(aes(y = accuracy - .05, label = round(accuracy, 3))) +
  theme_bw() +
  scale_x_discrete(labels = function(x) str_wrap(x, width = 30)) +
  scale_y_continuous(expand = expansion(add = c(.02, 0))) +
  labs(x = "Size of Window", y = "Test Accuracy") +
  theme(
    axis.title = element_text(size = 20),
    axis.text = element_text(size = 18),
    title = element_text(size = 15),
    strip.background = element_rect(fill = "white"), 
    axis.title.x = element_blank()
  ) +
  scale_color_manual(values = c(
    blue_cols, "#F28E2B"), guide = "none") +
  scale_fill_manual(values = c(
    blue_cols, "#F28E2B"), guide = "none") +
  coord_cartesian(ylim = c(.5, 1))


# variables

tbl_variables <- prep_tbl_variables(task_settings)



#tbl_variables %>% filter(Condition == "ID-History") %>%
plt_shared_nohist <- plot_variables(tbl_variables, "Shared-NoHistory", mn_acc, FALSE, "Shared & No History")
plt_id_hist <- plot_variables(tbl_variables, "ID-History", mn_acc, FALSE, "ID & History")



# avg no history / baseline plot
tbl_plt_gain_base <- subselect_conditions(tbl_variables, c("History", "No History"), c("ID", "Shared")) %>% mutate(Condition = "compare")
plt_delta <- plot_variables(tbl_plt_gain_base, "compare", delta_mn, TRUE, "Difference", max(tbl_plt_gain_base$delta_mn))


grid.draw(arrangeGrob(arrangeGrob(
  plt_conditions, plt_masklength, nrow = 1, widths = c(.3, .7)
),
arrangeGrob(plt_shared_nohist, plt_id_hist, plt_delta, nrow = 1)))



# Moral Machine (MM) ------------------------------------------------------

task_settings = list(
  pth_variability_long = "data/mm-variability-age-culture-indiv.csv",
  pth_variability_means = "data/mm-variability-age-culture-means.csv",
  pth_idseq = "wandb/wandb_export_mm_idseq_noculture_conditions.csv",
  pth_cultureseq = "wandb/wandb_export_mm_cultureseq_noculture_conditions.csv",
  pth_culture_age_full = "wandb/wandb_export_mm_culture_age_full.csv",
  pth_culture_age_small = "wandb/wandb_export_mm_culture_age_small.csv",
  epochthxs = c(201, 250),
  n_epochs = 50
)

tbl_variability_long <- read_csv(task_settings$pth_variability_long)
tbl_variability_means <- read_csv(task_settings$pth_variability_means)

tbl_variability_means$variable <- factor(
  tbl_variability_means$variable, 
  labels = c("Girls", "Boys", "Women", "Men", "Old Women", "Old Men"),
  levels = c("girls_rml", "boys_rml", "women_rml", "men_rml", "oldwomen_rml", "oldmen_rml"),
  ordered = TRUE)
tbl_variability_long$variable <- factor(
  tbl_variability_long$variable, 
  labels = c("Girls", "Boys", "Women", "Men", "Old Women", "Old Men"),
  levels = c("girls_rml", "boys_rml", "women_rml", "men_rml", "oldwomen_rml", "oldmen_rml"),
  ordered = TRUE)
# evidence in favor of cultural differences
# means
pd <- position_dodge(.2)
plt_coef_mean <- ggplot(tbl_variability_means, aes(variable, mean, group = country_cluster)) +
  geom_line(aes(color = country_cluster), position = pd) +
  geom_errorbar(
    aes(ymin = mean - 1.96*sterr, ymax = mean + 1.96*sterr, color = country_cluster),
    width = .2, position = pd) +
  geom_point(color = "white", size = 3, position = pd) +
  geom_point(aes(color = country_cluster), position = pd) +
  geom_hline(yintercept = 0, color = "red", alpha = .3, linetype = "dotdash", linewidth = 1) +
  scale_color_brewer(palette = "Set2") +
  scale_y_continuous(breaks = seq(0, .5, by = .1)) +
  scale_x_discrete() +
  labs(x = element_blank(), y = "Mean Coefficient") +
  theme_bw() +
  theme(
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12),
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.background = element_rect(fill = "white", color = "grey"),
    strip.text = element_text(size = 12),
    legend.title = element_blank(),
    legend.position = "inside",
    legend.position.inside = c(.75, .8),
    legend.background = element_rect(color = "black", fill = "white"),
    legend.box.background = element_rect(color = "black")
  ) +
  coord_cartesian(ylim = c(-.05, .525))

# densities
plt_densities <- tbl_variability_long %>%
  filter(variable %in% c("Girls", "Boys")) %>%
  ggplot(aes(value, color = country_cluster)) +
  geom_density(adjust = 2) +
  facet_grid( ~ variable) +
  scale_color_brewer(palette = "Set2", guide = "none") +
  scale_y_continuous() +
  scale_x_continuous() +
  labs(y = "Coefficient", x = "Density") +
  theme_bw() +
  theme(
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12),
    strip.background = element_rect(fill = "white", color = "grey"),
    strip.text = element_text(size = 12)
  ) + coord_cartesian(xlim = c(-1.5, 2))

plts_pro_diff <- arrangeGrob(plt_coef_mean, plt_densities, nrow = 1, widths = c(.5, .5))

# evidence against cultural differences

tbl_culture_idseq <- prep_tbl_nohistory(task_settings, "pth_idseq")
tbl_culture_idseq$seq_level <- "One Person"
tbl_culture_cultureseq <- prep_tbl_nohistory(task_settings, "pth_cultureseq")
tbl_culture_cultureseq$seq_level <- "Same Culture"

plt_seqlevel <- bind_rows(tbl_culture_idseq, tbl_culture_cultureseq) %>%
  mutate(ID = factor(ID, labels = c("Shuffled", "Proper"))) %>% 
  ggplot(aes(seq_level, mn_acc, group = ID)) +
  geom_hline(yintercept = .5, linetype = "dotdash", linewidth = 1, color = "red", alpha = .3) +
  geom_col(aes(fill = ID), position = position_dodge(.8), width = .75) +
  geom_label(aes(y = mn_acc - .05, label = round(mn_acc, 2)), position = position_dodge(.8)) +
  scale_fill_brewer(palette = "Set2") +
  scale_color_brewer(palette = "Set2", guide = "none") +
  scale_y_continuous() +
  scale_x_discrete() +
  labs(y = "Test Accuracy", x = "Sequence Level", title = "Level 1 Shuffling") +
  theme_bw() +
  theme(
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12),
    strip.background = element_rect(fill = "white", color = "grey"),
    strip.text = element_text(size = 12),
    legend.position = "inside",
    legend.position.inside = c(.5, .75),
    legend.background = element_rect(color = "black", fill = "white"),
    legend.box.background = element_rect(color = "black"),
    legend.title = element_blank()
  ) + coord_cartesian(ylim = c(.5, 1))


tbl_culture_shuffle_long <- prep_tbl_culture(task_settings, "pth_culture_age_full", c(451, 500))
tbl_culture_shuffle_long$ivars <- "All Variables"
tbl_culture_shuffle_short <- prep_tbl_culture(task_settings, "pth_culture_age_small", c(451, 500))
tbl_culture_shuffle_short$ivars <- "Only Age & Culture"

plt_shuffle_age_culture <- bind_rows(tbl_culture_shuffle_long, tbl_culture_shuffle_short) %>%
  ggplot(aes(Culture, mn_acc, group = Age)) +
  geom_hline(yintercept = .5, linetype = "dotdash", linewidth = .75, color = "red", alpha = .2) +
  geom_line(aes(color = Age)) +
  geom_point(color = "white", size = 3) +
  geom_point(aes(color = Age)) +
  facet_wrap(~ ivars) +
  scale_color_brewer(palette = "Set2") +
  scale_y_continuous() +
  scale_x_discrete() +
  labs(y = "Test Accuracy", x = element_blank(), title = "Level 2 Shuffling") +
  theme_bw() +
  theme(
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12),
    strip.background = element_rect(fill = "white", color = "grey"),
    strip.text = element_text(size = 12),
    legend.position = "inside",
    legend.position.inside = c(.75, .7),
    legend.background = element_rect(color = "black", fill = "white"),
    legend.box.background = element_rect(color = "black"),
    legend.title = element_blank()
  ) + coord_cartesian(ylim = c(.5, 1))



plts_con_diff <- arrangeGrob(plt_seqlevel, plt_shuffle_age_culture, nrow = 1)


grid.draw(arrangeGrob(plts_pro_diff, plts_con_diff, nrow = 2, heights = c(.6, .4)))







