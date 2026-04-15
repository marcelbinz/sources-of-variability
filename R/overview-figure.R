rm(list = ls())

library(tidyverse)
library(DT)
library(data.table)
library(grid)
library(gridExtra)


# Generate Data -----------------------------------------------------------


v_ids <- seq(1, 2, by = 1)
v_iv1 <- c(-.5, .5)
v_iv2 <- v_iv1
tbl_original <- crossing(SID = v_ids, V1 = v_iv1, V2 = v_iv2) %>%
  group_by(SID) %>%
  mutate(`Trial Nr.` = row_number()) %>%
  relocate(`Trial Nr.`, .after = SID)

# y driven by IVs, learning over trials plus random by-subject intercept
# no measurement noise
set.seed(1234)
tbl_original <- tbl_original %>%
  mutate(
    y = 5 + V1*1.5 + V2*0.5 + `Trial Nr.`*.05
  ) %>% group_by(SID) %>%
  mutate(y_add_sid = rnorm(n = 1, sd = 3)) %>%
  ungroup() %>%
  mutate(y = round(y + y_add_sid, 2)) %>% select(-y_add_sid)

# shuffle trials within sid
tbl_id_nohist <- tbl_original %>%
  group_by(SID) %>% slice_sample(prop = 1) %>%
  mutate(`Trial Nr. New` = row_number()) %>%
  ungroup()

# shuffle sids within trial
tbl_shared_hist <- tbl_original %>%
  group_by(`Trial Nr.`) %>% slice_sample(prop = 1) %>%
  mutate(`SID New` = row_number()) %>%
  ungroup() %>% arrange(`SID New`, `Trial Nr.`)

tbl_shared_nohist <- tbl_original %>%
  group_by(SID) %>%
  slice_sample(prop = 1) %>%
  mutate(
    `Trial Nr. New` = row_number()
  ) %>%
  group_by(`Trial Nr. New`) %>%
  slice_sample(prop = 1) %>%
  mutate(`SID New` = row_number()) %>%
  arrange(`SID New`, `Trial Nr. New`) %>%
  relocate(`SID New`, .before = SID) %>%
  relocate(`Trial Nr. New`, .before = `Trial Nr.`) %>%
  ungroup()



# Level 1 Shuffling -------------------------------------------------------



# use original twice
# once colored trial nr -> then show id no hist (id stays)
# once colored sid -> then show shared hist (trial stays)
# once everything shuffled, no coloring
# rename Trial Nr. Old -> Trial Nr. New vs. Trial Nr.

colors <- RColorBrewer::brewer.pal(n = length(unique(tbl_original$SID)), "Set2")

# original: color by subject
tbl_original$color <- colors[as.numeric(tbl_original$SID)]

datatable(tbl_original, options = list(
  columnDefs = list(
    list(visible = FALSE, targets = ncol(tbl_original))  # hide last column
  ), pageLength = 12,
  rowCallback = JS(
  "function(row, data) {
     $('td', row).css('background-color', data[data.length - 1]);
   }"
))) %>%
  formatStyle(
    columns = c("Trial Nr."),
    border = "2px solid blue"
  )

# original: color by trial nr
colors <- RColorBrewer::brewer.pal(n = length(unique(tbl_original$`Trial Nr.`)), "Set2")

tbl_original <- tbl_original %>% arrange(`Trial Nr.`)
tbl_original$color <- colors[as.numeric(tbl_original$`Trial Nr.`)]

datatable(tbl_original, options = list(
  columnDefs = list(
    list(visible = FALSE, targets = ncol(tbl_original))  # hide last column
  ), pageLength = 12,
  rowCallback = JS(
    "function(row, data) {
     $('td', row).css('background-color', data[data.length - 1]);
   }"
  ))) %>%
  formatStyle(
    columns = c("SID"),
    border = "2px solid blue"
  )


# id nohist
colors <- RColorBrewer::brewer.pal(n = length(unique(tbl_id_nohist$SID)), "Set2")
tbl_id_nohist$color <- colors[as.numeric(tbl_id_nohist$SID)]
tbl_id_nohist <- tbl_id_nohist %>% 
  relocate(`Trial Nr. New`, .after = SID)

datatable(tbl_id_nohist, options = list(
  columnDefs = list(
    list(visible = FALSE, targets = c(ncol(tbl_id_nohist)))  # hide last column
  ), pageLength = 12,
  rowCallback = JS(
    "function(row, data) {
     $('td', row).css('background-color', data[data.length - 1]);
   }"
  ))) %>%
  formatStyle(
    columns = c("Trial Nr. New", "Trial Nr."),
    border = "2px solid blue"
  )


# shared hist
colors <- RColorBrewer::brewer.pal(n = length(unique(tbl_shared_hist$`Trial Nr.`)), "Set2")
tbl_shared_hist$color <- colors[as.numeric(tbl_shared_hist$`Trial Nr.`)]
tbl_shared_hist <- tbl_shared_hist %>% relocate(`SID New`, .before = `SID`) %>% arrange(`Trial Nr.`)

datatable(tbl_shared_hist, options = list(
  columnDefs = list(
    list(visible = FALSE, targets = c(ncol(tbl_shared_hist)))  # hide last column
  ), pageLength = 12,
  rowCallback = JS(
    "function(row, data) {
     $('td', row).css('background-color', data[data.length - 1]);
   }"
  ))) %>%
  formatStyle(
    columns = c("SID New", "SID"),
    border = "2px solid blue"
  )


# shared no hist
colors <- RColorBrewer::brewer.pal(n = length(unique(tbl_shared_nohist$`SID`)), "Set2")
tbl_shared_nohist$color <- colors[as.numeric(tbl_shared_nohist$`SID`)]

datatable(tbl_shared_nohist, options = list(
  columnDefs = list(
    list(visible = FALSE, targets = c(ncol(tbl_shared_nohist)))  # hide last column
  ), pageLength = 12,
  rowCallback = JS(
    "function(row, data) {
     $('td', row).css('background-color', data[data.length - 1]);
   }"
  )))



# Level 2 Shuffling -------------------------------------------------------

tbl_shared_nohist_l2 <- tbl_shared_nohist %>%
  select(-color) %>%
  mutate(is_swap = sample(rep(c(TRUE, FALSE), nrow(.)/2), size = nrow(.), replace = FALSE)) %>%
  rowwise() %>% mutate(V1_new = ifelse(is_swap, V2, V1), V2 = ifelse(is_swap, V1, V2), V1 = V1_new) %>% select(-V1_new)



# tbl unshuffled between cols as base
colors <- RColorBrewer::brewer.pal(n = 1, "Set2")
tbl_shared_nohist$color <- colors[1]

datatable(tbl_shared_nohist, options = list(
  columnDefs = list(
    list(visible = FALSE, targets = c(ncol(tbl_shared_nohist)))  # hide last column
  ), pageLength = 12,
  rowCallback = JS(
    "function(row, data) {
     $('td', row).css('background-color', data[data.length - 1]);
   }"
  ))) %>%
  formatStyle(
    columns = c("V1", "V2"),
    border = "2px solid blue"
  )


# then tbl shuffled between cols

colors <- RColorBrewer::brewer.pal(n = length(unique(tbl_shared_nohist_l2$is_swap)), "Set2")
tbl_shared_nohist_l2$color <- colors[as.numeric(tbl_shared_nohist_l2$is_swap) + 1]

datatable(tbl_shared_nohist_l2, options = list(
  columnDefs = list(
    list(visible = FALSE, targets = c((ncol(tbl_shared_nohist_l2)-1):ncol(tbl_shared_nohist_l2)))  # hide last column
  ), pageLength = 12,
  rowCallback = JS(
    "function(row, data) {
     $('td', row).css('background-color', data[data.length - 1]);
   }"
  ))) %>%
  formatStyle(
    columns = c("V1", "V2"),
    border = "2px solid blue"
  )



# Level 1 Example Effects -------------------------------------------------

pd <- position_dodge(width = .1)
plot_example_conditions <- function(my_tbl, ttl, plot_legend = FALSE) {
  idx_legend <- 1
  if (plot_legend) idx_legend <- 2
  ggplot(my_tbl, aes(History, y, group = ID)) +
  geom_hline(yintercept = .5, linetype = "dotdash", linewidth = 1, color = "red", alpha = .3) +
  geom_line((aes(color = ID)), position = pd) +
  geom_point(color = "white", size = 5, position = pd) +
  geom_point(aes(color = ID), size = 3, position = pd) +
  scale_color_brewer(palette = "Set1") +
  scale_y_continuous(expand = expansion(add = c(.02, 0))) +
  labs(y = "Test Accuracy", x = element_blank(), title = ttl) +
  theme_bw() +
  theme(
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 14),
    legend.text = element_text(size = 14),
    legend.title = element_blank(),
    axis.title.x = element_blank(),
    legend.position = c("omit", "inside")[idx_legend],
    legend.position.inside = c(.3, .8),
    title = element_text(size = 14)
  ) +
  coord_cartesian(ylim = c(.5, 1))
}


v_history <- factor(levels = c("No History", "History"), ordered = TRUE)
v_ID <- factor(levels = c("Shared", "ID"), ordered = TRUE)
tbl_main_id <- crossing(History = v_history, ID = v_ID) %>%
  mutate(y = c(.7, .8, .7, .8))
tbl_main_seq <- crossing(History = v_history, ID = v_ID) %>%
  mutate(y = c(.7, .7, .8, .8))
tbl_idio_seq <- crossing(History = v_history, ID = v_ID) %>%
  mutate(y = c(.7, .7, .7, .8))

plt_seq_only <- plot_example_conditions(tbl_main_seq, "Only Sequence Effect", plot_legend = TRUE)
plt_id_only <- plot_example_conditions(tbl_main_id, "Only Ind. Differences")
plt_idio_seq <- plot_example_conditions(tbl_idio_seq, "Sequence Effect By ID")

grid.draw(arrangeGrob(plt_seq_only, plt_id_only, plt_idio_seq, nrow = 1))


