rm(list = ls())

library(tidyverse)
library(DT)
library(data.table)
library(grid)
library(gridExtra)
library(htmltools)


# Generate Data -----------------------------------------------------------


v_ids <- seq(1, 2, by = 1)
v_iv1 <- c(1, 2)
v_iv2 <- c(3, 4)
tbl_original <- crossing(SID = v_ids, V1 = v_iv1, V2 = v_iv2) %>%
  group_by(SID) %>%
  mutate(`Trial Nr.` = row_number()) %>%
  relocate(`Trial Nr.`, .after = SID)

# y driven by IVs, learning over trials plus random by-subject intercept
# no measurement noise
set.seed(123)
tbl_original <- tbl_original %>%
  mutate(
    y = 5 + V1*1.5 + V2*0.5 + `Trial Nr.`*.05
  ) %>% group_by(SID) %>%
  mutate(y_add_sid = rnorm(n = 1, sd = 3)) %>%
  ungroup() %>%
  mutate(y = round(y + y_add_sid, 1)) %>% select(-y_add_sid)

# shuffle trials within sid
tbl_id_nohist <- tbl_original %>%
  group_by(SID) %>% slice_sample(prop = 1) %>%
  mutate(`Trial Nr. New` = row_number()) %>%
  ungroup()

# shuffle sids within trial
set.seed(88973)
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

colors <- c(
  RColorBrewer::brewer.pal(nrow(tbl_original)/2, "Reds"),
  RColorBrewer::brewer.pal(nrow(tbl_original)/2, "Purples")
)

# original: color by subject
tbl_original$color <- colors

my_formatting <- function(dt){
  dt  %>%
    formatStyle(
      columns = c("SID", "V1", "V2", "color"),
      fontFamily = "Arial",
      fontSize = "18px"
    ) %>%
    htmlwidgets::prependContent(
      htmltools::tags$style(
        htmltools::HTML("
        table.dataTable thead th {
          font-family: 'Arial';
          font-size: 18px;
          font-weight: 600;
          border: 2px solid #333;   /* header border */
          padding: 6px;             /* optional: makes it look cleaner */
        }
        table.dataTable td {
          font-family: 'Arial';
          font-size: 16px;
        }
        table.dataTable {
          border-collapse: collapse;
        }

        /* Body cells: vertical lines */
        table.dataTable tbody td {
          font-family: 'Arial';
          font-size: 14px;
          padding-top: 6px !important;
          padding-bottom: 6px !important;
          line-height: 14px;
          border-right: 1px solid #aaa;
        }
        
        /* center the first col */
        table.dataTable tbody td:nth-child(1),
        table.dataTable thead th:nth-child(1){
          text-align: center !important;
        }

        
      ")
      )
    )
}

dt2.1 <- datatable(
  tbl_original %>% select(-c("Trial Nr.", "y")) %>% mutate(SID = ""),
  rownames = FALSE,
  colnames = c("ID", "x left", "x right", "color"),
  options = list(
    paging = FALSE,
    scrollY = FALSE,
    autoHeight = TRUE,
    info = FALSE,
    searching = FALSE,
    dom = 't',
    columnDefs = list(
      list(visible = FALSE, targets = ncol(tbl_original)-3)  # hide last column
    ), pageLength = 12,
    rowCallback = JS(
      "function(row, data) {
     $('td', row).css('background-color', data[data.length - 1]);
   }"
    )))%>% my_formatting()


# id nohist
tbl_id_nohist <- tbl_id_nohist %>% left_join(tbl_original %>% select(SID, `Trial Nr.`, color), by = c("SID", "Trial Nr."))
tbl_id_nohist <- tbl_id_nohist %>% 
  select(-`Trial Nr. New`)

dt1.2 <- datatable(
  tbl_id_nohist %>% select(-c("Trial Nr.", "y")) %>% mutate(SID = ""), 
  rownames = FALSE,
  colnames = c("ID", "x left", "x right", "color"),
  options = list(
    paging = FALSE,
    scrollY = FALSE,
    autoHeight = TRUE,
    info = FALSE,
    searching = FALSE,
    dom = 't',
    columnDefs = list(
      list(visible = FALSE, targets = c(ncol(tbl_id_nohist)-3))  # hide last column
    ), pageLength = 12,
    rowCallback = JS(
      "function(row, data) {
     $('td', row).css('background-color', data[data.length - 1]);
   }"
    ))) %>%
  my_formatting()



tbl_original_preserve <- tbl_original
tbl_original <- tbl_original %>% arrange(`Trial Nr.`)
# original: color by trial nr
# colors <- c(
#   RColorBrewer::brewer.pal(nrow(tbl_original), "Reds")[c(3, 7)],
#   RColorBrewer::brewer.pal(nrow(tbl_original), "Purples")[c(3, 7)],
#   RColorBrewer::brewer.pal(nrow(tbl_original), "Greens")[c(3, 7)],
#   RColorBrewer::brewer.pal(nrow(tbl_original), "Greys")[c(3, 7)]
# )
# tbl_original$color <- colors

# shared hist
tbl_shared_hist <- tbl_shared_hist %>% left_join(tbl_original %>% select(SID, `Trial Nr.`, color), by = c("SID", "Trial Nr."))
tbl_shared_hist <- tbl_shared_hist %>% select(-`SID New`)# %>% arrange(`Trial Nr.`)

dt2.2 <- datatable(
  tbl_shared_hist %>% select(-c("Trial Nr.", "y")) %>% mutate(SID = ""), 
  rownames = FALSE,
  colnames = c("ID", "x left", "x right", "color"), 
  options = list(
    paging = FALSE,
    scrollY = FALSE,
    autoHeight = TRUE,
    info = FALSE,
    searching = FALSE,
    dom = 't',
    columnDefs = list(
      list(visible = FALSE, targets = c(ncol(tbl_shared_hist)-3))  # hide last column
    ), pageLength = 12,
    rowCallback = JS(
      "function(row, data) {
     $('td', row).css('background-color', data[data.length - 1]);
   }"
    ))) %>% my_formatting()


# shared no hist
tbl_shared_nohist <- tbl_shared_nohist %>% left_join(tbl_original_preserve %>% select(SID, `Trial Nr.`, color), by = c("SID", "Trial Nr."))
tbl_shared_nohist <- tbl_shared_nohist %>% select(-c("SID New", "Trial Nr. New"))
dt3.2 <- datatable(
  tbl_shared_nohist  %>% select(-c("Trial Nr.", "y")) %>% mutate(SID = ""),
  rownames = FALSE,
  colnames = c("ID", "x left", "x right", "color"),
  options = list(
    paging = FALSE,
    scrollY = FALSE,
    autoHeight = TRUE,
    info = FALSE,
    searching = FALSE,
    dom = 't',
    columnDefs = list(
      list(visible = FALSE, targets = c(ncol(tbl_shared_nohist)-3))  # hide last column
    ), pageLength = 12,
    rowCallback = JS(
      "function(row, data) {
     $('td', row).css('background-color', data[data.length - 1]);
   }"
    ))) %>% my_formatting()



htmltools::browsable(
  tags$div(
    style = "
      display: grid;
      grid-template-columns: 1fr 1fr;
      grid-template-rows: auto auto;
      column-gap: 10px;
      row-gap: 85px;
      align-items: start;
    ",
    
    tags$div(
      style = "grid-column: 1; grid-row: 1; align-self: center; height:260px; width:300px",
      dt2.1
    ),
    
    tags$div(
      style = "grid-column: 1; grid-row: 2; height:260px; width:300px",
      dt1.2
    ),

    tags$div(
      style = "grid-column: 2; grid-row: 1; height:260px; width:300px",
      dt2.2
    ),
    
    tags$div(
      style = "
        grid-column: 2;
        grid-row: 2;
        align-self: center;
        height:260px; width:300px",
      dt3.2
    )
  )
)



# Level 2 Shuffling -------------------------------------------------------
tbl_original <- tbl_original %>%
  arrange(SID, "Trial Nr.")
tbl_original$color[1:4] <- tbl_original$color[5:8]

set.seed(873)
tbl_original_l2 <- tbl_original %>%
  select(-color) %>%
  mutate(is_swap = sample(rep(c(TRUE, FALSE), nrow(.)/2), size = nrow(.), replace = FALSE)) %>%
  rowwise() %>% mutate(V1_new = ifelse(is_swap, V2, V1), V2 = ifelse(is_swap, V1, V2), V1 = V1_new) %>% select(-V1_new)



# tbl unshuffled between cols as base



dt.l2.1 <- datatable(
  tbl_original %>% select(-c("Trial Nr.", "y")) %>% mutate(SID = ""),
  colnames = c("ID", "x left", "x right", "color"),
  rownames = FALSE,
  options = list(
    paging = FALSE,
    scrollY = FALSE,
    autoHeight = TRUE,
    info = FALSE,
    searching = FALSE,
    dom = 't',
    columnDefs = list(
      list(visible = FALSE, targets = c(ncol(tbl_original)-3))  # hide last column
    ), pageLength = 12,
    rowCallback = JS(
      "function(row, data) {
     $('td', row).css('background-color', data[data.length - 1]);
   }"
    )
  )) %>%
  my_formatting()


# then tbl shuffled between cols

tbl_original_l2$color <- tbl_original$color
tbl_original_l2$color[tbl_original_l2$is_swap] <- colors[2]
tbl_original_l2$color2 <- tbl_original$color

dt.l2.2 <- datatable(tbl_original_l2 %>% select(-c("Trial Nr.", "y")) %>% mutate(SID = ""),
          rownames = FALSE,
          colnames = c("ID", "x left", "x right", "is_swap", "color", "color2"),
          options = list(
            paging = FALSE,
            scrollY = FALSE,
            autoHeight = TRUE,
            info = FALSE,
            searching = FALSE,
            dom = 't',
            columnDefs = list(
              list(visible = FALSE, targets = c((ncol(tbl_original_l2)-5):(ncol(tbl_original_l2))))  # hide last column
            ), pageLength = 12,
            rowCallback = JS(
              "function(row, data) {
         var color = data[data.length - 2];   // last column = color
         var color2 = data[data.length - 1];

         // color only column 3 (index 2)
         $('td:eq(1)', row).css('background-color', color);

         // color only column 5 (index 4)
         $('td:eq(2)', row).css('background-color', color);
         
         // color only column 3 (index 2)
         $('td:eq(0)', row).css('background-color', color2);
         
       }"
              )
            )) %>%
  my_formatting()



htmltools::browsable(
  tags$div(
    style = "
      display: grid;
      grid-template-columns: 1fr 1fr;
      grid-template-rows: auto;
      column-gap: 10px;
      align-items: start;
    ",
    
    tags$div(
      style = "grid-column: 1; grid-row: 1; height: 250px; width:300px",
      dt.l2.1
    ),
    
    tags$div(
      style = "grid-column: 2; grid-row: 1; height: 250px; width:300px",
      dt.l2.2
    )
  )
)


# Level 1 Example Effects -------------------------------------------------

pd <- position_dodge(width = .1)
plot_example_conditions <- function(my_tbl, ttl, plot_legend = FALSE) {
  idx_legend <- 1
  if (plot_legend) idx_legend <- 2
  library(stringr)
  
  ggplot(my_tbl, aes(History, y, group = ID)) +
    geom_hline(yintercept = .5, linetype = "dotdash", linewidth = 1, color = "red", alpha = .3) +
    geom_line((aes(color = ID)), position = pd) +
    geom_point(color = "white", size = 5, position = pd) +
    geom_point(aes(color = ID), size = 3, position = pd) +
    scale_color_brewer(palette = "Set1") +
    scale_y_continuous(expand = expansion(add = c(.02, 0))) +
    labs(y = "Accuracy", x = element_blank(), title = ttl) +
    scale_x_discrete(
      labels = function(x) str_replace(x, "^(.{8})(.*)$", "\\1\n\\2")
    ) +
    theme_classic() +
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


v_history <- factor(levels = c("Shuffled History", "Original History"), ordered = TRUE)
v_ID <- factor(levels = c("Shuffled ID", "Original ID"), ordered = TRUE)
tbl_main_id <- crossing(History = v_history, ID = v_ID) %>%
  mutate(y = c(.7, .8, .7, .8))
tbl_idio_seq <- crossing(History = v_history, ID = v_ID) %>%
  mutate(y = c(.7, .7, .7, .8))

plt_id_only <- plot_example_conditions(tbl_main_id, "Individual Differences")
plt_idio_seq <- plot_example_conditions(tbl_idio_seq, "Idiosyncratic Seq. Effect", plot_legend = TRUE)

grid.draw(arrangeGrob(plt_id_only, plt_idio_seq, nrow = 1))
