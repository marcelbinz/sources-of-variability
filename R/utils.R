kalman_learning <- function(tbl_df, no, sigma_epsilon_sq, m0) {
  #' Kalman filter without choice model given chosen options by participants
  #' 
  #' @description applies Kalman filter equations for a given bandit task with existing choices by participants
  #' @param tbl_df with made choices and collected rewards as columns
  #' @param no number of response options
  #' @param sigma_epsilon_sq error variance (prior variance)
  #' @return a tbl with by-trial posterior means and variances for all bandits
  rewards <- tbl_df$reward
  choices <- tbl_df$chosen
  v0 <- 100
  nt <- length(rewards) # number of time points
  m <- matrix(m0, ncol = no, nrow = nt + 1) # to hold the posterior means
  v <- matrix(v0, ncol = no, nrow = nt + 1) # to hold the posterior variances
  sigma_xi_sq <- rep(tbl_df$sigma_xi_squared[1], no)
  
  for(t in 1:nt) {
    kt <- rep(0, no)
    # set the Kalman gain for the chosen option
    # sigma xi differs between options so need to index it
    kt[choices[t]+1] <- (v[t,choices[t]+1] + sigma_xi_sq[choices[t]+1])/(v[t,choices[t]+1] + sigma_epsilon_sq + sigma_xi_sq[choices[t]+1])
    # compute the posterior means
    m[t+1,] <- m[t,] + kt*(rewards[t] - m[t,])
    # compute the posterior variances
    v[t+1,] <- (1-kt)*(v[t,]) + sigma_xi_sq
  }
  tbl_m <- as.data.frame(m)
  # constrain v from becoming to small
  v <- t(apply(v, 1, function(x) pmax(x, .0001)))
  tbl_v <- as.data.frame(v)
  colnames(tbl_m) <- paste("m_", 1:no, sep = "")
  colnames(tbl_v) <- paste("v_", 1:no, sep = "")
  tbl_return <- tibble(cbind(tbl_m, tbl_v))
  
  
  tbl_return <- tbl_return[1:(nrow(tbl_return)-1), ]
  tbl_return$ID <- tbl_df$ID
  tbl_return$block <- tbl_df$block
  tbl_return$right_picked <- tbl_df$chosen
  
  return(tbl_return)
}



plot_four_conditions <- function(pth, ttl = NULL, task_settings){
  "read results from four conditions and plot"
  tbl_accuracy <- read_csv(pth)
  tbl_accuracy_long <- tbl_accuracy %>%
    select(epoch, ends_with("dev/acc_epoch") | ends_with("dev/loss_epoch")) %>%
    pivot_longer(-epoch) %>%
    mutate(
      Condition = str_match(name, "condition=([a-z_]*)_[md]")[,2],
      shuffle = str_match(name, "more_shuffle=([a-z_/-]*)_tf")[,2],
      metric = factor(str_detect(name, "loss"), labels = c("Accuracy", "Loss"))
    ) %>% select(-name) %>%
    pivot_wider(values_from = "value", names_from = "metric")
  
  
  # select epochs to be analyzed
  tbl_conditions <- tbl_accuracy_long %>%
    filter(between(epoch, task_settings$epochthxs[1], task_settings$epochthxs[2])) %>%
    group_by(Condition, shuffle) %>%
    summarize(
      mn_acc = mean(Accuracy), sd_acc = sd(Accuracy)/sqrt(task_settings$n_epochs),
      mn_loss = mean(Loss), sd_loss = sd(Loss)/sqrt(task_settings$n_epochs), .groups = "drop")
  
  tbl_conditions$Condition <- factor(
    tbl_conditions$Condition,
    levels = c("shared_nohist", "shared_hist", "id_nohist", "original"),
    labels = c("Shared-NoHistory", "Shared-History", "ID-NoHistory", "ID-History")
  )
  
  # plot and save
  
  tbl_conditions <- tbl_conditions %>% mutate(
    History = as.numeric(!str_detect(Condition, "NoHistory")),
    ID = as.numeric(str_detect(Condition, "ID")),
    )
  tbl_conditions <- tbl_conditions %>% 
    mutate(
    History = factor(History, labels = c("Shuffled\nHistory", "Original\nHistory")[1:length(unique(.$History))], ordered = TRUE),
    ID = factor(ID, labels = c("Shuffled ID", "Original ID")[1:length(unique(.$ID))], ordered = TRUE)
  )
  
  
  pl_conditions <- ggplot(tbl_conditions, aes(History, mn_acc, group = ID)) +
    geom_hline(yintercept = .5, linetype = "dotdash", linewidth = 1, color = "red", alpha = .3) +
    geom_errorbar(aes(
      History, ymin = mn_acc - 2*sd_acc, ymax = mn_acc + 2*sd_acc, color = ID
    ), width = .2, linewidth = .75) +
    geom_line((aes(color = ID))) +
    geom_point(color = "white", size = 5) +
    geom_point(aes(color = ID), size = 3) +
    scale_color_brewer(palette = "Set1") +
    scale_y_continuous(expand = expansion(add = c(.02, 0))) +
    labs(y = "Test Accuracy") +
    theme_bw() +
    theme(
      axis.title = element_text(size = 16),
      axis.text = element_text(size = 14),
      legend.text = element_text(size = 14),
      legend.title = element_blank(),
      axis.title.x = element_blank(),
      legend.position = "inside",
      legend.position.inside = task_settings$pl_conditions_labelpos
    ) +
    coord_cartesian(ylim = c(.5, 1))
  
  if (!is.null(ttl)) pl_conditions <- pl_conditions + labs(title = ttl)
  
  return(list(pl_conditions = pl_conditions, tbl_conditions = tbl_conditions))
}

prep_tbl_variables <- function(task_settings) {
  "load and subset variables analysis results"
  tbl_accuracy <- read_csv(task_settings$pth_variables)
  tbl_accuracy_long <- tbl_accuracy %>%
    select(epoch, contains("dev/acc_epoch") | contains("dev/loss_epoch")) %>%
    pivot_longer(-epoch) %>%
    mutate(
      Condition = str_match(name, "condition=([a-z_]*)_[md]")[,2],
      shuffle = str_match(name, "more_shuffle=([A-Za-z_/-]*)_tf")[,2],
      metric = factor(str_detect(name, "loss"), labels = c("Accuracy", "Loss"))
    ) %>% select(-name) %>%
    pivot_wider(values_from = "value", names_from = "metric")
  
  tbl_accuracy_long$Condition <- factor(
    tbl_accuracy_long$Condition,
    levels = c("shared_nohist", "shared_hist", "id_nohist", "original"),
    labels = c("Shared-NoHistory", "Shared-History", "ID-NoHistory", "ID-History")
  )
  
  tbl_accuracy_long$available <- tbl_accuracy_long$shuffle
  tbl_accuracy_long$available[tbl_accuracy_long$available == "time-val"] <- "val-time"
  tbl_accuracy_long$available[tbl_accuracy_long$available == "time-val-picked_prev"] <- "val-time-picked_prev"
  tbl_accuracy_long$available[tbl_accuracy_long$available == "prob-val-picked_prev"] <- "val-prob-picked_prev"
  tbl_accuracy_long$available[tbl_accuracy_long$available == "prob-val"] <- "val-prob"
  tbl_accuracy_long$available[tbl_accuracy_long$available == "v-m"] <- "m-v"
  tbl_accuracy_long$available[tbl_accuracy_long$available == "v-m-picked_prev"] <- "m-v-picked_prev"
  
  
  if (task_settings$task == "mm") {
    tbl_accuracy_long$available[
      str_detect(tbl_accuracy_long$available, "^(?=.*Eastern)(?=.*woman)(?=.*picked_prev).*")
    ] <- "Nothing"
    tbl_accuracy_long$available[
      str_detect(tbl_accuracy_long$available, "^(?=.*Eastern)(?=.*woman)(?!.*picked_prev).*")
    ] <- "R"
    tbl_accuracy_long$available[
      str_detect(tbl_accuracy_long$available, "^(?=.*Eastern)(?!.*woman)(?=.*picked_prev).*")
    ] <- "A"
    tbl_accuracy_long$available[
      str_detect(tbl_accuracy_long$available, "^(?!.*Eastern)(?=.*woman)(?=.*picked_prev).*")
    ] <- "C"
    tbl_accuracy_long$available[
      str_detect(tbl_accuracy_long$available, "^(?=.*Eastern)(?!.*woman)(?!.*picked_prev).*")
    ] <- "A&R"
    tbl_accuracy_long$available[
      str_detect(tbl_accuracy_long$available, "^(?!.*Eastern)(?=.*woman)(?!.*picked_prev).*")
    ] <- "C&R"
    tbl_accuracy_long$available[
      str_detect(tbl_accuracy_long$available, "^(?!.*Eastern)(?!.*woman)(?=.*picked_prev).*")
    ] <- "A&C"
    tbl_accuracy_long$available[tbl_accuracy_long$available == "nothing"] <- "A&C&R"
  }
  
  tbl_accuracy_long$available <- factor(
    tbl_accuracy_long$available, labels = task_settings$indep_vars_labels_incoming
  )
  tbl_accuracy_long$available <- fct_relevel(tbl_accuracy_long$available, task_settings$indep_vars_labels_ordered)
  
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
    summarize(
      mn_acc = mean(Accuracy), 
      se_acc = sd(Accuracy)/sqrt(task_settings$n_epochs), 
      mn_loss = mean(Loss), 
      se_loss = sd(Loss)/sqrt(task_settings$n_epochs), 
      .groups = "drop")
  
  return(tbl_variables)
}

subselect_conditions <- function(tbl_plt, h, i) {
  "compare two conditions and calculate delta"
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


prep_tbl_nohistory <- function(task_settings, pth_select) {
  "load test accuracy for only no-history conditions"
  tbl_culture_idseq <- read_csv(task_settings[[pth_select]])
  tbl_idseq_long <- tbl_culture_idseq %>%
    select(epoch, ends_with("dev/acc_epoch")) %>%
    pivot_longer(-epoch) %>%
    mutate(
      Condition = str_match(name, "condition=([a-z_]*)_[md]")[,2],
      shuffle = str_match(name, "moreshuffle=([a-z_/-]*)_tf")[,2]
    )
  
  # select epochs to be analyzed
  tbl_conditions <- tbl_idseq_long %>%
    filter(between(epoch, task_settings$epochthxs[1], task_settings$epochthxs[2])) %>%
    group_by(Condition, shuffle) %>%
    summarize(mn_acc = mean(value), sd = sd(value)/sqrt(task_settings$n_epochs), .groups = "drop")
  
  tbl_conditions$Condition <- factor(
    tbl_conditions$Condition,
    levels = c("id_nohist", "shared_nohist"),
    labels = c("ID-NoHistory", "Shared-NoHistory")
  )
  
  tbl_conditions <- tbl_conditions %>% mutate(
    History = as.numeric(!str_detect(Condition, "NoHistory")),
    History = factor(History, labels = c("No History"), ordered = TRUE),
    ID = as.numeric(str_detect(Condition, "ID")),
    ID = factor(ID, labels = c("Shared", "ID"), ordered = TRUE)
  )
  
  return(tbl_conditions)
}


prep_tbl_culture <- function(task_settings, pth_select, thxs_epochs) {
  "load and prep culture level 2 shuffling data"
  
  tbl_culture_age <- read_csv(task_settings[[pth_select]])
  
  tbl_culture_agg <- tbl_culture_age %>%
    select(epoch, ends_with("dev/acc_epoch")) %>%
    pivot_longer(-epoch) %>%
    mutate(
      Condition = str_match(name, "condition=([a-z_]*)_[md]")[,2],
      shuffle = str_match(name, "more_shuffle=([A-Za-z_/-]*)_tf")[,2],
      available = shuffle#factor(shuffle, labels = c("Nothing", "Age", "Culture", "Culture&Age"))
    )
  
  tbl_culture_agg$available[tbl_culture_agg$available == "nothing"] <- "Culture&Age"
  tbl_culture_agg$available[tbl_culture_agg$available == "Eastern"] <- "Age"
  tbl_culture_agg$available[str_detect(tbl_culture_agg$available, "Eastern")] <- "Nothing"
  tbl_culture_agg$available[str_detect(tbl_culture_agg$available, "old")] <- "Culture"
  
  tbl_culture_agg <- tbl_culture_agg %>%
    mutate(
      Age = factor(str_detect(available, "Age"), labels = c("No Age", "Age")),
      Culture = factor(str_detect(available, "Culture"), labels = c("No Culture", "Culture"))
    )
  
  # select epochs to be analyzed
  tbl_conditions <- tbl_culture_agg %>%
    filter(between(epoch, thxs_epochs[1], thxs_epochs[2])) %>%
    group_by(Age, Culture) %>%
    summarize(mn_acc = mean(value), sd = sd(value)/sqrt(task_settings$n_epochs), .groups = "drop")
  
  return(tbl_conditions)
  
}


plot_variables <- function(tbl_variables, cd, ivar, is_delta, ttl, max_y_delta = 0, min_y_delta = 0) {
  plt <- tbl_variables %>% filter(Condition == cd) %>%
    ggplot(aes(available, {{ivar}}, group = available))
  
  if(!is_delta) {plt <- plt  + geom_hline(yintercept = .5, color = "red", linetype = "dotdash", alpha = .7, linewidth = 1)}
  plt <- plt + geom_col(aes(fill = available), width = .75) +
    #geom_errorbar(aes(ymin = {{ivar}} - 1.96 * se, ymax = {{ivar}} + 1.96 * se)) + # really invisible...
    scale_fill_brewer(palette = "Set2", guide = "none") +
    scale_color_brewer(palette = "Set2", guide = "none") +
    scale_x_discrete() +
    labs(y = "Test Accuracy", x = "Available Variables", title = ttl) +
    theme_bw() +
    theme(
      axis.title = element_text(size = 16),
      axis.text = element_text(size = 14),
      strip.background = element_rect(fill = "white", color = "grey"),
      strip.text = element_text(size = 14)
    )
  if (is_delta) {
    plt <- plt + coord_cartesian(ylim = c(min_y_delta, max_y_delta)) +
      scale_y_continuous(expand = expansion(mult = 0, add = c(.01, .01))) +
      labs(y = "ID - Shared")
    # + labs(caption = "Note. T:Time, V:Value, R:Prev.Response")
  } else {
    plt <-  plt + coord_cartesian(ylim = c(.475, 1)) +
      scale_y_continuous(breaks = seq(-1, 1, by = .1), expand = expansion(mult = 0, add = c(.02, 0)))
  }
  return(plt)
}

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


prep_tbls_masklength <- function(task_settings) {
  
  tbl_constrain <- read_csv(task_settings$pth_masklength)
  tbl_constrain <- tbl_constrain %>% select(
    c("epoch",
      (contains("original") | contains("id_nohist")) &
        (contains("dev/acc_epoch") | contains("dev/loss_epoch")) & !contains("MIN") & !contains("MAX")
    ))
  
  # note. in mm masklength == 10 overfit to the train data between 201 and 250
  # therefore, use 160-210
  if (task == "mm") task_settings$epochthxs <- c(161, 210)
  
  tbl_constrain_long <- tbl_constrain %>% pivot_longer(-"epoch") %>%
    filter(between(epoch, task_settings$epochthxs[1], task_settings$epochthxs[2])) %>%
    mutate(
      masklength = str_match(name, "windowsize=([0-9]+)")[,2],
      metric = factor(str_detect(name, "loss"), labels = c("Accuracy", "Loss")),
      is_mask = str_detect(name, "windowed_causal"),
      is_history = str_detect(name, "original")
    ) %>% select(-name) %>%
    mutate(masklength = ifelse(is_mask, masklength, ifelse(is_history, 9999, 10000))) %>%
    pivot_wider(names_from = "metric", values_from = "value")
  tbl_constrain_long$masklength <- as.numeric(tbl_constrain_long$masklength)
  tbl_constrain_long$masklength <- factor(tbl_constrain_long$masklength)
  tbl_constrain_long$masklength <- factor(
    tbl_constrain_long$masklength, 
    labels = c(head(levels(tbl_constrain_long$masklength), -2), "All", "No Hist")
  )
  
  
  tbl_plt_masklength <- tbl_constrain_long %>%
    group_by(masklength) %>%
    summarize(accuracy = mean(Accuracy, na.rm = TRUE), .groups = "drop")
  
  tbl_masklength_loss <- tbl_constrain_long %>%
    group_by(masklength) %>%
    summarize(mn_acc = mean(Accuracy, na.rm = TRUE), mn_loss = mean(Loss, na.rm = TRUE), .groups = "drop")
  
  return(list(tbl_plt_masklength = tbl_plt_masklength, tbl_masklength_loss = tbl_masklength_loss))
}


