library("tidyverse")
library("afex")
library("emmeans")
theme_set(theme_bw(base_size = 15) + theme(legend.position = "bottom"))

# -----------------------------
# Load + prepare data
# -----------------------------
truth_ratings <- read_csv("ideal_counterbalanced.csv") %>%
  mutate(
    participant_id = factor(participant_id),
    repetition_numeric = as.numeric(repetition_times),
    repetition_factor  = factor(repetition_times, levels = c(0, 1, 9, 18, 27)),
    truth_score = as.integer(truth_score)
  )

glimpse(truth_ratings)

# -----------------------------
# 1) Descriptive stats (means/SD/SE/CI) by repetition condition
# -----------------------------
desc_by_cond <- truth_ratings %>%
  group_by(repetition_factor) %>%
  summarise(
    n = n(),
    mean = mean(truth_score, na.rm = TRUE),
    sd = sd(truth_score, na.rm = TRUE),
    se = sd / sqrt(n),
    ci95 = qt(0.975, df = n - 1) * se,
    .groups = "drop"
  )

desc_by_cond

# (Optional) participant-level means per condition (useful sanity check)
desc_by_participant <- truth_ratings %>%
  group_by(participant_id, repetition_factor) %>%
  summarise(mean_truth = mean(truth_score), .groups = "drop")

desc_by_participant

# -----------------------------
# 2) Plot: mean truth rating vs repetition condition (with 95% CI)
# -----------------------------
ggplot(desc_by_cond, aes(x = repetition_factor, y = mean, group = 1)) +
  geom_point(size = 2) +
  geom_line() +
  geom_errorbar(aes(ymin = mean - ci95, ymax = mean + ci95), width = 0.15) +
  labs(
    x = "Repetition times",
    y = "Mean truth rating",
    title = "Truth ratings by repetition condition"
  )

# -----------------------------
# 3) Repeated-measures ANOVA (within-subject factor: repetition_factor)
# -----------------------------
afex::afex_options(type = 3)

aov_rm <- aov_car(
  truth_score ~ repetition_factor + Error(participant_id/repetition_factor),
  data = truth_ratings,
  factorize = FALSE
)

aov_rm

# -----------------------------
# 4) Linear vs logarithmic fit (paper-style: per-participant correlations + paired t-test)
# -----------------------------
corrs <- truth_ratings %>%
  group_by(participant_id) %>%
  summarise(
    r_linear = cor(truth_score, repetition_numeric),
    r_log    = cor(truth_score, log(repetition_numeric + 1)),
    .groups = "drop"
  )

corrs

# Paired t-test: is log correlation > linear correlation?
tt <- t.test(corrs$r_log, corrs$r_linear, paired = TRUE)
tt

# Paired-samples Cohen's d on the difference in correlations
diff_r <- corrs$r_log - corrs$r_linear
d_paired <- mean(diff_r) / sd(diff_r)
d_paired

# Summary (mean/SD of correlations)
corrs %>%
  summarise(
    mean_r_linear = mean(r_linear), sd_r_linear = sd(r_linear),
    mean_r_log    = mean(r_log),    sd_r_log    = sd(r_log)
  )

# -----------------------------
# 5) Pairwise comparisons between repetition conditions
# -----------------------------

# Estimated marginal means
emm_rep <- emmeans(aov_rm, ~ repetition_factor)

# All pairwise comparisons (within-subject)
pairs_rep <- pairs(emm_rep, adjust = "none")  # match paper style (often unadjusted)
pairs_rep

# -----------------------------
# 6) Model comparison: Linear vs Log model (this is an extra analysis not in the paper. R sq log > R sq linear)
# -----------------------------

# Fit linear model
model_linear <- lm(
  truth_score ~ repetition_numeric,
  data = truth_ratings
)

# Fit log model
model_log <- lm(
  truth_score ~ log(repetition_numeric + 1),
  data = truth_ratings
)

# Compare models via AIC
AIC(model_linear, model_log)

# Compare via R-squared
summary(model_linear)$r.squared
summary(model_log)$r.squared

# Formal comparison (nested test is not exact because predictors differ,
# but gives relative fit)
anova(model_linear, model_log)


# -----------------------------
# 7) Graphs with fitted lines
# -----------------------------

means_plot <- truth_ratings %>%
  group_by(repetition_numeric) %>%
  summarise(
    mean_truth = mean(truth_score),
    se = sd(truth_score)/sqrt(n()),
    .groups = "drop"
  )

ggplot(means_plot, aes(x = repetition_numeric, y = mean_truth)) +
  geom_point(size = 3) +
  geom_line() +
  stat_smooth(
    method = "lm",
    se = FALSE,
    linetype = "dashed"
  ) +
  stat_smooth(
    method = "lm",
    formula = y ~ log(x + 1),
    se = FALSE
  ) +
  labs(
    x = "Repetition",
    y = "Mean Truth Rating",
    title = "Linear vs Logarithmic Fit"
  )


# -----------------------------
# 8) Cohen d's
# -----------------------------

nice(aov_rm, es = "pes")

# Ensure repetition_factor became character before pivoting
wide_means <- truth_ratings %>%
  group_by(participant_id, repetition_factor) %>%
  summarise(mean_truth = mean(truth_score), .groups = "drop") %>%
  mutate(repetition_factor = as.character(repetition_factor)) %>%
  pivot_wider(names_from = repetition_factor, values_from = mean_truth)

# Function for paired Cohen's d
paired_d <- function(x, y) {
  diff <- x - y
  mean(diff, na.rm = TRUE) / sd(diff, na.rm = TRUE)
}

levels_rep <- c("0","1","9","18","27")

# Create empty matrix
d_matrix <- matrix(NA,
                   nrow = length(levels_rep),
                   ncol = length(levels_rep),
                   dimnames = list(levels_rep, levels_rep))

# Fill matrix
for (i in 1:length(levels_rep)) {
  for (j in 1:length(levels_rep)) {
    if (i < j) {
      d_matrix[i, j] <- paired_d(
        wide_means[[levels_rep[i]]],
        wide_means[[levels_rep[j]]]
      )
    }
  }
}

# Convert to dataframe for nicer printing
d_table <- as.data.frame(d_matrix)
d_table


