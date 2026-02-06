import numpy as np
import pandas as pd

# -----------------------------
# Experiment 2 "ideal replication" simulation
# One row = one (participant × statement) truth rating in Session 2
# Structure per participant:
#   - 32 NEW statements (repetition_times = 0)
#   - 32 OLD statements split evenly:
#       8 statements at 1 repetition
#       8 statements at 9 repetitions
#       8 statements at 18 repetitions
#       8 statements at 27 repetitions
# Total = 64 statements per participant
# -----------------------------

np.random.seed(42)

# ---- Parameters (match Exp 2 structure) ----
n_participants = 2
n_statements_total = 64

n_new = int(n_statements_total / 2)
repeated_levels = [1, 9, 18, 27]
n_per_repeated_level = 8  # 8 statements per repetition condition
assert n_new + len(repeated_levels) * n_per_repeated_level == n_statements_total

participants = [f"P{str(i).zfill(2)}" for i in range(1, n_participants + 1)]
statements = [f"S{str(i).zfill(3)}" for i in range(1, n_statements_total + 1)]

genders = ["female", "male", "other"]
edu_levels = ["high_school", "some_college", "undergrad", "postgrad"]

# ---- "Ideal replication" means (monotonic, diminishing returns) ----
# You can tweak these to match the paper's reported means if you prefer.
mean_truth = {
    0: 3.64,  # new
    1: 4.26,
    9: 4.70,
    18: 4.80,
    27: 4.87
}

# Within-condition variability (smaller = more "ideal/clean")
sd_within = 0.35

rows = []

for pid in participants:
    # participant-level demographics (constant across their 64 rows)
    age = np.random.randint(18, 36)  # 18–35 inclusive
    gender = np.random.choice(genders)
    edu = np.random.choice(edu_levels)

    # Assign 32 statements as NEW (0) and 32 as OLD (1/9/18/27)
    # Shuffle statement IDs per participant to avoid statement-specific artifacts
    sids = statements.copy()
    np.random.shuffle(sids)

    new_sids = sids[:n_new]
    old_sids = sids[n_new:]

    # Build repetition_times for OLD statements: 8 per repeated level
    reps_old = np.concatenate([np.repeat(rep, n_per_repeated_level) for rep in repeated_levels])
    np.random.shuffle(reps_old)  # randomize assignment of repetition counts to old statements

    # Combine into one trial list for Session 2 and randomize order
    trials = []

    for sid in new_sids:
        trials.append((sid, 0))

    for sid, rep in zip(old_sids, reps_old):
        trials.append((sid, int(rep)))

    np.random.shuffle(trials)  # random presentation order in Session 2

    # Generate truth scores
    for sid, rep in trials:
        truth = np.random.normal(loc=mean_truth[rep], scale=sd_within)
        truth = float(np.clip(truth, 1, 6))

        rows.append({
            "participant_id": pid,
            "statement_id": sid,
            "repetition_times": rep,
            "truth_score": round(truth, 2),
            "age": int(age),
            "gender": str(gender),
            "edu_level": str(edu)
        })

df = pd.DataFrame(rows)

# ---- Sanity checks ----
expected_rows = n_participants * n_statements_total
print("Rows:", df.shape[0], "Expected:", expected_rows)

# Check per-participant counts per repetition condition
counts = df.groupby(["participant_id", "repetition_times"]).size().unstack(fill_value=0)
print("\nPer-participant condition counts (should be 32 for 0, and 8 for 1/9/18/27):")
print(counts.describe())

# Global counts
print("\nGlobal repetition_times counts:")
print(df["repetition_times"].value_counts().sort_index())

# ---- Save CSV ----
out_path = "ideal_replication_experiment2_session2.csv"
df.to_csv(out_path, index=False)
print("\nSaved:", out_path)
