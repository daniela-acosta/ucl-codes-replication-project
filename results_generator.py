import numpy as np
import pandas as pd

np.random.seed(42)

# -----------------------------
# Parameters (Exp 2 structure)
# -----------------------------
n_participants = 200          # 8 participants -> perfectly balanced across 8 counterbalance versions (1 per version)
n_sets = 8                  # 8 sets of statements (4 old, 4 new per participant)
statements_per_set = 8
n_statements_total = n_sets * statements_per_set  # 64

repeated_levels = [1, 9, 18, 27]  # repetition conditions for OLD sets
n_old_sets = 4
n_new_sets = 4
assert n_old_sets + n_new_sets == n_sets

# Ideal/clean mean truth pattern (diminishing returns)
# mean_truth = {0: 3.64, 1: 4.26, 9: 4.70, 18: 4.80, 27: 4.87}
# sd_within = 0.35

# Failed replication
mean_truth = {
    0: 3.60,
    1: 3.65,
    9: 4.00,
    18: 4.40,
    27: 4.80
}
sd_within = 0.35

genders = ["female", "male", "other"]
edu_levels = ["high_school", "some_college", "undergrad", "postgrad"]

# -----------------------------
# Build statement bank: 8 sets × 8 statements
# -----------------------------
statement_bank = []
for set_id in range(1, n_sets + 1):
    for j in range(1, statements_per_set + 1):
        statement_bank.append({
            "set_id": set_id,
            "statement_id": f"SET{set_id}_S{j:02d}"
        })
statement_bank = pd.DataFrame(statement_bank)

# -----------------------------
# Counterbalancing scheme (8 versions)
# For version v0 (0..7):
#   old sets = [v0+1, v0+2, v0+3, v0+4] (wrapping 1..8)
#   old sets map to [1, 9, 18, 27] in that order
#   remaining 4 sets are new (0)
# -----------------------------
def wrap_set(x: int) -> int:
    return ((x - 1) % n_sets) + 1

def rep_map_for_version(v0: int) -> dict:
    old_sets = [wrap_set(v0 + k) for k in range(1, 5)]  # 4 old sets
    new_sets = [s for s in range(1, n_sets + 1) if s not in old_sets]

    mapping = {s: rep for s, rep in zip(old_sets, repeated_levels)}
    for s in new_sets:
        mapping[s] = 0
    return mapping  # set_id -> repetition_times

# Precompute mappings for versions 1..8
version_maps = {v: rep_map_for_version(v0=v - 1) for v in range(1, 9)}

# -----------------------------
# Assign participants to versions (balanced)
# -----------------------------
participants = [f"P{str(i).zfill(2)}" for i in range(1, n_participants + 1)]
versions = np.tile(np.arange(1, 9), int(np.ceil(n_participants / 8)))[:n_participants]
np.random.shuffle(versions)

rows = []

# -----------------------------
# Simulate Session 2 truth ratings (integer 1–6)
# One row per participant × statement (64 rows each participant)
# -----------------------------
def gen_truth(rep: int) -> int:
    """Generate an integer truth rating on a 1–6 Likert scale."""
    t = np.random.normal(loc=mean_truth[int(rep)], scale=sd_within)
    t = np.clip(t, 1, 6)
    return int(round(t))

for pid, cb_version in zip(participants, versions):
    age = int(np.random.randint(18, 36))
    gender = str(np.random.choice(genders))
    edu = str(np.random.choice(edu_levels))

    mapping = version_maps[int(cb_version)]  # set_id -> repetition_times

    # Build participant’s trials (all 64 statements)
    trials = statement_bank.copy()
    trials["participant_id"] = pid
    trials["cb_version"] = int(cb_version)
    trials["repetition_times"] = trials["set_id"].map(mapping).astype(int)

    # Randomize Session 2 presentation order (optional)
    trials = trials.sample(frac=1, random_state=np.random.randint(0, 1_000_000)).reset_index(drop=True)

    # Generate integer truth scores
    trials["truth_score"] = trials["repetition_times"].apply(gen_truth).astype(int)

    # Add demographics (constant per participant)
    trials["age"] = age
    trials["gender"] = gender
    trials["edu_level"] = edu

    rows.append(trials)

df = pd.concat(rows, ignore_index=True)

# -----------------------------
# Sanity checks
# -----------------------------
print("Total rows:", df.shape[0], "Expected:", n_participants * n_statements_total)

print("\nPer-participant repetition counts (should be 32 new(0) and 8 each of 1/9/18/27):")
counts = df.groupby(["participant_id", "repetition_times"]).size().unstack(fill_value=0)
print(counts)

print("\nCounterbalancing check: across all participants, how often each set appears in each repetition_times:")
cb_check = (
    df.drop_duplicates(["participant_id", "set_id"])
      .groupby(["set_id", "repetition_times"])
      .size()
      .unstack(fill_value=0)
)
print(cb_check)

print("\nTruth score value counts (should only be integers 1–6):")
print(df["truth_score"].value_counts().sort_index())

# -----------------------------
# Save
# -----------------------------
out_path = "failed_counterbalanced.csv"
df.to_csv(out_path, index=False)
print("\nSaved:", out_path)
