import pandas as pd
from pathlib import Path

# -------- CONFIG --------
INPUT_CSV = "experiment setup - statements.csv"
OUTPUT_DIR = "session1_csvs"

SET_SIZE = 8
REPETITIONS = [1, 9, 18, 27]  # repetition conditions
# ------------------------


def main():
    # read statements
    df = pd.read_csv(INPUT_CSV)

    if len(df) != 64:
        raise ValueError("Expected exactly 64 statements")

    # ensure correct order
    df = df.sort_values("statement_id").reset_index(drop=True)

    # assign statement sets (fixed across versions)
    df["statement_set"] = (df.index // SET_SIZE) + 1

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    # build 8 counterbalance groups
    for version in range(8):
        rows = []

        # choose 4 sets to include in Session 1
        repeated_sets = [
            ((version + i) % 8) + 1
            for i in range(4)
        ]

        # assign repetition counts to those sets
        repetition_map = {
            repeated_sets[i]: REPETITIONS[i]
            for i in range(4)
        }

        for _, row in df.iterrows():
            set_id = row["statement_set"]

            if set_id in repetition_map:
                reps = repetition_map[set_id]
                for _ in range(reps):
                    rows.append({
                        "statement_set": set_id,
                        "statement": row["statement"],
                        "statement_id": row["statement_id"],
                        "repetitions": reps
                    })

        out_df = pd.DataFrame(rows)

        out_path = output_dir / f"session1_version_{version + 1}.csv"
        out_df.to_csv(out_path, index=False)

    print(f"Done. Session 1 CSVs written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
