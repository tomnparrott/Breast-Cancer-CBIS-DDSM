import pandas as pd

# Load the evaluation CSV produced by the pipeline
p = r"C:\Users\tompa\Desktop\Final Project\Breast Cancer-CBIS-DDSM\Data\Processed\runs\20260223_185850_seed42\eval_outputs\2026-02-25\20260223_185850_seed42\metrics\test_predictions_with_meta.csv"
# Read prediction metadata into a df for audit analysis.
df = pd.read_csv(p)

# Classify each row into a standard confusion matrix outcome
df["outcome"] = "TN"
df.loc[(df.y_true==0) & (df.y_pred==1), "outcome"] = "FP"
df.loc[(df.y_true==1) & (df.y_pred==0), "outcome"] = "FN"
df.loc[(df.y_true==1) & (df.y_pred==1), "outcome"] = "TP"

# Convert a boolean into a percentage 
def rate(x): 
    return (x.mean()*100) if len(x) else 0

# Collect summary rows before building the final output table
summ = []
# Add the overall audit flag rate across the full dataset
summ.append(("overall", "", len(df), rate(df["audit_flag_any"])))

if "density_group" in df.columns:
    for g, d in df.groupby("density_group"):
        summ.append(("density_group", g, len(d), rate(d["audit_flag_any"])))

# report the audit flag rate for each prediction outcome bucket
for o, d in df.groupby("outcome"):
    summ.append(("outcome", o, len(d), rate(d["audit_flag_any"])))

# Build the summary table and save it alongside the CSV
out = pd.DataFrame(summ, columns=["group_by", "group", "n", "pct_flag_any"])
out_path = p.replace("test_predictions_with_meta.csv", "audit_summary.csv")
out.to_csv(out_path, index=False)

# Print the sorted summary for inspection and show where it was written
print(out.sort_values(["group_by","pct_flag_any"], ascending=[True, False]))
print("\nSaved:", out_path)