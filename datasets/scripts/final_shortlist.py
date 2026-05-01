import pandas as pd
from datetime import timedelta

# Load filtered dataset
df = pd.read_csv("real_companies_news.csv", parse_dates=["Date"])

# Define 1-year window
window_days = 365

print("🚀 Finding optimal 1-year windows per company...\n")

results = []

for company in df["Stock_symbol"].unique():
    company_df = df[df["Stock_symbol"] == company].sort_values("Date")
    
    if company_df.empty:
        continue

    dates = company_df["Date"].tolist()
    max_count = 0
    best_start = None
    best_end = None

    # Sliding window (strict 1-year)
    start_idx = 0
    for end_idx in range(len(dates)):
        while dates[end_idx] - dates[start_idx] >= timedelta(days=window_days):
            start_idx += 1
        count = end_idx - start_idx + 1
        if count > max_count:
            max_count = count
            best_start = dates[start_idx]
            best_end = best_start + timedelta(days=window_days)

    results.append({
        "Company": company,
        "Start_Date": best_start.date() if best_start else None,
        "End_Date": best_end.date() if best_end else None,
        "News_Count": max_count
    })

# Convert to DataFrame and sort by News_Count descending
result_df = pd.DataFrame(results).sort_values("News_Count", ascending=False)

# Show only top 20 companies
top_20_df = result_df.head(20)

# Save top 20 to CSV
top_20_df.to_csv("top20_companies_best_1yr_window.csv", index=False)

print("✅ Top 20 companies with optimal 1-year news windows:\n")
print(top_20_df)