import pandas as pd
import sys
import csv

# 🔥 Handle large CSV fields
csv.field_size_limit(sys.maxsize)

FILE_PATH = "nasdaq_exteral_data.csv"

print("🚀 Finding companies with >5000 occurrences...")

# Dictionary to hold counts
company_counts = {}

# 🔹 Read CSV in chunks
for chunk in pd.read_csv(
    FILE_PATH,
    usecols=["Stock_symbol"],
    chunksize=10000,
    on_bad_lines='skip',
    engine='python'
):
    counts = chunk["Stock_symbol"].value_counts()
    for company, count in counts.items():
        company_counts[company] = company_counts.get(company, 0) + count

# 🔹 Convert to Series for easy filtering
company_counts_series = pd.Series(company_counts)

# 🔹 Filter by count > 5000
filtered_companies = company_counts_series[company_counts_series > 5000]

# 🔹 Sort descending by counts
filtered_companies = filtered_companies.sort_values(ascending=False)

print(f"✅ Total companies with >5000 rows: {len(filtered_companies)}\n")

# 🔹 Print results
for company, count in filtered_companies.items():
    print(f"{company}: {count}")