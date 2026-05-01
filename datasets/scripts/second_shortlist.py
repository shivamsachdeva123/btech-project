import pandas as pd
import sys
import csv

# 🔥 Handle large CSV fields
csv.field_size_limit(sys.maxsize)

INPUT_FILE = "nasdaq_exteral_data.csv"
OUTPUT_FILE = "real_companies_news.csv"

# ✅ Updated real companies list
REAL_COMPANIES = [
    "AAL","AAPL","ABBV","ADBE","AEO","AIG","ALK","AMD","AMGN","AMT","AMZN",
    "ANF","BA","BABA","BHP","BIDU","BIIB","BSX","BX","C","CAG","CAT","CI",
    "CMCSA","CMG","COP","COST","CRM","CSX","CVX","D","DIS","DPZ","DUK","EA",
    "EBAY","EOG","FSMB","GE","GILD","GOOG","GRPN","GS","GSK","HAL","HUM","INTC",
    "KO","KSS","MMM","MRK","MS","MSFT","MU","NEE","NKE","NOK","NVDA","NVS","ORCL",
    "PANW","PBR","PCG","PEP","QCOM","SBUX","SIRI","SLB","T","TJX","TM","TSLA",
    "TSM","TSN","TXN","ULTA","V","VRTX","WDC","WFC","WMT","XOM"
]

print("🚀 Filtering dataset for updated real companies list...")

filtered_chunks = []

# 🔹 Read CSV in chunks (memory efficient)
for chunk in pd.read_csv(
    INPUT_FILE,
    chunksize=10000,
    on_bad_lines='skip',
    engine='python'
):
    # 🔹 Keep only rows with real companies
    chunk = chunk[chunk["Stock_symbol"].isin(REAL_COMPANIES)]

    # 🔹 Convert Date column safely
    chunk["Date"] = pd.to_datetime(chunk["Date"], errors='coerce')

    # 🔹 Drop rows with invalid dates
    chunk = chunk.dropna(subset=["Date"])

    if not chunk.empty:
        filtered_chunks.append(chunk)

# 🔹 Combine all filtered data
if filtered_chunks:
    final_df = pd.concat(filtered_chunks, ignore_index=True)
else:
    print("❌ No data found for the selected companies.")
    exit()

# 🔹 Save filtered CSV
final_df.to_csv(OUTPUT_FILE, index=False)
print(f"💾 Saved filtered data to: {OUTPUT_FILE}")

# -------------------------------
# 📊 Min & Max Date Per Company
# -------------------------------
print("\n📊 Min and Max Date per Company:\n")
date_summary = final_df.groupby("Stock_symbol")["Date"].agg(["min", "max"])
print(date_summary)

# 🔹 Pretty print
print("\n📈 Pretty output:\n")
for company, row in date_summary.iterrows():
    print(f"{company}: {row['min'].date()} → {row['max'].date()}")