import pandas as pd
import numpy as np

# Load the raw USGS data
try:
    df = pd.read_csv('data/raw/usgs/MCS2026_Commodities_Data.csv', encoding='latin-1')
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit(1)

# Ensure 'Value' is numeric where possible for aggregration
def clean_value(val):
    if pd.isna(val): return 0.0
    val_str = str(val).replace(',', '').strip()
    try:
        return float(val_str)
    except:
        return 0.0

df['NumericValue'] = df['Value'].apply(clean_value)

commodities_to_check = ['rare earth', 'graphite', 'tungsten']

print("=== DOMINANT PRODUCERS ANALYSIS ===\n")

for target in commodities_to_check:
    # Find matching commodity components
    com_df = df[df['Commodity'].str.lower().str.contains(target, na=False)]
    
    # Filter for production statistics
    prod_df = com_df[
        com_df['Statistics'].str.lower().str.contains('production|mine', na=False) &
        (com_df['Country'].notna()) & 
        (com_df['Country'].str.lower() != 'world total (rounded)') &
        (com_df['Country'].str.lower() != 'world total') &
        (com_df['Country'].str.lower() != 'other countries') &
        (com_df['Country'].str.lower() != 'united states')
    ]
    
    if len(prod_df) == 0:
        print(f"No country-level production data found for {target.upper()}")
        continue
        
    # Get the latest year available for this commodity's production
    latest_year = prod_df['Year'].max()
    latest_prod = prod_df[prod_df['Year'] == latest_year]
    
    # Aggregate by country 
    country_prod = latest_prod.groupby('Country')['NumericValue'].sum().reset_index()
    country_prod = country_prod.sort_values(by='NumericValue', ascending=False)
    
    total_prod = country_prod['NumericValue'].sum()
    
    print(f"--- {target.upper()} (Year: {latest_year}) ---")
    if total_prod == 0:
        print("Total production is 0 or unparseable.\n")
        continue
        
    for idx, row in country_prod.head(5).iterrows():
        pct = (row['NumericValue'] / total_prod) * 100
        print(f"  {row['Country']:<20}: {row['NumericValue']:>12,.0f} ({pct:>5.1f}%)")
    
    # Check if China is #1
    if len(country_prod) > 0 and 'china' in country_prod.iloc[0]['Country'].lower():
        print(f"  -> CONCLUSION: YES, China is the dominant producer of {target.upper()}.\n")
    else:
        top_country = country_prod.iloc[0]['Country'] if len(country_prod) > 0 else "Unknown"
        print(f"  -> CONCLUSION: NO, {top_country} is the dominant producer of {target.upper()}.\n")
