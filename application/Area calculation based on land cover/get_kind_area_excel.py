import pandas as pd

# Read Excel file
file_path = r'\CPVPD-2024-Excel.xlsx'
df = pd.read_excel(file_path)

# Group by 'kind' and calculate total area
grouped = df.groupby('kind')['Area'].sum().reset_index()

# Convert area from square meters to square kilometers
grouped['Area'] = grouped['Area'] / 1000000

# Save results to a new Excel file
output_path = r'\kind_area_statistics_results.xlsx'
grouped.to_excel(output_path, index=False)

print(f"Statistics results have been saved to {output_path}")
