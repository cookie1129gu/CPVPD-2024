import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set Chinese font support
plt.rcParams["font.family"] = ["Times New Roman"]
plt.rcParams['axes.unicode_minus'] = False

# Define font size variables
FONT_SIZE = 10
plt.rcParams.update({
    'font.size': FONT_SIZE,
    'axes.titlesize': FONT_SIZE,
    'axes.labelsize': 1.5 * FONT_SIZE,
    'xtick.labelsize': FONT_SIZE,
    'ytick.labelsize': 1.5 * FONT_SIZE,
    'legend.fontsize': FONT_SIZE
})

# Read data
df = pd.read_excel(r'\CPVPD-2024_SurfaceTerrain_Statistics_Percentage.xlsx')

categories = df['kind']
percentages = df['Percentage']
areas = df['Area']

# Create figure and primary axis (increase figure height)
fig, ax1 = plt.subplots(figsize=(10, 7))  # Height increased from 6 to 7

# Set x-axis positions
x = np.arange(len(categories))
width = 0.4

# ========== Area bar chart (primary axis) ==========
bars = ax1.bar(x, areas, width, color='skyblue',
               edgecolor='grey', alpha=0.7,
               label='Area (km²)')
ax1.set_ylabel('Area (km²)', color='skyblue')
ax1.tick_params(axis='y', labelcolor='skyblue')

# ========== Percentage line chart (secondary axis) ==========
ax2 = ax1.twinx()
line, = ax2.plot(x, percentages, color='salmon',
                 marker='o', markersize=8,
                 linewidth=2, label='Percentage (%)')
ax2.set_ylabel('Percentage (%)', color='salmon')
ax2.tick_params(axis='y', labelcolor='salmon')

# ========== Label optimization ==========
# X-axis labels (rotated 45 degrees)
ax1.set_xticks(x)
ax1.set_xticklabels(categories, rotation=45,
                    ha='right', rotation_mode='anchor')


# ========== Keep only bar chart labels ==========
def add_bar_labels(ax, values, vertical_offset=0.02):
    """Add only bar chart labels"""
    max_value = max(values)
    offset = max_value * vertical_offset

    for i, v in enumerate(values):
        ax.text(i, v + offset, f'{v:.3f}',
                ha='center', va='bottom',
                color=ax.yaxis.label.get_color())


add_bar_labels(ax1, areas, vertical_offset=max(areas) * 0.02)

# ========== Legend and title ==========
# Combine legends and adjust position
lines = [bars, line]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right',
           bbox_to_anchor=(0.98, 0.98))  # Slightly moved inward

# # Add title (increase pad value)
# plt.title('Comparison of Area and Percentage by Category', pad=20)
# ax1.grid(True, linestyle='--', alpha=0.4, axis='y')

# ========== Manual margin adjustment ==========
plt.subplots_adjust(
    left=0.12,  # Left margin
    right=0.88,  # Right margin (space for right y-axis)
    bottom=0.25,  # Bottom margin (space for rotated labels)
    top=0.9  # Top margin
)

# Save image
try:
    plt.savefig(r'\Terrain_Statistics_Comparison.png',
                dpi=400, bbox_inches=None)
    print("Chart successfully saved to specified path")
except Exception as e:
    print(f"Error saving chart: {e}")

plt.show()