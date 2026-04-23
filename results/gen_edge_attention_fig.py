import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set IEEE style parameters
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'serif'],
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

def plot_edge_attention():
    csv_path = r'S:\Demand_Forecast\results\usgs\edge_type_importance.csv'
    out_path = r'S:\Demand_Forecast\results\fig_edge_attention.pdf'
    png_out_path = r'S:\Demand_Forecast\results\fig_edge_attention.png'
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    
    # Clean up labels for presentation
    df['label'] = df['edge_type'].str.replace('_', ' ').str.title()
    
    # Sort for horizontal bar chart (highest at top)
    df = df.sort_values('mean_attention', ascending=True)
    
    fig, ax = plt.subplots(figsize=(3.5, 3.0)) # IEEE single column width is ~3.5 inches
    
    # Create horizontal bars
    y_pos = np.arange(len(df))
    bars = ax.barh(y_pos, df['mean_attention'], xerr=df['std_attention'], 
                   align='center', color='#4a7fb8', edgecolor='black', 
                   linewidth=0.5, capsize=2, error_kw={'elinewidth': 0.8})
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['label'])
    ax.set_xlabel('Mean Attention Weight')
    ax.set_title('USGS Structural Attention by Edge Type')
    
    # Grid lines
    ax.xaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.savefig(png_out_path, format='png', bbox_inches='tight', dpi=300)
    print(f"Saved {out_path} and {png_out_path}")

if __name__ == '__main__':
    plot_edge_attention()
