import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# --- Configuration ---
INPUT_FILE = 'top_100_usd_profitable.tsv' # Changed input file
OUTPUT_DIR = 'charts-usd' # Changed output directory
HOURS_PER_YEAR = 365 * 24

# --- Create output directory ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading data from {INPUT_FILE}...")
try:
    # Load the data, specifying tab separator
    df = pd.read_csv(INPUT_FILE, sep='\\t', engine='python')
    print("Data loaded successfully.")
    print(f"Initial rows: {len(df)}")

    # --- Data Cleaning and Preparation ---
    print("Cleaning and preparing data...")
    numeric_cols = [
        'tick_lower', 'tick_upper', 'ticks', 'deposited_token0',
        'deposited_token1', 'total_deposited_usd', 'total_profit_usd',
        'apr_30d', 'apr', 'duration_h', 'in_range_h', 'in_range_percent',
        'impermanent_loss', 'price_lower', 'price_upper', 'price_range',
        'price_mid', 'auto_rebalance', 'cutoff_tick_low', 'cutoff_tick_high',
        'buffer_ticks_below', 'buffer_ticks_above', 'tick_spaces_below',
        'tick_spaces_above', 'dust_bp'
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"Warning: Expected numeric column '{col}' not found.")

    essential_cols = ['total_profit_usd', 'total_deposited_usd', 'duration_h', 'apr', 'ticks', 'in_range_percent', 'auto_rebalance']
    df.dropna(subset=[col for col in essential_cols if col in df.columns], inplace=True)
    print(f"Rows after dropping NaNs in essential columns: {len(df)}")

    # --- Calculations ---
    print("Performing calculations...")
    df['calculated_apr'] = np.where(
        (df['total_deposited_usd'] > 0) & (df['duration_h'] > 0),
        (df['total_profit_usd'] / df['total_deposited_usd']) * (HOURS_PER_YEAR / df['duration_h']) * 100,
        0
    )

    # Handle potential infinite values in APR calculations if duration_h is very small or zero after cleaning
    df.replace([np.inf, -np.inf], np.nan, inplace=True) # Replace infinities with NaN
    df.dropna(subset=['calculated_apr'], inplace=True) # Drop rows where calculation resulted in NaN
    print(f"Rows after handling potential infinite APRs: {len(df)}")

    # Cap APRs for visualization stability (using a high but reasonable cap)
    apr_cap = 5000 # Example cap, adjust if needed based on data
    df['apr_capped'] = df['apr'].clip(upper=apr_cap)
    df['calculated_apr_capped'] = df['calculated_apr'].clip(upper=apr_cap)

    if 'auto_rebalance' in df.columns:
         df['strategy_type'] = df['auto_rebalance'].map({0: 'Manual', 1: 'VFAT Auto'})
    else:
         df['strategy_type'] = 'Unknown'

    print("Data preparation complete.")

    # --- Visualizations ---
    print("Generating visualizations...")
    sns.set_theme(style="whitegrid")

    # 1. Profit Distribution (Enhanced Ticks)
    plt.figure(figsize=(10, 6))
    sns.histplot(df['total_profit_usd'], kde=True, bins=20)
    plt.title('Distribution of Total Profit (Top 100 USD-USD Positions)')
    plt.xlabel('Total Profit (USD)')
    plt.ylabel('Number of Positions')
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ','))) # Format ticks
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '1_profit_distribution.png'))
    plt.close()
    print("Saved: 1_profit_distribution.png")

    # 2. Profit by Pool (Enhanced Labels)
    if 'pool_symbol' in df.columns:
        plt.figure(figsize=(10, 7)) # Adjusted size
        profit_by_pool = df.groupby('pool_symbol')['total_profit_usd'].sum().sort_values(ascending=False)
        ax = sns.barplot(x=profit_by_pool.values, y=profit_by_pool.index, palette="viridis")
        plt.title('Total Profit by Pool Symbol (Top 100 USD-USD Positions)')
        plt.xlabel('Total Profit Sum (USD)')
        plt.ylabel('Pool Symbol')
        ax.bar_label(ax.containers[0], fmt='$%.0f', padding=3) # Add labels to bars
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '2_profit_by_pool.png'))
        plt.close()
        print("Saved: 2_profit_by_pool.png")
    else:
        print("Skipping Profit by Pool: 'pool_symbol' column not found.")

    # 3. In-Range Percentage Distribution (Clearer Title)
    if 'in_range_percent' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df['in_range_percent'], kde=False, bins=10, binwidth=10, binrange=(0,100)) # Use fixed bins
        plt.title('Distribution of Position Time In Defined Range (%)')
        plt.xlabel('Percentage of Time Position Was In Range')
        plt.ylabel('Number of Positions')
        plt.xticks(range(0, 101, 10))
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '3_in_range_distribution.png'))
        plt.close()
        print("Saved: 3_in_range_distribution.png")
    else:
         print("Skipping In-Range Distribution: 'in_range_percent' column not found.")

    # 4. Profit by Strategy (Added Stripplot Overlay)
    if 'strategy_type' in df.columns and df['strategy_type'].nunique() > 1:
        plt.figure(figsize=(8, 7))
        sns.boxplot(x='strategy_type', y='total_profit_usd', data=df, palette="coolwarm", showfliers=False) # Hide outlier points from boxplot
        sns.stripplot(x='strategy_type', y='total_profit_usd', data=df, color=".25", size=4, alpha=0.6) # Overlay individual points
        plt.title('Profit Comparison: Manual vs. VFAT Auto-Rebalance')
        plt.xlabel('Rebalancing Strategy')
        plt.ylabel('Total Profit (USD)')
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'${y:,.0f}'))
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '4_profit_by_strategy.png'))
        plt.close()
        print("Saved: 4_profit_by_strategy.png")
    else:
        print("Skipping Profit by Strategy: Not enough data or strategy column missing.")

    # 5. APR Comparison Distribution (Combined Plot)
    if 'apr_capped' in df.columns and 'calculated_apr_capped' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.histplot(df['apr_capped'], color="skyblue", label='Stored APR', kde=True, stat="density", linewidth=0, bins=30)
        sns.histplot(df['calculated_apr_capped'], color="red", label='Calculated APR', kde=True, stat="density", linewidth=0, alpha=0.6, bins=30)
        plt.title(f'Distribution Comparison: Stored vs. Calculated APR (Capped at {apr_cap}%)')
        plt.xlabel('APR (%)')
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '5_apr_comparison_distribution.png'))
        plt.close()
        print("Saved: 5_apr_comparison_distribution.png")
    else:
        print("Skipping APR Comparison: APR columns not found or insufficient data after capping.")

    # --- Scatter Plots (Enhanced) ---
    scatter_cols = {
        'duration_h': ('Duration (Hours)', True), # Use log scale
        'total_deposited_usd': ('Total Deposited (USD)', True), # Use log scale
        'ticks': ('Range Width (Ticks)', True) # Use log scale
    }

    for col, (label, use_log) in scatter_cols.items():
        if col in df.columns:
            plt.figure(figsize=(10, 7))
            sns.scatterplot(
                x=col, y='total_profit_usd', data=df,
                hue='strategy_type' if 'strategy_type' in df.columns and df['strategy_type'].nunique() > 1 else None,
                size='total_deposited_usd', sizes=(20, 200), # Vary point size by deposit
                alpha=0.7, palette='viridis'
            )
            plt.title(f'Profit vs. {label} (Top 100 USD-USD)')
            plt.xlabel(label)
            plt.ylabel('Total Profit (USD)')
            if use_log:
                plt.xscale('log')
                plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
            plt.yscale('log')
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'${y:,.0f}'))
            plt.legend(title=f'{label}/Strategy/Deposit Size', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend
            plt.savefig(os.path.join(OUTPUT_DIR, f'6_{col}_vs_profit.png'))
            plt.close()
            print(f"Saved: 6_{col}_vs_profit.png")
        else:
            print(f"Skipping Profit vs {label}: '{col}' column not found.")

    # --- Correlation Heatmap ---
    if len(df) > 1:
        corr_cols = ['total_profit_usd', 'total_deposited_usd', 'apr_capped', 'calculated_apr_capped', 'duration_h', 'in_range_percent', 'ticks', 'impermanent_loss']
        corr_cols_present = [c for c in corr_cols if c in df.columns]
        if len(corr_cols_present) > 1:
            correlation_matrix = df[corr_cols_present].corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
            plt.title('Correlation Matrix of Key Position Features (Top 100 USD-USD)')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, '7_correlation_heatmap.png'))
            plt.close()
            print("Saved: 7_correlation_heatmap.png")
        else:
            print("Skipping Correlation Heatmap: Not enough numeric columns present.")
    else:
         print("Skipping Correlation Heatmap: Not enough data rows.")

    # --- Pairplot (Optional - can be slow/dense for many variables) ---
    pair_cols = ['total_profit_usd', 'total_deposited_usd', 'duration_h', 'in_range_percent', 'ticks', 'strategy_type']
    pair_cols_present = [c for c in pair_cols if c in df.columns]
    if len(df) > 1 and 'strategy_type' in pair_cols_present and len(pair_cols_present) > 2:
        print("Generating pairplot (this might take a moment)...")
        pair_plot_data = df[pair_cols_present].copy()
        # Apply log transform for visualization where it makes sense
        for col in ['total_profit_usd', 'total_deposited_usd', 'duration_h', 'ticks']:
             if col in pair_plot_data.columns:
                 pair_plot_data[col] = np.log1p(pair_plot_data[col]) # Use log1p to handle zeros
                 pair_plot_data.rename(columns={col: f'log_{col}'}, inplace=True)

        pp = sns.pairplot(pair_plot_data, hue='strategy_type', palette='viridis', diag_kind='kde', height=2.5)
        pp.fig.suptitle('Pairwise Relationships Between Key Features (Log Scaled)', y=1.02) # Add title
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '8_pairplot.png'))
        plt.close()
        print("Saved: 8_pairplot.png")
    else:
        print("Skipping Pairplot: Not enough data or required columns (incl. strategy_type) missing.")

    # --- Top 5 Positions Detailed Plot ---
    if len(df) >= 5:
        print("Generating Top 5 Positions details plot...")
        top_5_df = df.sort_values('total_profit_usd', ascending=False).head(5)

        plt.figure(figsize=(12, 8))
        ax = sns.barplot(x='total_profit_usd', y=top_5_df['id'].astype(str), data=top_5_df, palette='magma', orient='h')

        plt.title('Details of Top 5 Most Profitable USD-USD Positions')
        plt.xlabel('Total Profit (USD)')
        plt.ylabel('Position ID')
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        # Add annotations
        for index, row in top_5_df.iterrows():
            profit_val = row['total_profit_usd']
            annotation_text = (
                f"Pool: {row['pool_symbol']} | APR: {row['apr']:.2f}%\n"
                f"Range: {row['price_lower']:.4f} - {row['price_upper']:.4f} ({int(row['ticks'])} ticks)\n"
                f"Deposited: ${row['total_deposited_usd']:,.2f}\n"
                f"Duration: {int(row['duration_h'])}h | In Range: {int(row['in_range_percent'])}%\n"
                f"Strategy: {row['strategy_type']}"
            )
            # Position text slightly to the right of the bar
            ax.text(profit_val * 1.01 , # Position x slightly right of bar end
                    top_5_df.index.get_loc(index), # Position y (bar index)
                    annotation_text,
                    va='center', # Vertical alignment
                    ha='left', # Horizontal alignment
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.5)) # Optional background box

        plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout slightly if text overlaps
        plt.savefig(os.path.join(OUTPUT_DIR, '9_top_5_positions_details.png'))
        plt.close()
        print("Saved: 9_top_5_positions_details.png")
    else:
        print("Skipping Top 5 Details plot: Fewer than 5 positions available.")

    print("\nAll visualizations generated successfully in the 'charts-usd' directory.")

except FileNotFoundError:
    print(f"Error: Input file '{INPUT_FILE}' not found.")
except pd.errors.EmptyDataError:
    print(f"Error: Input file '{INPUT_FILE}' is empty.")
except Exception as e:
    print(f"An unexpected error occurred: {e}") 