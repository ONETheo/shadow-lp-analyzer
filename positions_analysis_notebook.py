# %%
# # Shadow Analytics - Position Analysis Notebook
#
# This script analyzes Shadow DEX position data, focusing on profitability and APR verification.
# It's formatted to be run cell-by-cell in environments supporting the Jupyter interactive Python format (e.g., VS Code).

# %%
# ## 1. Setup and Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
INPUT_FILE = 'top_1000_profitable.tsv' # Using the extracted top 1000
OUTPUT_DIR = 'notebook_charts' # Separate dir for charts from this notebook
HOURS_PER_YEAR = 365 * 24

# --- Plotting Style ---
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 7) # Default figure size

# --- Create output directory ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Setup complete.")

# %%
# ## 2. Load Data
# Load the dataset extracted previously.

print(f"Loading data from {INPUT_FILE}...")
try:
    df = pd.read_csv(INPUT_FILE, sep='\t', engine='python')
    print(f"Data loaded successfully. Shape: {df.shape}")
    print("First 5 rows:")
    display(df.head())
except FileNotFoundError:
    print(f"Error: Input file '{INPUT_FILE}' not found. Make sure the awk extraction step completed.")
except Exception as e:
    print(f"An error occurred during loading: {e}")

# %%
# ## 3. Data Cleaning and Preparation
# Convert relevant columns to numeric types and handle potential errors/missing values.

print("Cleaning and preparing data...")

if 'df' in locals(): # Check if df was loaded successfully
    numeric_cols = [
        'tick_lower', 'tick_upper', 'ticks', 'deposited_token0',
        'deposited_token1', 'total_deposited_usd', 'total_profit_usd',
        'apr_30d', 'apr', 'duration_h', 'in_range_h', 'in_range_percent',
        'impermanent_loss', 'price_lower', 'price_upper', 'price_range',
        'price_mid', 'auto_rebalance', 'cutoff_tick_low', 'cutoff_tick_high',
        'buffer_ticks_below', 'buffer_ticks_above', 'tick_spaces_below',
        'tick_spaces_above', 'dust_bp'
    ]
    initial_rows = len(df)

    for col in numeric_cols:
        if col in df.columns:
            # Handle potential commas and convert to numeric, coercing errors
            df[col] = df[col].astype(str).str.replace(',', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"Warning: Expected numeric column '{col}' not found.")

    # Define essential columns for core analysis
    essential_cols = ['total_profit_usd', 'total_deposited_usd', 'duration_h', 'apr']
    df.dropna(subset=[col for col in essential_cols if col in df.columns], inplace=True)
    print(f"Rows before cleaning: {initial_rows}, Rows after dropping essential NaNs: {len(df)}")

    # Add strategy type column for clarity
    if 'auto_rebalance' in df.columns:
         df['strategy_type'] = df['auto_rebalance'].map({0: 'Manual', 1: 'VFAT Auto'}).fillna('Unknown')
    else:
         df['strategy_type'] = 'Unknown'

    print("Data cleaning finished.")
    print("\nData Types after cleaning:")
    display(df.info())
else:
    print("DataFrame 'df' not loaded. Skipping cleaning.")

# %%
# ## 4. Calculate APR
# Recalculate APR based on profit, deposit, and duration for verification.

print("Calculating APR...")
if 'df' in locals():
    # Calculate APR, handle division by zero or missing values
    df['calculated_apr'] = np.where(
        (df['total_deposited_usd'] > 0) & (df['duration_h'] > 0),
        (df['total_profit_usd'] / df['total_deposited_usd']) * (HOURS_PER_YEAR / df['duration_h']) * 100,
        np.nan # Assign NaN if calculation is not possible
    )

    # Handle potential infinities resulting from calculation (e.g., duration_h = 0)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Calculate absolute and relative difference
    df['apr_diff_abs'] = (df['calculated_apr'] - df['apr']).abs()
    df['apr_diff_rel'] = np.abs(np.where(
        df['apr'] != 0,
        df['apr_diff_abs'] / df['apr'].abs(),
        np.where(df['calculated_apr'] != 0, 1, 0) # Handle zero division: if stored is 0, relative diff is 100% unless calculated is also 0
    ))

    print("APR calculation complete.")
    print("\nSummary statistics for APR differences:")
    display(df[['apr_diff_abs', 'apr_diff_rel']].describe())
else:
    print("DataFrame 'df' not loaded. Skipping APR calculation.")


# %%
# ## 5. Analyze APR Discrepancies
# Identify and display positions with significant differences between stored and calculated APR.

print("Analyzing APR discrepancies...")
if 'df' in locals():
    # Define thresholds for significant discrepancy
    abs_diff_threshold = 1.0 # e.g., difference > 1 percentage point
    rel_diff_threshold = 0.05 # e.g., difference > 5%

    discrepancies = df[
        (df['apr_diff_abs'] > abs_diff_threshold) & 
        (df['apr_diff_rel'] > rel_diff_threshold)
    ].sort_values('apr_diff_abs', ascending=False)

    print(f"Found {len(discrepancies)} positions with significant APR discrepancies.")

    if not discrepancies.empty:
        print("\nTop 10 discrepancies (by absolute difference):")
        display(discrepancies[['id', 'pool_symbol', 'apr', 'calculated_apr', 'apr_diff_abs', 'apr_diff_rel', 'total_profit_usd', 'total_deposited_usd', 'duration_h']].head(10))
    else:
        print("No significant APR discrepancies found based on the thresholds.")
else:
    print("DataFrame 'df' not loaded. Skipping discrepancy analysis.")


# %%
# ## 6. Basic Statistics
# Explore summary statistics for key numerical columns.

print("Calculating basic statistics...")
if 'df' in locals():
    stats_cols = ['total_profit_usd', 'total_deposited_usd', 'duration_h', 'apr', 'calculated_apr', 'in_range_percent', 'ticks']
    stats_cols_present = [c for c in stats_cols if c in df.columns]
    if stats_cols_present:
        display(df[stats_cols_present].describe())
    else:
        print("No columns available for statistics.")
else:
    print("DataFrame 'df' not loaded. Skipping statistics.")

# %%
# ## 7. Visualizations
# Create plots to understand distributions and relationships.

print("Generating visualizations...")

# %% [markdown]
# ### 7.1 Profit Distribution

# %%
if 'df' in locals():
    plt.figure(figsize=(10, 6))
    sns.histplot(df['total_profit_usd'], kde=True, bins=30)
    plt.title(f'Distribution of Total Profit (Top {len(df)} Positions)')
    plt.xlabel('Total Profit (USD)')
    plt.ylabel('Number of Positions')
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'nb_profit_distribution.png'))
    plt.show()
else:
    print("DataFrame 'df' not loaded. Skipping plot.")

# %% [markdown]
# ### 7.2 Profit by Pool (Top 15)

# %%
if 'df' in locals() and 'pool_symbol' in df.columns:
    plt.figure(figsize=(12, 8))
    profit_by_pool = df.groupby('pool_symbol')['total_profit_usd'].sum().sort_values(ascending=False).head(15)
    ax = sns.barplot(x=profit_by_pool.values, y=profit_by_pool.index, palette="viridis")
    plt.title(f'Total Profit by Pool Symbol (Top 15 Pools among Top {len(df)} Positions)')
    plt.xlabel('Total Profit Sum (USD)')
    plt.ylabel('Pool Symbol')
    ax.bar_label(ax.containers[0], fmt='$%.0f', padding=3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'nb_profit_by_pool.png'))
    plt.show()
else:
    print("DataFrame 'df' not loaded or 'pool_symbol' missing. Skipping plot.")

# %% [markdown]
# ### 7.3 Profit vs. Duration
# Scatter plot showing relationship between profit and position duration.

# %%
if 'df' in locals() and 'duration_h' in df.columns:
    plt.figure(figsize=(12, 7))
    sns.scatterplot(
        x='duration_h', y='total_profit_usd', data=df,
        hue='strategy_type', size='total_deposited_usd', sizes=(20, 250),
        alpha=0.6, palette='viridis'
    )
    plt.title(f'Profit vs. Duration (Top {len(df)} Positions)')
    plt.xlabel('Duration (Hours) - Log Scale')
    plt.ylabel('Total Profit (USD) - Log Scale')
    plt.xscale('log')
    plt.yscale('log')
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'${y:,.0f}'))
    plt.legend(title='Strategy / Deposit Size', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(os.path.join(OUTPUT_DIR, 'nb_profit_vs_duration.png'))
    plt.show()
else:
    print("DataFrame 'df' not loaded or 'duration_h' missing. Skipping plot.")

# %% [markdown]
# ### 7.4 Stored APR vs Calculated APR
# Scatter plot comparing the two APR values. Points diverging from the y=x line indicate discrepancies.

# %%
if 'df' in locals() and 'apr' in df.columns and 'calculated_apr' in df.columns:
    plt.figure(figsize=(10, 10))
    # Cap APR for better visualization
    apr_plot_cap = df[['apr', 'calculated_apr']].quantile(0.98).max() # Cap at 98th percentile of both
    apr_plot_cap = min(apr_plot_cap, 5000) # Ensure cap is not excessively high

    plot_df = df.copy()
    plot_df['apr'] = plot_df['apr'].clip(upper=apr_plot_cap)
    plot_df['calculated_apr'] = plot_df['calculated_apr'].clip(upper=apr_plot_cap)

    sns.scatterplot(x='apr', y='calculated_apr', data=plot_df, hue='strategy_type', alpha=0.7)
    # Add y=x line for reference
    max_val = max(plot_df['apr'].max(), plot_df['calculated_apr'].max())
    plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', label='y=x (Perfect Match)')

    plt.title(f'Stored APR vs. Calculated APR (Capped at {apr_plot_cap:.0f}%)')
    plt.xlabel('Stored APR (%)')
    plt.ylabel('Calculated APR (%)')
    plt.legend()
    plt.axis('equal') # Ensure aspect ratio is equal for y=x line clarity
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'nb_apr_comparison_scatter.png'))
    plt.show()
else:
    print("DataFrame 'df' not loaded or APR columns missing. Skipping plot.")

# %% [markdown]
# ## 8. What-If Scenario Analysis (Based on Historical Averages)
#
# **Disclaimer:** This section allows you to filter the loaded historical data (Top 1000 Profitable Positions) based on parameters you set. It then calculates the *average* performance (Profit, APR) of the positions **within that historical subset** that match your criteria. 
#
# **This is NOT a prediction of future returns.**
# - It's based only on the top 1000 most profitable historical positions, which is a biased sample.
# - Real-world performance depends heavily on unpredictable future market conditions (volatility, volume, price action).
#
# Use this to explore how positions with certain characteristics performed *in the past* within this specific dataset.

# %%
# ### 8.1 Define Scenario Parameters
# Modify the values below to set your desired scenario.
# Use `None` to ignore a specific filter (e.g., `target_pool = None` considers all pools).

# --- Scenario Inputs ---
target_pool = 'SHADOW/stS'  # Specific pool symbol (e.g., 'SHADOW/stS', 'wS/WETH', 'USDC.e/scUSD') or None
target_strategy = 'Manual' # 'Manual', 'VFAT Auto', or None

min_ticks = 50       # Minimum range width in ticks (e.g., 100) or None
max_ticks = 500      # Maximum range width in ticks (e.g., 1000) or None

min_deposit_usd = 1000  # Minimum initial deposit in USD or None
max_deposit_usd = 10000 # Maximum initial deposit in USD or None

min_duration_h = 10    # Minimum position duration in hours or None
max_duration_h = 100   # Maximum position duration in hours or None
# ----------------------

print("Scenario Parameters Set:")
print(f"- Pool: {target_pool or 'Any'}")
print(f"- Strategy: {target_strategy or 'Any'}")
print(f"- Ticks: {(min_ticks or '')} - {(max_ticks or '')}".strip(' - ') or 'Any')
print(f"- Deposit: ${(min_deposit_usd or '')} - ${(max_deposit_usd or '')}".strip(' $- ') or 'Any')
print(f"- Duration: {(min_duration_h or '')}h - {(max_duration_h or '')}h".strip('h - ') or 'Any')

# %%
# ### 8.2 Filter Data Based on Scenario

print("Filtering data based on scenario...")
if 'df' in locals():
    filtered_df = df.copy()

    # Apply filters step-by-step
    if target_pool is not None and 'pool_symbol' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['pool_symbol'] == target_pool]
        print(f"  - Applied pool filter. Rows remaining: {len(filtered_df)}")

    if target_strategy is not None and 'strategy_type' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['strategy_type'] == target_strategy]
        print(f"  - Applied strategy filter. Rows remaining: {len(filtered_df)}")

    if min_ticks is not None and 'ticks' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['ticks'] >= min_ticks]
        print(f"  - Applied min_ticks filter. Rows remaining: {len(filtered_df)}")
    if max_ticks is not None and 'ticks' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['ticks'] <= max_ticks]
        print(f"  - Applied max_ticks filter. Rows remaining: {len(filtered_df)}")

    if min_deposit_usd is not None and 'total_deposited_usd' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['total_deposited_usd'] >= min_deposit_usd]
        print(f"  - Applied min_deposit filter. Rows remaining: {len(filtered_df)}")
    if max_deposit_usd is not None and 'total_deposited_usd' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['total_deposited_usd'] <= max_deposit_usd]
        print(f"  - Applied max_deposit filter. Rows remaining: {len(filtered_df)}")

    if min_duration_h is not None and 'duration_h' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['duration_h'] >= min_duration_h]
        print(f"  - Applied min_duration filter. Rows remaining: {len(filtered_df)}")
    if max_duration_h is not None and 'duration_h' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['duration_h'] <= max_duration_h]
        print(f"  - Applied max_duration filter. Rows remaining: {len(filtered_df)}")

    print(f"\nFiltering complete. Found {len(filtered_df)} historical positions matching the criteria.")

else:
    print("DataFrame 'df' not loaded. Cannot filter.")
    filtered_df = None # Ensure variable exists but is None

# %%
# ### 8.3 Calculate Average Performance for Matched Positions

print("Calculating average performance for matched historical positions...")
if filtered_df is not None and not filtered_df.empty:
    avg_profit = filtered_df['total_profit_usd'].mean()
    avg_stored_apr = filtered_df['apr'].mean()
    avg_calculated_apr = filtered_df['calculated_apr'].mean()

    print("\n--- Average Historical Performance (Matched Positions) ---")
    print(f"Number of Matched Positions: {len(filtered_df)}")
    print(f"Average Total Profit (USD): ${avg_profit:,.2f}")
    print(f"Average Stored APR: {avg_stored_apr:.2f}%")
    print(f"Average Calculated APR: {avg_calculated_apr:.2f}% (Note: may differ from stored)")
    print("--------------------------------------------------------")

    # Display a sample of the matched positions
    print("\nSample of matched positions:")
    display(filtered_df.head())

elif filtered_df is not None:
    print("\nNo historical positions in the dataset match the specified criteria.")
else:
    print("Cannot calculate performance, data was not filtered.")

# %%
# ### 8.4 Visualize Matched Positions' Profit Distribution

if filtered_df is not None and not filtered_df.empty:
    plt.figure(figsize=(10, 6))
    sns.histplot(filtered_df['total_profit_usd'], kde=True, bins=max(10, len(filtered_df)//5)) # Adjust bins based on count
    plt.title(f'Profit Distribution for {len(filtered_df)} Matched Historical Positions')
    plt.xlabel('Total Profit (USD)')
    plt.ylabel('Number of Positions')
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'nb_scenario_profit_dist.png'))
    plt.show()
elif filtered_df is not None:
    print("\nSkipping visualization: No matched positions found.")
else:
    print("Skipping visualization: Data not available.")

# %% [markdown]
# ## 9. End of Notebook
# You can add more cells below to perform further custom analysis on the `df` or `filtered_df` DataFrames.

print("Notebook script finished. You can now interact with the 'df' and 'filtered_df' DataFrames.")

# By default, the notebook script (`positions_analysis_notebook.py`) is configured to load and analyze the `top_1000_profitable.tsv` subset for faster initial exploration.
# You can modify the `INPUT_FILE` variable within the script (Cell 1) to point directly to `position_stats_all.tsv` if you wish to analyze the complete dataset,
# but be aware this may require significantly more memory and processing time. 