import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# --- Configuration ---
INPUT_FILE = 'top_100_profitable.tsv'
OUTPUT_DIR = 'charts'
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
    # Identify potential numeric columns (adjust based on actual data inspection)
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
            # Attempt to convert, coercing errors to NaN
            # Handle potential commas as thousands separators if needed
            df[col] = df[col].astype(str).str.replace(',', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"Warning: Expected numeric column '{col}' not found.")

    # Drop rows where essential numeric values are missing after conversion
    essential_cols = ['total_profit_usd', 'total_deposited_usd', 'duration_h', 'apr', 'ticks', 'in_range_percent', 'auto_rebalance']
    df.dropna(subset=[col for col in essential_cols if col in df.columns], inplace=True)
    print(f"Rows after dropping NaNs in essential columns: {len(df)}")

    # --- Calculations ---
    print("Performing calculations...")
    # Calculate APR only if deposited > 0 and duration > 0
    df['calculated_apr'] = np.where(
        (df['total_deposited_usd'] > 0) & (df['duration_h'] > 0),
        (df['total_profit_usd'] / df['total_deposited_usd']) * (HOURS_PER_YEAR / df['duration_h']) * 100,
        0 # Assign 0 if calculation is not possible
    )
    # Cap calculated APR for visualization purposes if extreme values exist
    df['calculated_apr_capped'] = df['calculated_apr'].clip(upper=df['calculated_apr'].quantile(0.99)) # Cap at 99th percentile


    # Convert auto_rebalance to more readable labels
    if 'auto_rebalance' in df.columns:
         df['strategy_type'] = df['auto_rebalance'].map({0: 'Manual', 1: 'VFAT Auto'})
    else:
         df['strategy_type'] = 'Unknown' # Handle case where column is missing


    print("Data preparation complete.")

    # --- Visualizations ---
    print("Generating visualizations...")
    sns.set_theme(style="whitegrid")

    # 1. Profit Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['total_profit_usd'], kde=True, bins=20)
    plt.title('Distribution of Total Profit (Top 100 Positions)')
    plt.xlabel('Total Profit (USD)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '1_profit_distribution.png'))
    plt.close()
    print("Saved: 1_profit_distribution.png")

    # 2. Profit by Pool
    if 'pool_symbol' in df.columns:
        plt.figure(figsize=(12, 8))
        profit_by_pool = df.groupby('pool_symbol')['total_profit_usd'].sum().sort_values(ascending=False)
        sns.barplot(x=profit_by_pool.values, y=profit_by_pool.index, palette="viridis")
        plt.title('Total Profit by Pool Symbol (Top 100 Positions)')
        plt.xlabel('Total Profit (USD)')
        plt.ylabel('Pool Symbol')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '2_profit_by_pool.png'))
        plt.close()
        print("Saved: 2_profit_by_pool.png")
    else:
        print("Skipping Profit by Pool: 'pool_symbol' column not found.")


    # 3. In-Range Percentage Distribution
    if 'in_range_percent' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df['in_range_percent'], kde=True, bins=10)
        plt.title('Distribution of Time In Range (%) (Top 100 Positions)')
        plt.xlabel('In Range Percentage')
        plt.ylabel('Frequency')
        plt.xlim(0, 100)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '3_in_range_distribution.png'))
        plt.close()
        print("Saved: 3_in_range_distribution.png")
    else:
         print("Skipping In-Range Distribution: 'in_range_percent' column not found.")


    # 4. Profit by Auto-Rebalance Strategy
    if 'strategy_type' in df.columns and df['strategy_type'].nunique() > 1:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='strategy_type', y='total_profit_usd', data=df, palette="coolwarm")
        plt.title('Total Profit by Rebalancing Strategy (Top 100 Positions)')
        plt.xlabel('Strategy Type')
        plt.ylabel('Total Profit (USD)')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '4_profit_by_strategy.png'))
        plt.close()
        print("Saved: 4_profit_by_strategy.png")
    else:
        print("Skipping Profit by Strategy: Not enough data or 'strategy_type' column missing.")

    # 5. APR (Stored) Distribution
    if 'apr' in df.columns:
        # Filter out extreme APR values for better visualization if necessary
        apr_filtered = df[df['apr'] < df['apr'].quantile(0.98)] # Exclude top 2% for visualization
        plt.figure(figsize=(10, 6))
        sns.histplot(apr_filtered['apr'], kde=True, bins=20)
        plt.title('Distribution of Stored APR (Top 100 Positions, capped)')
        plt.xlabel('APR (%)')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '5_apr_distribution.png'))
        plt.close()
        print("Saved: 5_apr_distribution.png")
    else:
        print("Skipping APR Distribution: 'apr' column not found.")

    # 6. Profit vs. Duration
    if 'duration_h' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='duration_h', y='total_profit_usd', data=df, hue='strategy_type' if 'strategy_type' in df.columns and df['strategy_type'].nunique() > 1 else None, palette='Set2', alpha=0.7)
        plt.title('Total Profit vs. Position Duration (Top 100)')
        plt.xlabel('Duration (Hours)')
        plt.ylabel('Total Profit (USD)')
        plt.xscale('log') # Use log scale if duration varies widely
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '6_profit_vs_duration.png'))
        plt.close()
        print("Saved: 6_profit_vs_duration.png")
    else:
        print("Skipping Profit vs Duration: 'duration_h' column not found.")

    # 7. Profit vs. Deposited USD
    if 'total_deposited_usd' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='total_deposited_usd', y='total_profit_usd', data=df, hue='strategy_type' if 'strategy_type' in df.columns and df['strategy_type'].nunique() > 1 else None, palette='Set2', alpha=0.7)
        plt.title('Total Profit vs. Total Deposited USD (Top 100)')
        plt.xlabel('Total Deposited (USD)')
        plt.ylabel('Total Profit (USD)')
        plt.xscale('log') # Log scale likely needed
        plt.yscale('log') # Log scale might be useful too
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '7_profit_vs_deposited.png'))
        plt.close()
        print("Saved: 7_profit_vs_deposited.png")
    else:
        print("Skipping Profit vs Deposited: 'total_deposited_usd' column not found.")


    # 8. Profit vs. Range Width (Ticks)
    if 'ticks' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='ticks', y='total_profit_usd', data=df, hue='strategy_type' if 'strategy_type' in df.columns and df['strategy_type'].nunique() > 1 else None, palette='Set2', alpha=0.7)
        plt.title('Total Profit vs. Range Width (Ticks) (Top 100)')
        plt.xlabel('Range Width (Ticks)')
        plt.ylabel('Total Profit (USD)')
        plt.xscale('log') # Log scale likely needed
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '8_profit_vs_range_width.png'))
        plt.close()
        print("Saved: 8_profit_vs_range_width.png")
    else:
        print("Skipping Profit vs Range Width: 'ticks' column not found.")


    print("\nAll visualizations generated successfully in the 'charts' directory.")

except FileNotFoundError:
    print(f"Error: Input file '{INPUT_FILE}' not found.")
except pd.errors.EmptyDataError:
    print(f"Error: Input file '{INPUT_FILE}' is empty.")
except Exception as e:
    print(f"An unexpected error occurred: {e}") 