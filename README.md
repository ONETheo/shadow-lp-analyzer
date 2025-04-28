# Shadow DEX Position Analyzer

## Introduction

This project provides tools and a notebook-style Python script to analyze the performance of liquidity provider (LP) positions on the Shadow decentralized exchange (DEX). The primary focus is on understanding profitability, verifying Annual Percentage Rates (APRs), and exploring the characteristics of successful historical positions.

The main analysis is performed using the `positions_analysis_notebook.py` script, which offers an interactive, cell-by-cell execution flow similar to a Jupyter notebook.

## Data

This repository includes two primary data files:

*   **`position_stats_all.tsv`**: The full dataset containing historical statistics for Shadow DEX positions.
*   **`top_1000_profitable.tsv`**: A subset containing only the top 1000 most profitable positions from the full dataset, generated for convenience.

By default, the notebook script (`positions_analysis_notebook.py`) is configured to load and analyze the `top_1000_profitable.tsv` subset for faster initial exploration. You can modify the `INPUT_FILE` variable within the script (Cell 1) to point directly to `position_stats_all.tsv` if you wish to analyze the complete dataset, but be aware this may require significantly more memory and processing time.

## Setup

1.  **Prerequisites:**
    *   Python (3.8 or later recommended)
    *   `pip` (Python package installer)
    *   `awk` command-line utility (Optional: needed only if you want to generate custom data subsets, see section below. Available by default on Linux/macOS, may need installation or alternative on Windows, e.g., Git Bash).

2.  **Clone the Repository:**
    ```bash
    git clone <repository-url> # Replace with the actual repo URL
    cd shadow-lp-analyzer # Or your chosen repo name
    ```

3.  **(Recommended) Create a Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Generating Custom Data Subsets (Optional)

While this repository includes the full dataset and a pre-generated top 1000 profitable subset, you might want to create your own custom subsets for analysis (e.g., different number of top positions, filtering by specific pools or criteria).

You can use the `awk` command-line tool for this. Here is the example command used to generate the included `top_1000_profitable.tsv`:

```bash
# Example: Generate top 1000 profitable positions
awk 'BEGIN { FS="\t"; OFS="\t"; limit = 1000; min_profit = -1e99; min_idx = 0; count = 0; header = ""; printf "Extracting top %d profitable positions...\n", limit; } NR == 1 { header = $0; next } { profit = $8 + 0; if (count < limit) { count++; top_profits[count] = profit; top_lines[count] = $0; if (count == limit) { min_profit = top_profits[1]; min_idx = 1; for (i = 2; i <= limit; i++) { if (top_profits[i] < min_profit) { min_profit = top_profits[i]; min_idx = i; } } } } else if (profit > min_profit) { top_profits[min_idx] = profit; top_lines[min_idx] = $0; min_profit = top_profits[1]; min_idx = 1; for (i = 2; i <= limit; i++) { if (top_profits[i] < min_profit) { min_profit = top_profits[i]; min_idx = i; } } } } END { printf "Sorting top %d...\n", limit; for (i=1; i<=count; i++) idx[i] = i; for (i = 1; i <= count; i++) { for (j = 1; j <= count - i; j++) { if (top_profits[idx[j]] < top_profits[idx[j+1]]) { temp = idx[j]; idx[j] = idx[j+1]; idx[j+1] = temp; } } } print header > "top_1000_profitable.tsv"; for (i = 1; i <= count; i++) { print top_lines[idx[i]] >> "top_1000_profitable.tsv"; } printf "\nExtraction complete. Top %d profitable positions saved to top_1000_profitable.tsv.\n", count; }' "position_stats_all.tsv"
```

**Modifying the Command:**

*   Change `limit = 1000` to extract a different number of top positions.
*   Change the output filename (`top_1000_profitable.tsv`) to save to a different file.
*   Add filtering conditions before the profit check (e.g., `tolower($NF) ~ /usd.*\/.*usd/ { ... }` to filter for USD-USD pools before finding the top N within that filter).

After generating a custom file, remember to update the `INPUT_FILE` variable in the `positions_analysis_notebook.py` script to point to your new file name before running the notebook analysis.

## Using the Analysis Notebook (`positions_analysis_notebook.py`)

This script allows for interactive analysis of the position data.

**How to Run:**

*   **VS Code (Recommended):** Open the project folder in Visual Studio Code with the Python and Jupyter extensions installed. Open `positions_analysis_notebook.py`. You will see "Run Cell" links above each code block (marked with `# %%`). Run the cells sequentially.
*   **Other Environments:** You can convert the script to a standard `.ipynb` notebook using tools like `jupytext` (`jupytext --to notebook positions_analysis_notebook.py`) or manually copy-paste the cells into a new Jupyter Notebook.

**Notebook Sections:**

1.  **Setup and Imports:** Loads libraries and sets configurations.
2.  **Load Data:** Loads the dataset (defaults to `top_1000_profitable.tsv`).
3.  **Data Cleaning:** Converts columns to appropriate types, handles missing data.
4.  **Calculate APR:** Calculates APR based on profit, deposit, and duration.
5.  **Analyze APR Discrepancies:** Identifies positions where the stored APR differs significantly from the calculated one.
6.  **Basic Statistics:** Displays summary statistics for key columns.
7.  **Visualizations:** Generates various plots (profit distribution, profit by pool, relationships between variables, APR comparison).
8.  **What-If Scenario Analysis:** Allows you to interactively filter the historical data based on parameters (pool, range width, deposit, duration, strategy) and see the *average* performance of matching historical positions. **Remember the disclaimer:** this is an analysis of past top performers, not a prediction of future results.

**Output:**

*   Plots generated by the notebook are saved to the `notebook_charts/` directory.
*   Intermediate data (like `filtered_df` in the scenario analysis) is available in memory for further interactive exploration within your Python environment.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## License

MIT License