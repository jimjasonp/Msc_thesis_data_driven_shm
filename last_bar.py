def extract_metadata(filename):
    import re
    """Extract noise, transformation, and number of points from filename."""
    filename = filename.lower()
    noise_match = re.search(r'noise(\d+)', filename)
    n_points_match = re.search(r'n(\d+)', filename)
    transform_match = re.search(r'(rotation|scaling|translation)', filename)

    noise = noise_match.group(1) if noise_match else "Unknown"
    n_points = n_points_match.group(1) if n_points_match else "Unknown"
    transform = transform_match.group(1) if transform_match else "Unknown"

    return noise, n_points, transform

def determine_problem_type(filename):
    """Determine if the file is for classification or regression."""
    if 'regression' in filename.lower():
        return 'regression', 'mean_mape'
    elif 'classification' in filename.lower():
        return 'classification', 'mean_acc'
    else:
        raise ValueError("Filename must contain either 'regression' or 'classification'.")

def shorten_model_name(model_name):
    """Shorten specific model names for display."""
    if model_name == "RandomForest":
        return "RF"
    elif model_name == "LinearRegression":
        return "LR"
    return model_name

def plot_all_algorithms(csv_path, font_size=12):
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import os
    # Load data and extract metadata
    df = pd.read_csv(csv_path)
    filename = os.path.basename(csv_path)
    problem_type, metric_col = determine_problem_type(filename)
    noise_file, n_points, _ = extract_metadata(filename)

    # Identify unique values
    algorithms = df['model'].unique()
    short_names = [shorten_model_name(algo) for algo in algorithms]
    data_percentages = sorted(df['data_percentage'].unique())
    transformations = df['transformation'].unique()
    noise_levels = df['noise_percent'].unique()

    colors = ['skyblue', 'orange', 'green']

    # Generate plots
    for transform in transformations:
        for noise_value in noise_levels:
            df_filtered = df[
                (df['transformation'] == transform) &
                (df['noise_percent'] == noise_value)
            ]

            if df_filtered.empty:
                continue

            fig, ax = plt.subplots(figsize=(10, 6))
            bar_width = 0.2
            x = np.arange(len(algorithms))

            for i, data_pct in enumerate(data_percentages):
                df_pct = df_filtered[df_filtered['data_percentage'] == data_pct]
                heights = []
                for algo in algorithms:
                    value = df_pct[df_pct['model'] == algo][metric_col]
                    heights.append(value.values[0] if not value.empty else 0)
                positions = x + (i - 1) * bar_width
                ax.bar(positions, heights, width=bar_width, label=f'{data_pct}%', color=colors[i])

            ax.set_xticks(x)
            ax.set_xticklabels(short_names, fontsize=font_size, rotation=45)
            if metric_col.upper() =='MEAN_MAPE':
                ax.set_ylabel('MAPE', fontsize=font_size)
            else:
                ax.set_ylabel('Accuracy', fontsize=font_size)
            ax.set_title(
                f"{problem_type.capitalize()} | Noise: {noise_value}% | Transform: {transform} |",
                fontsize=font_size + 2
            )
            ax.legend(title='Data percentage', fontsize=font_size, title_fontsize=font_size)
            ax.tick_params(axis='y', labelsize=font_size)

            plt.tight_layout()
            if noise_value == 10:
                plt.show()

# Example usage
font_size = 16

plot_all_algorithms(r"C:\Users\jimja\Desktop\regression_results_n750.csv", font_size=font_size)
plot_all_algorithms(r"C:\Users\jimja\Desktop\classification_results_n750.csv", font_size=font_size)
