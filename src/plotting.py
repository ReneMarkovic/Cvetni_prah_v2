#src/plotting.py
import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from src.utils import path_for_export, save_plot, generate_base_path
from src.data_loader import load_raw_data


def plot_global_data(df_processed, location):
    base_path = generate_base_path(location)
    print(df_processed["Type"].unique())

    # Aggregate by day
    y_day = df_processed.groupby("Date").sum()
    x_day = y_day.index

    #  Aggregate by week
    y_week = df_processed.groupby(pd.Grouper(key='Date', freq='W')).sum()
    x_week = y_week.index

    # Aggregate by month
    y_month = df_processed.groupby(pd.Grouper(key='Date', freq='ME')).sum()
    x_month = y_month.index

    # Aggregate by quarter
    y_quarter = df_processed.groupby(pd.Grouper(key='Date', freq='QE')).sum()
    x_quarter = y_quarter.index

    # Aggregate by year
    y_year = df_processed.groupby(pd.Grouper(key='Date', freq='YE')).sum()
    x_year = y_year.index

    plt.figure(figsize=(20,4))
    plt.plot(x_week,y_week["Value"])
    plt.title("Dnevna koncentracije pelodov")
    plt.xlabel("Datum")
    plt.ylabel("Koncentracija pelodov")
    plt.xlim([datetime.datetime(2002,1,1),datetime.datetime(2023,12,31)])
    plt.xticks(np.arange(datetime.datetime(2002,1,1),datetime.datetime(2023,12,31),step=datetime.timedelta(days=365)))
    plt.xticks(rotation=45,size=12, ha = 'right')
    path_for_fig = os.path.join(base_path,f"{location}_Fig_01a.png")
    plt.tight_layout()
    plt.savefig(path_for_fig, dpi = 150)
    plt.close()

    plt.figure(figsize=(20,4))
    plt.plot(x_day,y_day["Value"])
    plt.title("Dnevna koncentracije pelodov")
    plt.xlabel("Datum")
    plt.ylabel("Koncentracija pelodov")

    year = 2019
    quarterly_ticks = pd.date_range(start=datetime.datetime(year-1, 1, 1), 
                                    end=datetime.datetime(year+1, 12, 31), 
                                    freq='QE')
    plt.xticks(quarterly_ticks, rotation=45, size=12, ha='right')
    plt.xticks(rotation=45,size=12, ha = 'right')
    plt.xlim([datetime.datetime(year-2,12,31),datetime.datetime(year+1,12,31)])
    path_for_fig = os.path.join(base_path,f"{location}_Fig_01b.png")
    plt.tight_layout()
    plt.savefig(path_for_fig, dpi = 150)
    plt.close()
    
    plt.figure(figsize=(20, 6))

    # Group the data by 'Date' and 'Type' to create the stacked bar plot data
    df_grouped = df_processed.groupby(['Date', 'Type'])['Value'].sum().unstack().fillna(0)

    # Filter the data to include only the dates within the desired range
    df_grouped_plot = df_grouped.loc[(df_grouped.index >= datetime.datetime(year - 1, 1, 1)) & 
                                    (df_grouped.index <= datetime.datetime(year + 1, 12, 31))]

    # Plot the stacked bar chart
    ax = df_grouped_plot.plot(kind='bar', stacked=True, figsize=(20, 6), width=1)

    # Set title and labels
    plt.title('Prispevek posameznih tipov vegetacije')
    plt.xlabel('Datum')
    plt.ylabel('Koncentracija pelodov')

    # Generate quarterly x-ticks based on the actual date range from the plot
    quarterly_ticks = pd.date_range(start=datetime.datetime(year-1, 1, 1), 
                                    end=datetime.datetime(year+1, 12, 31), 
                                    freq='QE')

    # Adjust the x-ticks to match the actual number of bars
    ax.set_xticks(np.linspace(0, len(df_grouped_plot.index) - 1, len(quarterly_ticks)))

    # Set the x-tick labels using the quarterly date range
    ax.set_xticklabels([tick.strftime('%Y-%m-%d') for tick in quarterly_ticks], rotation=45, size=10, ha='right')

    # Add a grid for the y-axis
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Position the legend outside the plot
    plt.legend(title='Tip vegetacije', bbox_to_anchor=(1.05, 1), loc='upper left')


    path_for_fig = os.path.join(base_path,f"{location}_Fig_01c.png")
    plt.tight_layout()
    plt.savefig(path_for_fig, dpi = 150)
    plt.close()

def plot_completeness_analysis(location):
    """Generate and save completeness analysis by year and type for the provided location."""
    
    # Define the base path for saving plots
    base_path = generate_base_path(location)

    # Load the raw data
    path_load = os.path.join("data","raw",f"{location}2024.xlsx")
    if os.path.exists(path_load):
        df_raw = load_raw_data(path_load)
    else:
        path_load = os.path.join("data","raw", f"{location}2024.xlsx")
        if os.path.exists(path_load):
            df_raw = load_raw_data(path_load)
        else:
            raise ValueError(f"Data file not found for location '{location}'.")

    # Ensure 'Date' is of datetime type and add 'Year' and 'Month' columns
    for data_value in df_raw["Date"].values:
        if isinstance(data_value, (np.datetime64, pd.Timestamp)):
            pass
        else:
            print(f"Invalid date format: {data_value}")
    df_raw["Year"] = df_raw["Date"].dt.year
    df_raw["Month"] = df_raw["Date"].dt.month

    # Completeness by Year and Type
    dg = df_raw.groupby(["Type", "Year"])
    completeness = {"Year": [], "Type": [], "N_nan": [], "N_all": [], "Percentage": []}

    for (data_type, year), data_group in dg:
        N_nan = data_group["Value"].isna().sum()
        N_all = len(data_group)
        completeness["Year"].append(year)
        completeness["Type"].append(data_type)
        completeness["N_nan"].append(N_nan)
        completeness["N_all"].append(N_all)
        completeness["Percentage"].append((1.0 - N_nan / N_all) * 100)

    # Create DataFrame for Completeness Analysis
    df_completeness = pd.DataFrame(completeness)

    # Pivot Table to Show Completeness by Year and Type
    completeness_pivot = df_completeness.pivot(index="Year", columns="Type", values="Percentage").sort_index(ascending=False)
    avg_completeness = completeness_pivot.mean(axis=1).sort_index(ascending=True)

    # Plot Completeness Heatmap and Average Completeness Bar Chart
    fig, ax = plt.subplots(ncols=2, figsize=(12, 6.5), gridspec_kw={'width_ratios': [4, 1]}, dpi=150, facecolor='lightblue')
    sns.heatmap(completeness_pivot, ax=ax[0], cbar=True, cmap="viridis", annot=True, fmt=".1f", annot_kws={"size": 8})
    ax[0].set_xlabel("Vrsta")
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45, horizontalalignment='right')
    ax[0].set_ylabel("Leto")
    ax[0].set_title("Popolnost podatkov glede na vrsto in leto (%)")

    avg_completeness.plot(kind='barh', ax=ax[1], color='skyblue')
    ax[1].set_xlim(50, 100)
    ax[1].grid(axis='x')
    ax[1].set_xlabel("Popolnost podatkov (%)")
    ax[1].set_ylabel("Leto")
    ax[1].set_title("Povprečna popolnost podatkov")

    plt.tight_layout()
    save_plot(base_path, location, "Letna_popolnost_podatkov")

    # Monthly Completeness Heatmap
    df_raw["Year_int"] = df_raw["Year"] - df_raw["Year"].min()
    Ny = int(df_raw["Year"].max() - df_raw["Year"].min() + 1)
    Nx = int(12)
    data = np.zeros((Ny, Nx))

    # Group by Year and Month
    dg_month = df_raw.groupby(["Year_int", "Month"])
    for (year_int, month), data_group in dg_month:
        N_all = len(data_group)
        N_nan = data_group["Value"].isna().sum()
        data[int(year_int), int(month - 1)] = 1.0 - N_nan / N_all

    # Plot Monthly Completeness Heatmap
    cmap = ListedColormap(sns.color_palette("RdBu", 10).as_hex()[::-1])
    fig, ax = plt.subplots(figsize=(9, 8), dpi=150, facecolor='lightblue')
    sns.heatmap(data, ax=ax, cbar=True, cmap=cmap, annot=True, fmt=".1f", annot_kws={"size": 8}, linecolor='white', linewidths=0.5)
    ax.set_xlabel("Mesec")
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "Maj", "Jun", "Jul", "Avg", "Sep", "Okt", "Nov", "Dec"], rotation=45, horizontalalignment='right')
    ax.set_ylabel("Leto")
    ax.set_yticklabels(range(int(df_raw["Year"].min()), int(df_raw["Year"].max()) + 1), rotation=0)
    ax.invert_yaxis()
    ax.set_title("Popolnost podatkov glede na mesec in leto (%)")

    plt.tight_layout()
    save_plot(base_path, location, "Mesečna_popolnost_podatkov")

def plot_auc_and_ci(results, colors, location):
    df_res_2 = pd.DataFrame(results)
    ##----------------------------------- Season start ------------------------------------------##
    fig, ax = plt.subplots(ncols=3,
                        nrows=1,
                        figsize=(12, 4),
                        gridspec_kw={'width_ratios': [1, 1, 1]},
                        dpi=150,
                        facecolor='lightblue')
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    var = "K10"
    mean_values = df_res_2.groupby('Type')[var].mean().sort_values()
    iterate = mean_values.index.values
    sns.boxplot(data = df_res_2,x = "Type",y = var, order=mean_values.index,fliersize=0,ax = ax[0])
    for i,tip in enumerate(iterate):
        dfs = df_res_2[df_res_2["Type"]==tip].sort_values("Year")[var].values
        x = df_res_2[df_res_2["Type"]==tip].sort_values("Year")["Year"].values   
        color_pallete = [colors[leto] for leto in x]
        ax[0].scatter([i for leto in x],dfs,color=color_pallete,s=5)
    ax[0].set_xlabel("Vrsta")
    ax[0].set_ylabel("Pričetek sezone K10")
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45, horizontalalignment='right',fontsize=6)
    ##----------------------------------- Season End ------------------------------------------##
    var = "K90"
    iterate = mean_values.index.values
    sns.boxplot(data = df_res_2,x = "Type",y = var, order=mean_values.index,fliersize=0,ax = ax[1])
    for i,tip in enumerate(iterate):
        dfs = df_res_2[df_res_2["Type"]==tip].sort_values("Year")[var].values
        x = df_res_2[df_res_2["Type"]==tip].sort_values("Year")["Year"].values   
        color_pallete = [colors[leto] for leto in x]
        ax[1].scatter([i for leto in x],dfs,color=color_pallete,s=5)

    ax[1].set_xlabel("Vrsta")
    ax[1].set_ylabel("Konec sezone K90")
    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45, horizontalalignment='right',fontsize=6)


    ##----------------------------------- Order of change------------------------------------------##
    var = "10-90 interval"
    iterate = mean_values.index.values
    mean_K10 = df_res_2.groupby('Type')[var].mean().sort_values().to_dict()
    print(mean_K10)
    for i,tip in enumerate(iterate):
        dfs = df_res_2[df_res_2["Type"]==tip].sort_values("Year")[var].values
        dfs = dfs - mean_K10[tip]
        x = df_res_2[df_res_2["Type"]==tip].sort_values("Year")["Year"].values
        ax[2].scatter([i for leto in x],dfs,color=color_pallete,s=5,alpha = 0.3)
    ax[2].set_xlabel("Vrsta")
    ax[2].set_ylabel("Konec sezone 10-90 interval")
    ax[2].set_xticklabels(ax[2].get_xticklabels(), rotation=45, horizontalalignment='right',fontsize=6)
    #ax[2].set_ylim(-50,50)
    base_path = path_for_export(lv1 = "Graphs",lv2 = "Cvetni_prah",lv3 = location)
    file_path = path_for_export(lv1=base_path, name=f"AUC_CI_90_{location}a.png")
    plt.savefig(file_path, dpi=150)
    plt.close()

    ##----------------------------------- Rate ------------------------------------------------##
    plt.figure(figsize=(8,4),dpi=150, facecolor='lightblue')
    plt.title("Rate of change")
    mean_values = df_res_2.groupby('Type')['rate'].mean().sort_values()
    iterate = mean_values.index.values
    sns.boxplot(data = df_res_2,x = "Type",y = "rate", order=mean_values.index,fliersize=0)
    for i,tip in enumerate(iterate):
        dfs = df_res_2[df_res_2["Type"]==tip].sort_values("Year")["rate"].values
        x = df_res_2[df_res_2["Type"]==tip].sort_values("Year")["Year"].values   
        color_pallete = [colors[leto] for leto in x]
        plt.scatter([i for leto in x],dfs,color=color_pallete,s=5)
    plt.xticks(rotation=45, ha ="right")
    base_path = path_for_export(lv1 = "Graphs",lv2 = "Cvetni_prah",lv3 = location)
    file_path = path_for_export(lv1=base_path, name=f"AUC_CI_90_{location}b.png")
    plt.tight_layout()
    plt.savefig(file_path, dpi=150)
    plt.close()