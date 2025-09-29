import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from src.utils import path_for_export, save_plot, generate_base_path
from src.utils import moving_average, path_for_export
import seaborn as sns

def perform_cross_regional_correlation(df_list, locations):
    """
    Performs a cross-regional correlation analysis for each pollen type.
    
    Args:
        df_list (list): A list of pandas DataFrames, one for each location.
        locations (list): A list of strings with location names.
    
    Returns:
        dict: A dictionary of correlation DataFrames, one for each pollen type.
    """
    
    # Check if there's enough data for correlation
    if len(df_list) < 2:
        print("Potrebujete podatke vsaj dveh regij za korelacijsko analizo.")
        return {}
    
    # Get all unique pollen types across all dataframes by flattening the list of unique arrays
    all_types = sorted(list(set(item for df in df_list for item in df["Type"].unique())))
    
    correlation_results = {}
    
    print("\nIzvajanje korelacijske analize med regijami po vrsti cvetnega prahu...")
    
    for pollen_type in all_types:
        try:
            # Create a dictionary to hold the time series for the current pollen type from each location
            data_to_correlate = {}
            for i, loc in enumerate(locations):
                # Filter data for the current pollen type and set the date as index
                df_loc = df_list[i][df_list[i]["Type"] == pollen_type].copy()
                if not df_loc.empty:
                    df_loc.set_index("Date", inplace=True)
                    data_to_correlate[loc] = df_loc["Value"]
            
            # If we have data for at least two locations, create a DataFrame and compute the correlation
            if len(data_to_correlate) >= 2:
                df_combined = pd.DataFrame(data_to_correlate)
                correlation_matrix = df_combined.corr()
                correlation_results[pollen_type] = {
                    "correlation_matrix": correlation_matrix,
                    "combined_data": df_combined
                }
                print(f"  Izračunana korelacija za vrsto: {pollen_type}")
            else:
                print(f"  Preskok vrste '{pollen_type}': Podatki na voljo samo za eno ali nobeno regijo.")
        except Exception as e:
            print(f"  Napaka pri izračunu korelacije za vrsto '{pollen_type}': {e}")
            
    return correlation_results

def determine_season_by_reference_year(df: pd.DataFrame, location: str, pollen_type: str, reference_year: int, start_th = 0.025, end_th = 0.975):

    # Filter data for the specified location and pollen type
    df_filtered = df[(df["Type"] == pollen_type)].copy()
    print(f"    - Processing pollen type: {pollen_type} for location: {location}")
    if df_filtered.empty:
        print(f"Napaka: Podatki za vrsto '{pollen_type}' niso na voljo.")
        return None

    # Calculate the total annual sum for the reference year
    df_ref_year = df_filtered[df_filtered["Year"] == reference_year]
    print(f"    - Reference year: {reference_year}")
    if df_ref_year.empty:
        print(f"Napaka: Referenčno leto {reference_year} ni najdeno za vrsto '{pollen_type}'.")
        return None
        
    reference_total_sum = df_ref_year["Value"].apply(np.nansum).sum()
    print(f"    - Total pollen sum for reference year {reference_year}: {reference_total_sum}")
    if reference_total_sum == 0:
        print(f"Napaka: Vsota cvetnega prahu za referenčno leto {reference_year} je 0. Ne morem izračunati pragov.")
        return None

    # Calculate the start and end thresholds
    threshold_start_percent = start_th * reference_total_sum
    threshold_end_percent = end_th * reference_total_sum
    return {"Reference year": reference_year, "Total Sum": float(reference_total_sum), "Threshold Start": float(threshold_start_percent), "Threshold End": float(threshold_end_percent)}

def activation_reference(df: pd.DataFrame, location:str, reference_year: int = 2004):
    pollen_types = df["Type"].unique()
    pollen_type_season_reference = {}
    for pollen_type in pollen_types:
        print("    - Determining season thresholds for pollen type:", pollen_type)
        result = determine_season_by_reference_year(df, location, pollen_type, reference_year)
        print(result)
        if result is not None:
            pollen_type_season_reference[pollen_type] = result
    file_path = generate_base_path(location)
    #save to json file
    with open(os.path.join(file_path, "season_reference.json"), 'w') as f:
        json.dump(pollen_type_season_reference, f, indent=4)
    print(f"    - Season reference data saved to '{file_path}'")
    return pollen_type_season_reference

def plot_daily_values(ax, x, list_y, mean_y, colors, years, lw_1, lw_2, title):
    for idx, y in enumerate(list_y):
        ax.plot(x, y, label=f"{years[idx]}", alpha=0.7, color=colors[years[idx]], linewidth=lw_1)
    ax.plot(x, mean_y, label="Povprečje", c="black", linewidth=lw_2, zorder=3)
    ax.set_title(title, fontsize=15)
    ax.set_ylabel("Količina cvetnega prahu [enote]", fontsize=13)
    ax.set_xlabel("Dan v letu", fontsize=13)
    ax.grid(True, linestyle='--', alpha=0.5)

def plot_aggregated_yearly(ax, years, list_y, colors, lw_1, lw_2, title):
    """
    Za vsak vnos v list_y (vsako leto) nariši agregirano (kumulativno) vsoto skozi leto.
    Prikaži tudi povprečje kot debelejšo črto.
    """
    cum_sums = []
    for idx, y in enumerate(list_y):
        cum_sum = np.nancumsum(y)
        cum_sums.append(cum_sum)
        ax.plot(range(len(cum_sum)), cum_sum, label=f"{years[idx]}", color=colors[years[idx]], linewidth=lw_1, alpha=0.7)
    # Povprečna kumulativna vsota skozi leto (črna črta)
    mean_cum = np.nanmean(np.array(cum_sums), axis=0)
    ax.plot(range(len(mean_cum)), mean_cum, color="black", linewidth=lw_2, label="Povprečje", zorder=3)
    ax.set_title(title, fontsize=15)
    ax.set_ylabel("Agregirana vsota [enote]", fontsize=13)
    ax.set_xlabel("Dan v letu", fontsize=13)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=8, ncol=2, loc='upper left')

def plot_normalized_yearly(ax, years, list_y, colors, reference_total, lw_1, lw_2, title, reference_year):
    """
    Za vsak vnos v list_y (vsako leto) nariši normirano kumulativno vsoto skozi dni v letu.
    Povprečje podaj kot črno črto.
    """
    norm_cumsums = []
    starts, ends, k50s = [], [], []  # Za shranjevanje indeksov

    for idx, y in enumerate(list_y):
        cum_sum = np.nancumsum(y)
        norm_cum = cum_sum / reference_total if reference_total > 0 else np.zeros_like(cum_sum)
        norm_cumsums.append(norm_cum)
        ax.plot(range(len(norm_cum)), norm_cum, label=f"{years[idx]}", color=colors[years[idx]], linewidth=lw_1, alpha=0.7)
        # Izračun začetek/konec/K50
        # Začetek: ko kumulativna vsota preseže 5% normirane vsote
        start_idx = np.where(norm_cum > 0.05)[0]
        starts.append(start_idx[0] if len(start_idx) > 0 else np.nan)
        # Konec: ko kumulativna vsota preseže 97.5% normirane vsote
        end_idx = np.where(norm_cum > 0.975)[0]
        ends.append(end_idx[0] if len(end_idx) > 0 else np.nan)
        # K50: ko kumulativna vsota preseže 50% normirane vsote
        k50_idx = np.where(norm_cum > 0.5)[0]
        k50s.append(k50_idx[0] if len(k50_idx) > 0 else np.nan)

    # Povprečje normirane kumulativne vsote
    mean_norm_cum = np.nanmean(np.array(norm_cumsums), axis=0)
    ax.plot(range(len(mean_norm_cum)), mean_norm_cum, color="black", linewidth=lw_2, label="Povprečje", zorder=3)
    ax.set_title(f"{title} (referenčno leto {reference_year})", fontsize=15)
    ax.set_ylabel("Normirana agregirana vsota", fontsize=13)
    ax.set_xlabel("Dan v letu", fontsize=13)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=8, ncol=2, loc='upper left')

    return starts, ends, k50s

def plot_season_start(ax, years, starts, colors, lw_1, title):
    # scatter in črta
    ax.scatter(years, starts, color=[colors[y] for y in years], s=80, label="Začetek sezone")
    #ax.plot(years, starts, color="black", linewidth=lw_1, alpha=0.6)

    # Linearni fit (trend)
    fit_mask = ~np.isnan(starts)
    fit_years = np.array(years)[fit_mask]
    fit_starts = np.array(starts)[fit_mask]
    if len(fit_years) > 1:
        m, b = np.polyfit(fit_years, fit_starts, 1)
        trend = m * fit_years + b
        r2 = np.corrcoef(fit_years, fit_starts)[0, 1] ** 2
        ax.plot(fit_years, trend, color="crimson", linestyle="--", linewidth=2, label="Trend")
        # Pripis v graf
        ax.text(0.05, 0.90, fr"Trend: $y = {m:.2f}x + {b:.1f}$", transform=ax.transAxes, fontsize=12)
        ax.text(0.05, 0.80, fr"$R^2 = {r2:.3f}$", transform=ax.transAxes, fontsize=12)
    ax.set_title(title, fontsize=15)
    ax.set_ylabel("Dan v letu (začetek)", fontsize=13)
    ax.set_xlabel("Leto", fontsize=13)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9, loc='best')

def plot_season_end(ax, years, ends, colors, lw_1, title):
    """
    Prikaz konca sezone (npr. K90) skozi leta z linearno fit premico in R².
    """
    ax.scatter(years, ends, color=[colors[y] for y in years], s=80, label="Konec sezone")
    #ax.plot(years, ends, color="black", linewidth=lw_1, alpha=0.6)

    # Linearni trend (fit)
    fit_mask = ~np.isnan(ends)
    fit_years = np.array(years)[fit_mask]
    fit_ends = np.array(ends)[fit_mask]
    if len(fit_years) > 1:
        m, b = np.polyfit(fit_years, fit_ends, 1)
        trend = m * fit_years + b
        r2 = np.corrcoef(fit_years, fit_ends)[0, 1] ** 2
        ax.plot(fit_years, trend, color="crimson", linestyle="--", linewidth=2, label="Trend")
        ax.text(0.05, 0.90, fr"Trend: $y = {m:.2f}x + {b:.1f}$", transform=ax.transAxes, fontsize=12)
        ax.text(0.05, 0.80, fr"$R^2 = {r2:.3f}$", transform=ax.transAxes, fontsize=12)

    ax.set_title(title, fontsize=15)
    ax.set_ylabel("Dan v letu (konec)", fontsize=13)
    ax.set_xlabel("Leto", fontsize=13)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9, loc='best')

def plot_yearly_concentration(ax, years, total_yearly, colors, lw_1, title):
    """
    Prikaz letne koncentracije skozi leta z linearno fit premico in R².
    """
    ax.scatter(years, total_yearly, color=[colors[y] for y in years], s=80, label="Letna koncentracija")
    #ax.plot(years, total_yearly, color="black", linewidth=lw_1, alpha=0.6)

    # Linearni trend (fit)
    fit_mask = ~np.isnan(total_yearly)
    fit_years = np.array(years)[fit_mask]
    fit_totals = np.array(total_yearly)[fit_mask]
    if len(fit_years) > 1:
        m, b = np.polyfit(fit_years, fit_totals, 1)
        trend = m * fit_years + b
        r2 = np.corrcoef(fit_years, fit_totals)[0, 1] ** 2
        ax.plot(fit_years, trend, color="crimson", linestyle="--", linewidth=2, label="Trend")
        ax.text(0.05, 0.90, fr"Trend: $y = {m:.2f}x + {b:.1f}$", transform=ax.transAxes, fontsize=12)
        ax.text(0.05, 0.80, fr"$R^2 = {r2:.3f}$", transform=ax.transAxes, fontsize=12)

    ax.set_title(title, fontsize=15)
    ax.set_ylabel("Letna vsota [enote]", fontsize=13)
    ax.set_xlabel("Leto", fontsize=13)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9, loc='best')

def type_specific_activation(df: pd.DataFrame, location:str, ma:int = 7, step_name:str = "default", reference_year: int = 2004):
    cmap = plt.get_cmap("viridis")  # ali "plasma", "cividis", "coolwarm", "RdYlBu"
    dg = df.groupby("Type")
    print("    - Generating activation reference...")
    act_reference = activation_reference(df, location=location, reference_year=reference_year)
    print(act_reference)
    all_years = sorted(df["Year"].unique())
    colors = {year: cmap(i / (len(all_years) - 1)) for i, year in enumerate(all_years)}

    plt.rcParams.update({
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'axes.edgecolor': 'grey',
        'grid.color': 'lightgrey',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 11,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
    })

    lw_1 = 1.2
    lw_2 = 2.5
    w = ma

    results = {}  # <--- ADD THIS, so results exist before loop!

    for i, ii in dg:
        base_path = generate_base_path(location)
        output_path = path_for_export(lv1=base_path, lv2=step_name)

        list_y = []
        years = []
        for j, jj in ii.groupby("Year"):
            jj = jj.sort_values("Date")
            y = jj["Value"].values[:365]
            list_y.append(y)
            years.append(j)

        x_standard = list(range(365))
        mean_y = np.nanmean(list_y, axis=0)
        total_yearly = [np.nansum(y) for y in list_y]
        reference_total = act_reference[i]["Total Sum"] if i in act_reference else 1.0

        # --- Calculate starts, ends, k50 for each year ---
        cum_sums = [np.nancumsum(y) for y in list_y]
        norm_cumsums = [cum_sum / reference_total if reference_total > 0 else np.zeros_like(cum_sum) for cum_sum in cum_sums]
        starts = []
        ends = []
        k50s = []
        for norm_cum in norm_cumsums:
            start_idx = np.where(norm_cum > 0.05)[0]
            starts.append(start_idx[0] if len(start_idx) > 0 else np.nan)
            end_idx = np.where(norm_cum > 0.975)[0]
            ends.append(end_idx[0] if len(end_idx) > 0 else np.nan)
            k50_idx = np.where(norm_cum > 0.5)[0]
            k50s.append(k50_idx[0] if len(k50_idx) > 0 else np.nan)

        # --- STATISTICS BLOCK: Compute linear trends for each ---
        results[i] = {}
        # K10 (Start trend)
        fit_mask = ~np.isnan(starts)
        fit_years = np.array(years)[fit_mask]
        fit_starts = np.array(starts)[fit_mask]
        if len(fit_years) > 1:
            m, b = np.polyfit(fit_years, fit_starts, 1)
            r2 = np.corrcoef(fit_years, fit_starts)[0, 1] ** 2
            results[i]["K10"] = {
                "Trend": float(m),
                "Intercept": float(b),
                "R2": float(r2),
                "min(K10)": int(np.nanmin(fit_starts)),
                "avg(K10)": float(np.nanmean(fit_starts)),
                "max(K10)": int(np.nanmax(fit_starts))
            }
        else:
            results[i]["K10"] = {"Trend": np.nan, "Intercept": np.nan, "R2": np.nan, "min(K10)": np.nan, "avg(K10)": np.nan, "max(K10)": np.nan}

        # K90 (End trend)
        fit_mask = ~np.isnan(ends)
        fit_years = np.array(years)[fit_mask]
        fit_ends = np.array(ends)[fit_mask]
        if len(fit_years) > 1:
            m, b = np.polyfit(fit_years, fit_ends, 1)
            r2 = np.corrcoef(fit_years, fit_ends)[0, 1] ** 2
            results[i]["K90"] = {
                "Trend": float(m),
                "Intercept": float(b),
                "R2": float(r2),
                "min(K90)": int(np.nanmin(fit_ends)),
                "avg(K90)": float(np.nanmean(fit_ends)),
                "max(K90)": int(np.nanmax(fit_ends))
            }
        else:
            results[i]["K90"] = {"Trend": np.nan, "Intercept": np.nan, "R2": np.nan, "min(K90)": np.nan, "avg(K90)": np.nan, "max(K90)": np.nan}

        # K50 (mid-season trend)
        fit_mask = ~np.isnan(k50s)
        fit_years = np.array(years)[fit_mask]
        fit_k50s = np.array(k50s)[fit_mask]
        if len(fit_years) > 1:
            m, b = np.polyfit(fit_years, fit_k50s, 1)
            r2 = np.corrcoef(fit_years, fit_k50s)[0, 1] ** 2
            results[i]["K50"] = {
                "Trend": float(m),
                "Intercept": float(b),
                "R2": float(r2),
                "min(K50)": int(np.nanmin(fit_k50s)),
                "avg(K50)": float(np.nanmean(fit_k50s)),
                "max(K50)": int(np.nanmax(fit_k50s))
            }
        else:
            results[i]["K50"] = {"Trend": np.nan, "Intercept": np.nan, "R2": np.nan, "min(K50)": np.nan, "avg(K50)": np.nan, "max(K50)": np.nan}

        # CP (yearly concentration trend)
        fit_mask = ~np.isnan(total_yearly)
        fit_years = np.array(years)[fit_mask]
        fit_totals = np.array(total_yearly)[fit_mask]
        if len(fit_years) > 1:
            m, b = np.polyfit(fit_years, fit_totals, 1)
            r2 = np.corrcoef(fit_years, fit_totals)[0, 1] ** 2
            results[i]["CP"] = {
                "Trend": float(m),
                "Intercept": float(b),
                "R2": float(r2),
                "min(CP)": float(np.nanmin(fit_totals)),
                "avg(CP)": float(np.nanmean(fit_totals)),
                "max(CP)": float(np.nanmax(fit_totals))
            }
        else:
            results[i]["CP"] = {"Trend": np.nan, "Intercept": np.nan, "R2": np.nan, "min(CP)": np.nan, "avg(CP)": np.nan, "max(CP)": np.nan}

        # ---- PLOTTING (as before) ----
        fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(16, 10),
                              gridspec_kw={'width_ratios': [1, 1, 1]},
                              constrained_layout=True, dpi=300)

        # Zgornja vrstica
        plot_daily_values(ax[0, 0], x_standard, list_y, mean_y, colors, years, lw_1, lw_2, f"{i}: Dnevne vrednosti")
        plot_aggregated_yearly(ax[0, 1], years, list_y, colors, lw_1, lw_2, "Agregirana vsota skozi leto po letih")
        _starts, _ends, _k50s = plot_normalized_yearly(
            ax[0, 2], years, list_y, colors, reference_total, lw_1, lw_2,
            "Normirana agregirana vsota skozi leto", reference_year
        )
        # Spodnja vrstica
        plot_season_start(ax[1, 0], years, starts, colors, lw_1, "Začetek sezone (K10)")
        plot_season_end(ax[1, 1], years, ends, colors, lw_1, "Konec sezone (K90)")
        plot_yearly_concentration(ax[1, 2], years, total_yearly, colors, lw_1, "Letna koncentracija skozi leta")

        plt.tight_layout(rect=[0, 0.04, 1, 0.98])
        file_path = os.path.join(output_path, f"02_{i}_{location}.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    res_2 = {"Type":[],
            "Year":[],
            "Start":[],
            "End":[],
            "Sart-End interval":[],
            "K50":[],
            "rate":[],
            "max_CP":[]}

    # Združevanje po 'Type' namesto po 'Skupina'
    dg = df.groupby("Type")
    for i,ii in dg:
        for j,jj in ii.groupby("Year"):
            jj = jj.sort_values("Date")
            y = jj["Value"].values
            x = [ix for ix in range(len(y))]
            jj_cum_sum = [np.nansum(y[0:v]) for v in range(len(y[0:365]))]
            norm = np.nanmax(jj_cum_sum)
            total_sum = np.nansum(y)
            threshold_5_percent = 0.05 * total_sum

            if norm == 0:
                norm = 1.0

            jj_cum_sum_norm = np.array(jj_cum_sum) / norm

            Start_index = np.where(np.array(jj_cum_sum) > threshold_5_percent)[0]
            End_index = np.where(jj_cum_sum_norm > 0.975)[0]
            K50_index = np.where(jj_cum_sum_norm > 0.5)[0]

            K10 = int(Start_index[0]) if len(Start_index) > 0 else np.nan
            K50 = int(K50_index[0]) if len(K50_index) > 0 else np.nan
            K90 = int(End_index[0]) if len(End_index) > 0 else np.nan

            res_2["Start"].append(K10)
            res_2["K50"].append(K50)
            res_2["End"].append(K90)

            if not (np.isnan(K10) or np.isnan(K50) or np.isnan(K90)):
                interval = K90 - K10
                if interval > 0:
                    rate = float((jj_cum_sum[K90] - jj_cum_sum[K10]) / interval)
                else:
                    rate = np.nan
                
                res_2["Sart-End interval"].append(interval)
                res_2["rate"].append(rate)
            else:
                res_2["Sart-End interval"].append(np.nan)
                res_2["rate"].append(np.nan)

            res_2["max_CP"].append(norm)
            res_2["Type"].append(i)
            res_2["Year"].append(j)
    
    path = os.path.join("results",f"{location}_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=4)

    path = os.path.join("results",f"{location}_results_2.json")
    with open(path, "w") as f:
        json.dump(res_2, f, indent=4)
        
    return [results,res_2, colors]

def auc_and_ci_start_stop(results:dict, colors:dict, location:str):
    df_res_2 = pd.DataFrame(results)
    # display(df_res_2)  # če si v Jupyterju, sicer lahko odstraniš

    ##----------------------------------- Season start ------------------------------------------##
    fig, ax = plt.subplots(ncols=3,
                        nrows=1,
                        figsize=(10, 4),
                        gridspec_kw={'width_ratios': [1, 1, 1]},
                        dpi=150,
                        facecolor='white')
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    var = "Start"
    mean_values = df_res_2.groupby('Type')[var].mean().sort_values()
    iterate = mean_values.index.values
    sns.boxplot(data = df_res_2,x = "Type",y = var, order=mean_values.index,fliersize=0,ax = ax[0])
    for i,tip in enumerate(iterate):
        dfs = df_res_2[df_res_2["Type"]==tip].sort_values("Year")[var].values
        x = df_res_2[df_res_2["Type"]==tip].sort_values("Year")["Year"].values   
        color_palette = [colors[leto] for leto in x]
        ax[0].scatter([i for leto in x],dfs,color=color_palette,s=15)
    ax[0].set_xlabel("Vrsta")
    ax[0].set_ylabel("Pričetek sezone K10")
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45, horizontalalignment='right',fontsize=8)

    ##----------------------------------- Season End ------------------------------------------##
    var = "End"
    sns.boxplot(data = df_res_2,x = "Type",y = var, order=mean_values.index,fliersize=0,ax = ax[1])
    for i,tip in enumerate(iterate):
        dfs = df_res_2[df_res_2["Type"]==tip].sort_values("Year")[var].values
        x = df_res_2[df_res_2["Type"]==tip].sort_values("Year")["Year"].values   
        color_palette = [colors[leto] for leto in x]
        ax[1].scatter([i for leto in x],dfs,color=color_palette,s=15)
    ax[1].set_xlabel("Vrsta")
    ax[1].set_ylabel("Konec sezone K90")
    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45, horizontalalignment='right',fontsize=8)

    ##----------------------------------- Order of change------------------------------------------##
    var = "Sart-End interval"  # popravljeno ime ključa!
    mean_K10 = df_res_2.groupby('Type')[var].mean().sort_values().to_dict()
    for i,tip in enumerate(iterate):
        dfs = df_res_2[df_res_2["Type"]==tip].sort_values("Year")[var].values
        dfs = dfs - mean_K10[tip]
        x = df_res_2[df_res_2["Type"]==tip].sort_values("Year")["Year"].values
        color_palette = [colors[leto] for leto in x]
        ax[2].scatter([i for leto in x],dfs,color=color_palette,s=15,alpha = 0.5)
    ax[2].set_xlabel("Vrsta")
    ax[2].set_ylabel("Start-End interval (odstopanje od povprečja)")
    ax[2].set_xticklabels(ax[2].get_xticklabels(), rotation=45, horizontalalignment='right',fontsize=8)
    #ax[2].set_ylim(-50,50)

    # Shrani graf

    base_path = generate_base_path(location)
    file_path =  path_for_export(lv1=base_path, name=f"AUC_CI_90_{location}a.png")
    plt.tight_layout()
    plt.savefig(file_path, dpi=150)
    plt.show()
    plt.close()

    ##----------------------------------- Rate ------------------------------------------------##
    plt.figure(figsize=(8,4),dpi=150, facecolor='white')
    plt.title("Rate of change")
    mean_values = df_res_2.groupby('Type')['rate'].mean().sort_values()
    iterate = mean_values.index.values
    sns.boxplot(data = df_res_2,x = "Type",y = "rate", order=mean_values.index,fliersize=0)
    for i,tip in enumerate(iterate):
        dfs = df_res_2[df_res_2["Type"]==tip].sort_values("Year")["rate"].values
        x = df_res_2[df_res_2["Type"]==tip].sort_values("Year")["Year"].values   
        color_palette = [colors[leto] for leto in x]
        plt.scatter([i for leto in x],dfs,color=color_palette,s=15)
    plt.xticks(rotation=45, ha ="right", fontsize=8)
    base_path = generate_base_path(location)
    file_path =  path_for_export(lv1=base_path, name=f"AUC_CI_90_{location}b.png")
    plt.tight_layout()
    plt.savefig(file_path, dpi=150)
    plt.show()
    plt.close()

def save_correlation_results_to_json(correlation_dict, step_name):
    """
    Saves the correlation results to a structured JSON file.
    Args:
        correlation_dict (dict): A dictionary with correlation matrices.
        step_name (str): The name of the step to use for the output directory.
    """
    output_path = path_for_export(lv1="results", lv2=step_name)
    file_path = os.path.join(output_path, "correlation_results.json")
    
    # Convert DataFrames to dictionary for JSON serialization
    json_data = {}
    for pollen_type, data in correlation_dict.items():
        corr_matrix_dict = data["correlation_matrix"].to_dict()
        json_data[pollen_type] = {
            "correlation_matrix": corr_matrix_dict
        }
        
    with open(file_path, 'w') as f:
        json.dump(json_data, f, indent=4)
        
    print(f"Correlation results saved to '{file_path}'")

def show_results(results:dict):
    for i in results:
        print(i)
        for k in results[i]:
            print(f"  {k}")
            for z in results[i][k]:
                
                if type(results[i][k][z]) == float:
                    results[i][k][z] = round(results[i][k][z],3)
                print(f"    {z}: {results[i][k][z]}")
    
    df_K50 = {"Vrsta":[],
            "Trend":[],
            #"Intercept":[],
            "R2":[],
            "min(K50)":[],
            "avg(K50)":[],
            "max(K50)":[]}

    df_CP = {"Vrsta":[],
            "Trend":[],
            #"Intercept":[],
            "R2":[],
            "min(CP)":[],
            "avg(CP)":[],
            "max(CP)":[]}
    
    for i in results:
        print(i)
        df_K50["Vrsta"].append(i)
        df_K50["Trend"].append(f'{results[i]["K50"]["Trend"]:.1f}' if not np.isnan(results[i]["K50"]["Trend"]) else np.nan)
        #df_K50["Intercept"].append(results[i]["K50"]["Intercept"])
        df_K50["R2"].append(f'{results[i]["K50"]["R2"]:.1f}' if not np.isnan(results[i]["K50"]["R2"]) else np.nan)
        df_K50["min(K50)"].append(results[i]["K50"]["min(K50)"] if not np.isnan(results[i]["K50"]["min(K50)"]) else np.nan)
        df_K50["avg(K50)"].append(f'{results[i]["K50"]["avg(K50)"]:.1f}' if not np.isnan(results[i]["K50"]["avg(K50)"]) else np.nan)
        df_K50["max(K50)"].append(results[i]["K50"]["max(K50)"] if not np.isnan(results[i]["K50"]["max(K50)"]) else np.nan)
        
        df_CP["Vrsta"].append(i)
        df_CP["Trend"].append(f'{results[i]["CP"]["Trend"]:.1f}' if not np.isnan(results[i]["CP"]["Trend"]) else np.nan)
        #df_CP["Intercept"].append(results[i]["CP"]["Intercept"])
        df_CP["R2"].append(f'{results[i]["CP"]["R2"]:.1f}' if not np.isnan(results[i]["CP"]["R2"]) else np.nan)
        df_CP["min(CP)"].append(results[i]["CP"]["min(CP)"] if not np.isnan(results[i]["CP"]["min(CP)"]) else np.nan)
        df_CP["avg(CP)"].append(f'{results[i]["CP"]["avg(CP)"]:.1f}' if not np.isnan(results[i]["CP"]["avg(CP)"]) else np.nan)
        df_CP["max(CP)"].append(results[i]["CP"]["max(CP)"] if not np.isnan(results[i]["CP"]["max(CP)"]) else np.nan)

    df_K50 = pd.DataFrame(df_K50)
    df_CP = pd.DataFrame(df_CP)
    return df_K50, df_CP
