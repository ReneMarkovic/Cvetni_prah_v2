import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from src.utils import path_for_export, save_plot, generate_base_path
from src.utils import moving_average, path_for_export
import seaborn as sns
from scipy import stats

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
    all_types = sorted(list(set(item for df in df_list for item in df["Latinica"].unique())))
    
    correlation_results = {}
    
    print("\nIzvajanje korelacijske analize med regijami po vrsti cvetnega prahu...")
    
    for pollen_type in all_types:
        try:
            # Create a dictionary to hold the time series for the current pollen type from each location
            data_to_correlate = {}
            for i, loc in enumerate(locations):
                # Filter data for the current pollen type and set the date as index
                df_loc = df_list[i][df_list[i]["Latinica"] == pollen_type].copy()
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


def determine_season_by_mean(df: pd.DataFrame, location: str, pollen_type: str, start_th = 0.025, end_th = 0.975):
    df_filtered = df[(df["Latinica"] == pollen_type)].copy()
    print(f"    - Processing pollen type: {pollen_type} for location: {location}")
    if df_filtered.empty:
        print(f"Napaka: Podatki za vrsto '{pollen_type}' niso na voljo.")
        return None
    # Calculate the mean annual sum across all years
    mean_annual_sum = df_filtered.groupby("Year")["Value"].sum().mean()
    print(f"    - Mean annual sum for location {location}: {mean_annual_sum}")
    if mean_annual_sum == 0:
        print(f"Napaka: Povprečna letna vsota cvetnega prahu je 0 za vrsto '{pollen_type}'. Ne morem izračunati pragov.")
        return None
    # Calculate the start and end thresholds
    threshold_start_percent = start_th * mean_annual_sum
    threshold_end_percent = end_th * mean_annual_sum
    
    MAS = mean_annual_sum
    TotalSum = df_filtered["Value"].sum()
    Threshold_Start = start_th * MAS
    Threshold_End = end_th * MAS

    return {"Reference year": 0, "Total Sum": float(TotalSum), "Threshold Start": float(Threshold_Start), "Threshold End": float(Threshold_End)}

def determine_season_by_reference_year(df: pd.DataFrame, location: str, pollen_type: str, reference_year: int, start_th = 0.025, end_th = 0.975):

    # Filter data for the specified location and pollen type
    df_filtered = df[(df["Latinica"] == pollen_type)].copy()
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
    pollen_types = df["Latinica"].unique()
    pollen_type_season_reference = {}
    if reference_year == 0:
        #use the mean value of all years as reference
        for pollen_type in pollen_types:
            result = determine_season_by_mean(df, location, pollen_type, reference_year)
            if result is not None:
                pollen_type_season_reference[pollen_type] = result
    
    else:
        for pollen_type in pollen_types:
            #print("    - Determining season thresholds for pollen type:", pollen_type)
            result = determine_season_by_reference_year(df, location, pollen_type, reference_year)
            #print(result)
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
    ax.set_ylabel("Pollen count [units]", fontsize=13)
    ax.set_xlabel("Day of Year", fontsize=13)
    ax.grid(True, linestyle='--', alpha=0.5)

def plot_aggregated_yearly(ax, years, list_y, colors, lw_1, lw_2, title):
    """
    Za vsak vnos v list_y (vsako leto) nariši agregirano (kumulativno) vsoto skozi leto.
    Prikaži tudi povprečje kot debelejšo črto.
    """
    agg_y = [[np.nansum(y[0:v]) for v in range(len(y))] for y in list_y]
    mean_y = np.nanmean(agg_y, axis=0)
    
    norm_agg_y = [np.array(ay)/np.nanmax(ay) if np.nanmax(ay) > 0 else np.zeros_like(ay) for ay in agg_y]
    cum_sums = []
    for idx, y in enumerate(list_y):
        cum_sum = np.nancumsum(norm_agg_y[idx])
        cum_sums.append(cum_sum)
        ax.plot(range(len(cum_sum)), cum_sum, label=f"{years[idx]}", color=colors[years[idx]], linewidth=lw_1, alpha=0.7)
    # Povprečna kumulativna vsota skozi leto (črna črta)
    mean_cum = np.nanmean(np.array(cum_sums), axis=0)
    ax.plot(range(len(mean_cum)), mean_cum, color="black", linewidth=lw_2, label="Average", zorder=3)
    ax.set_title(title, fontsize=15)
    ax.set_ylabel("APIn", fontsize=13)
    ax.set_xlabel("Day of Year", fontsize=13)
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
    ax.plot(range(len(mean_norm_cum)), mean_norm_cum, color="black", linewidth=lw_2, label="Average", zorder=3)
    ax.set_title(f"{title} (reference year {reference_year})", fontsize=15)
    ax.set_ylabel("Normalized APIn", fontsize=13)
    ax.set_xlabel("Day of Year", fontsize=13)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=8, ncol=2, loc='upper left')

    return starts, ends, k50s

def plot_season_start(ax, years, starts, colors, lw_1, title):
    # scatter in črta
    ax.scatter(years, starts, color=[colors[y] for y in years], s=80, label="Season start")
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
    ax.set_ylabel("Day of Year (start)", fontsize=13)
    ax.set_xlabel("Year", fontsize=13)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9, loc='best')

def plot_season_end(ax, years, ends, colors, lw_1, title):
    """
    Prikaz konca sezone (npr. K90) skozi leta z linearno fit premico in R².
    """
    ax.scatter(years, ends, color=[colors[y] for y in years], s=80, label="Season end")
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
    ax.set_ylabel("Day of Year (end)", fontsize=13)
    ax.set_xlabel("Year", fontsize=13)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9, loc='best')

def plot_yearly_concentration(ax, years, total_yearly, colors, lw_1, title):
    """
    Prikaz letne koncentracije skozi leta z linearno fit premico in R².
    """
    ax.scatter(years, total_yearly, color=[colors[y] for y in years], s=80, label="Yearly concentration")
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
    ax.set_ylabel("APIn", fontsize=13)
    ax.set_xlabel("Year", fontsize=13)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9, loc='best')

def year_specific_activation(df: pd.DataFrame, location:str, ma:int = 7, step_name:str = "default", reference_year: int = 2004):
    cmap = plt.get_cmap("viridis")  # ali "plasma", "cividis", "coolwarm", "RdYlBu"
    dg = df.groupby("Latinica")
    print("    - Generating activation reference...")
    act_reference = activation_reference(df, location=location, reference_year=reference_year)
    #print(act_reference)
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
    print("    - Generating type-specific activation plots...")
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
            results[i]["Start"] = {
                "Trend": float(m),
                "Intercept": float(b),
                "R2": float(r2),
                "min(Start)": int(np.nanmin(fit_starts)),
                "avg(Start)": float(np.nanmean(fit_starts)),
                "max(Start)": int(np.nanmax(fit_starts))
            }
        else:
            results[i]["K10"] = {"Trend": np.nan, "Intercept": np.nan, "R2": np.nan, "min(Start)": np.nan, "avg(Start)": np.nan, "max(Start)": np.nan}

        # K90 (End trend)
        fit_mask = ~np.isnan(ends)
        fit_years = np.array(years)[fit_mask]
        fit_ends = np.array(ends)[fit_mask]
        if len(fit_years) > 1:
            m, b = np.polyfit(fit_years, fit_ends, 1)
            r2 = np.corrcoef(fit_years, fit_ends)[0, 1] ** 2
            results[i]["End"] = {
                "Trend": float(m),
                "Intercept": float(b),
                "R2": float(r2),
                "min(End)": int(np.nanmin(fit_ends)),
                "avg(End)": float(np.nanmean(fit_ends)),
                "max(End)": int(np.nanmax(fit_ends))
            }
        else:
            results[i]["End"] = {"Trend": np.nan, "Intercept": np.nan, "R2": np.nan, "min(End)": np.nan, "avg(End)": np.nan, "max(End)": np.nan}
        
        print(f"      - Trends for {i}: Start Trend={results[i]['Start']['Trend']}, End Trend={results[i]['End']['Trend']}")
        # K50 (mid-season trend)
        fit_mask = ~np.isnan(k50s)
        fit_years = np.array(years)[fit_mask]
        fit_k50s = np.array(k50s)[fit_mask]
        if len(fit_years) > 1:
            m, b = np.polyfit(fit_years, fit_k50s, 1)
            r2 = np.corrcoef(fit_years, fit_k50s)[0, 1] ** 2
            results[i]["mid"] = {
                "Trend": float(m),
                "Intercept": float(b),
                "R2": float(r2),
                "min(mid)": int(np.nanmin(fit_k50s)),
                "avg(mid)": float(np.nanmean(fit_k50s)),
                "max(mid)": int(np.nanmax(fit_k50s))
            }
        else:
            results[i]["mid"] = {"Trend": np.nan, "Intercept": np.nan, "R2": np.nan, "min(mid)": np.nan, "avg(mid)": np.nan, "max(mid)": np.nan}
        
        print(f"      - Trends for {i}: mid Trend={results[i]['mid']['Trend']}")
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
        
        print(f"      - Plotting figures for {i}...")
        # Zgornja vrstica
        plot_daily_values(ax[0, 0], x_standard, list_y, mean_y, colors, years, lw_1, lw_2, f"{i}: Daily values")
        plot_aggregated_yearly(ax[0, 1], years, list_y, colors, lw_1, lw_2, "Aggregated sum throughout the year by years")
        _starts, _ends, _k50s = plot_normalized_yearly(
            ax[0, 2], years, list_y, colors, reference_total, lw_1, lw_2,
            "Normalized aggregated sum throughout the year", reference_year
        )
        # Bottom row
        plot_season_start(ax[1, 0], years, starts, colors, lw_1, "Season start (Start)")
        plot_season_end(ax[1, 1], years, ends, colors, lw_1, "Season end (End)")
        plot_yearly_concentration(ax[1, 2], years, total_yearly, colors, lw_1, "Yearly concentration over the years")

        #plt.tight_layout(rect=[0, 0.04, 1, 0.98])
        file_path = os.path.join(output_path, f"02_{i}_{location}.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print("    - Saving results res_2...") 
    res_2 = {"Pollen taxa":[],
            "Year":[],
            "Start":[],
            "End":[],
            "Start-End interval":[],
            "mid":[],
            "rate":[],
            "max_CP":[]}

    # Združevanje po 'Type' namesto po 'Skupina'
    print("    - Calculating res_2 statistics...")
    dg = df.groupby("Latinica")
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
            res_2["mid"].append(K50)
            res_2["End"].append(K90)

            if not (np.isnan(K10) or np.isnan(K50) or np.isnan(K90)):
                interval = K90 - K10
                if interval > 0:
                    rate = float((jj_cum_sum[K90] - jj_cum_sum[K10]) / interval)
                else:
                    rate = np.nan
                
                res_2["Start-End interval"].append(interval)
                res_2["rate"].append(rate)
            else:
                res_2["Start-End interval"].append(np.nan)
                res_2["rate"].append(np.nan)

            res_2["max_CP"].append(norm)
            res_2["Pollen taxa"].append(i)
            res_2["Year"].append(j)
    
    print("    - Saving results to JSON...")
    path = os.path.join("results",f"{location}_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=4)

    path = os.path.join("results",f"{location}_results_2.json")
    with open(path, "w") as f:
        json.dump(res_2, f, indent=4)
        
    return [results,res_2, colors]


def type_specific_activation(df: pd.DataFrame, location: str, ma: int = 7, step_name: str = "default"):
    """
    Analiza sezonske aktivacije za posamezno vrsto in leto,
    s prikazom dinamike, kumulativne (nenormirane) in normirane vsote,
    ter trendov začetka, konca in letne koncentracije skozi leta.
    """

    cmap = plt.get_cmap("viridis")
    dg = df.groupby("Latinica")
    print("    - Izvajanje analize aktivacije (normirane in nenormirane)...")

    all_years = sorted(df["Year"].unique())
    colors = {year: cmap(i / (len(all_years) - 1)) for i, year in enumerate(all_years)}

    lw_1, lw_2 = 1.2, 2.5
    results = {}
    res_2 = {"Pollen taxa": [], "Year": [], "Start": [], "End": [], "Length": [], "mid": [], "rate": [], "max_CP": []}

    for pollen_type, df_type in dg:
        base_path = generate_base_path(location)
        output_path = path_for_export(lv1=base_path, lv2=step_name)

        years, starts, ends, mids, lengths, total_yearly = [], [], [], [], [], []
        list_y, list_cum, list_cum_norm = [], [], []

        for year, df_year in df_type.groupby("Year"):
            df_year = df_year.sort_values("Date")
            y = df_year["Value"].fillna(0).values[:365]
            if np.nansum(y) == 0:
                continue

            # --- kumulativne ---
            cum_sum = np.nancumsum(y)
            total_sum = np.nanmax(cum_sum)
            norm_cum = cum_sum / total_sum if total_sum > 0 else np.zeros_like(cum_sum)

            # --- pragovi ---
            start_idx = np.argmax(norm_cum >= 0.025) if np.any(norm_cum >= 0.025) else np.nan
            end_idx = np.argmax(norm_cum >= 0.975) if np.any(norm_cum >= 0.975) else np.nan
            mid_idx = np.argmax(norm_cum >= 0.5) if np.any(norm_cum >= 0.5) else np.nan
            length = end_idx - start_idx if not np.isnan(start_idx) and not np.isnan(end_idx) else np.nan

            # --- zapisi ---
            res_2["Pollen taxa"].append(pollen_type)
            res_2["Year"].append(year)
            res_2["Start"].append(start_idx)
            res_2["End"].append(end_idx)
            res_2["Length"].append(length)
            res_2["mid"].append(mid_idx)
            res_2["max_CP"].append(total_sum)

            rate = np.nan
            if not (np.isnan(start_idx) or np.isnan(end_idx)):
                interval = end_idx - start_idx
                if interval > 0:
                    rate = float((cum_sum[int(end_idx)] - cum_sum[int(start_idx)]) / interval)
            res_2["rate"].append(rate)

            years.append(year)
            starts.append(start_idx)
            ends.append(end_idx)
            mids.append(mid_idx)
            lengths.append(length)
            total_yearly.append(total_sum)
            list_y.append(y)
            list_cum.append(cum_sum)
            list_cum_norm.append(norm_cum)

        if not years:
            continue

        # --- trend helper ---
        def trend_block(var_list):
            fit_mask = ~np.isnan(var_list)
            fit_years = np.array(years)[fit_mask]
            fit_vals = np.array(var_list)[fit_mask]
            if len(fit_years) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(fit_years, fit_vals)
                r2 = r_value ** 2
                return dict(
                    Trend=float(slope), 
                    Intercept=float(intercept), 
                    R2=float(r2), # Dodatne ključne statistike za preverjanje značilnosti trenda:
                    P_Value=float(p_value), # P-vrednost
                    Std_Error=float(std_err), # Osnovna statistika
                    min=float(np.nanmin(fit_vals)), 
                    avg=float(np.nanmean(fit_vals)), 
                    max=float(np.nanmax(fit_vals))
                )
            # Vrne NaN, če ni dovolj podatkov
            return dict(
                Trend=np.nan, Intercept=np.nan, R2=np.nan, P_Value=np.nan, Std_Error=np.nan,
                min=np.nan, avg=np.nan, max=np.nan
            )

        results[pollen_type] = {
            "Start": trend_block(starts),
            "End": trend_block(ends),
            "Length": trend_block(lengths),
            "CP": trend_block(total_yearly),
            "mids": trend_block(mids)
        }

        # === RISANJE PANELNE POSTAVITVE ===
        fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(16, 10), constrained_layout=True, dpi=300)
        x = list(range(365))
        mean_y = np.nanmean(list_y, axis=0)
        mean_cum = np.nanmean(list_cum, axis=0)
        mean_cum_norm = np.nanmean(list_cum_norm, axis=0)

        # (0,0) Dnevne in zglajene dinamike
        for idx, y in enumerate(list_y):
            ax[0, 0].plot(x, y, color=colors[years[idx]], alpha=0.5, linewidth=lw_1)
        smooth = pd.Series(mean_y).rolling(window=ma, center=True, min_periods=1).mean()
        ax[0, 0].plot(x, mean_y, 'k-', linewidth=1.5, label="Avg daily values")
        ax[0, 0].plot(x, smooth, 'r--', linewidth=2.0, label=f"Smoothed ({ma}-day)")
        ax[0, 0].set_title("Daily values and smoothed dynamics")
        ax[0, 0].legend(fontsize=8)
        ax[0, 0].grid(True, linestyle='--', alpha=0.4)

        # (0,1) Nenormirana kumulativna vsota
        for idx, cum in enumerate(list_cum):
            ax[0, 1].plot(x, cum, color=colors[years[idx]], alpha=0.7, linewidth=lw_1)
        ax[0, 1].plot(x, mean_cum, 'k-', linewidth=lw_2, label="Average")
        ax[0, 1].set_title("APIn")
        ax[0, 1].grid(True, linestyle='--', alpha=0.4)

        # (0,2) Normirana kumulativna vsota
        for idx, norm_cum in enumerate(list_cum_norm):
            ax[0, 2].plot(x, norm_cum, color=colors[years[idx]], alpha=0.7, linewidth=lw_1)
        ax[0, 2].plot(x, mean_cum_norm, 'k-', linewidth=lw_2, label="Average")
        ax[0, 2].axhline(0.025, color='grey', linestyle='--', alpha=0.6)
        ax[0, 2].axhline(0.975, color='grey', linestyle='--', alpha=0.6)
        ax[0, 2].set_title("Normalized APIn")
        ax[0, 2].grid(True, linestyle='--', alpha=0.4)

        # (1,0) Začetek sezone (trend)
        plot_season_start(ax[1, 0], years, starts, colors, lw_1, "Start of season (2.5%)")

        # (1,1) Konec sezone (trend)
        plot_season_end(ax[1, 1], years, ends, colors, lw_1, "End of season (97.5%)")

        # (1,2) Letna koncentracija (trend)
        plot_yearly_concentration(ax[1, 2], years, total_yearly, colors, lw_1, "Annual pollen concentration")

        file_path = os.path.join(output_path, f"02_{pollen_type}_{location}.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"      - Grafi ustvarjeni za {pollen_type}")

    # --- Shranjevanje rezultatov ---
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", f"{location}_results_norm.json"), "w") as f:
        json.dump(results, f, indent=4)
    
    
    for key in res_2:
        print(f"    - {key}: {len(res_2[key])} entries")
    print("    - Saving res_2 results...")
    with open(os.path.join("results", f"{location}_results_2_norm.json"), "w") as f:
        json.dump(res_2, f, indent=4, default=numpy_encoder)

    print(f"    - Rezultati shranjeni v 'results/{location}_results_norm.json' in 'results/{location}_results_2_norm.json'")

    return [results, res_2, colors]

def numpy_encoder(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    # Za vsak primer obravnavamo tudi NaN, ki ni standarden JSON tip
    elif np.isnan(obj):
        return None 
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
    
def auc_and_ci_start_stop(results:dict, colors:dict, location:str):
    df_res_2 = pd.DataFrame(results)
    print("    - Generating AUC and CI plots...")

    ##----------------------------------- Season start ------------------------------------------##
    fig, ax = plt.subplots(ncols=3,
                        nrows=1,
                        figsize=(14, 6),
                        gridspec_kw={'width_ratios': [1, 1, 1]},
                        dpi=150,
                        facecolor='white')
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for a in ax:
        a.set_facecolor('white')  # AXES background white

    var = "Start"
    mean_values = df_res_2.groupby('Latinica')[var].mean().sort_values()
    iterate = mean_values.index.values

    # Boxplot: no color fill
    sns.boxplot(
        data=df_res_2, x="Latinica", y=var, order=mean_values.index, fliersize=0, ax=ax[0],
        boxprops=dict(facecolor='white', color='black'),
        medianprops=dict(color='black'),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black')
    )
    # Scatter: use colormap for years
    for i, tip in enumerate(iterate):
        df_filtered = df_res_2[df_res_2["Latinica"] == tip].sort_values("Year")
        dfs = df_filtered[var].values
        x = df_filtered["Year"].values
        color_palette = [colors[leto] for leto in x]
        ax[0].scatter([i]*len(x), dfs, color=color_palette, s=35, alpha=0.7, edgecolor='k', linewidth=0.5)
    ax[0].set_xlabel("Pollen taxa", fontsize=12)
    ax[0].set_ylabel("Start of season K10", fontsize=12)
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45, ha='right', fontsize=10)
    ax[0].set_title("Start of season (K10)", fontsize=14)
    ax[0].grid(True, linestyle='--', alpha=0.3)

    ##----------------------------------- Season End ------------------------------------------##
    var = "End"
    sns.boxplot(
        data=df_res_2, x="Latinica", y=var, order=mean_values.index, fliersize=0, ax=ax[1],
        boxprops=dict(facecolor='white', color='black'),
        medianprops=dict(color='black'),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black')
    )
    for i, tip in enumerate(iterate):
        df_filtered = df_res_2[df_res_2["Latinica"] == tip].sort_values("Year")
        dfs = df_filtered[var].values
        x = df_filtered["Year"].values
        color_palette = [colors[leto] for leto in x]
        ax[1].scatter([i]*len(x), dfs, color=color_palette, s=35, alpha=0.7, edgecolor='k', linewidth=0.5)
    ax[1].set_xlabel("Pollen taxa", fontsize=12)
    ax[1].set_ylabel("End of season", fontsize=12)
    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45, ha='right', fontsize=10)
    ax[1].set_title("End of season", fontsize=14)
    ax[1].grid(True, linestyle='--', alpha=0.3)

    ##----------------------------------- Order of change------------------------------------------##
    var = "Length"
    mean_K10 = df_res_2.groupby('Latinica')[var].mean().sort_values().to_dict()
    sns.boxplot(
        data=df_res_2, x="Latinica", y=var, order=mean_values.index, fliersize=0, ax=ax[2],
        boxprops=dict(facecolor='white', color='black'),
        medianprops=dict(color='black'),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black')
    )
    for i, tip in enumerate(iterate):
        df_filtered = df_res_2[df_res_2["Latinica"] == tip].sort_values("Year")
        dfs = df_filtered[var].values - mean_K10[tip]
        x = df_filtered["Year"].values
        color_palette = [colors[leto] for leto in x]
        ax[2].scatter([i]*len(x), dfs, color=color_palette, s=35, alpha=0.5, edgecolor='k', linewidth=0.5)
    ax[2].set_xlabel("Pollen taxa", fontsize=12)
    ax[2].set_ylabel("Start-End interval (deviation from average)", fontsize=12)
    ax[2].set_xticklabels(ax[2].get_xticklabels(), rotation=45, ha='right', fontsize=10)
    ax[2].set_title("Season length (deviation)", fontsize=14)
    ax[2].grid(True, linestyle='--', alpha=0.3)

    base_path = generate_base_path(location)
    file_path = path_for_export(lv1=base_path, name=f"AUC_CI_90_{location}a.png")
    plt.tight_layout()
    plt.savefig(file_path, dpi=150)
    plt.show()
    plt.close()

    ##----------------------------------- Rate ------------------------------------------------##
    plt.figure(figsize=(10, 5), dpi=150, facecolor='white')
    plt.title("Hitrost spremembe", fontsize=15)
    mean_values = df_res_2.groupby('Latinica')['rate'].mean().sort_values()
    iterate = mean_values.index.values
    sns.boxplot(
        data=df_res_2, x="Latinica", y="rate", order=mean_values.index, fliersize=0,
        boxprops=dict(facecolor='white', color='black'),
        medianprops=dict(color='black'),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black')
    )
    for i, tip in enumerate(iterate):
        df_filtered = df_res_2[df_res_2["Latinica"] == tip].sort_values("Year")
        dfs = df_filtered["rate"].values
        x = df_filtered["Year"].values
        color_palette = [colors[leto] for leto in x]
        plt.scatter([i]*len(x), dfs, color=color_palette, s=35, alpha=0.7, edgecolor='k', linewidth=0.5)
    plt.xlabel("Pollen taxa", fontsize=12)
    plt.ylabel("Rate of change", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    
    file_path = path_for_export(lv1=base_path, name=f"AUC_CI_90_{location}b.png")
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

def show_results(results: dict):
    # 1. Prvi prehod: Tiskanje in zaokroževanje vseh metrik
    print("--- Detaljni rezultati analize trenda ---")
    for taxa, taxa_data in results.items():
        print(f"## 🌸 {taxa}")
        for param_type, stats in taxa_data.items():
            print(f"  --> {param_type} parametri:")
            for stat_name, value in stats.items():
                
                # Zaokroževanje samo, če je vrednost float
                if type(value) == float:
                    # R2 in P-vrednost zaokrožimo na 3 decimalke za natančnost
                    if stat_name in ["R2", "P_Value"]:
                        rounded_value = round(value, 3)
                    # Trend in ostale statistike na 2 decimalki
                    elif stat_name in ["Trend", "Std_Error", "min", "avg", "max"]:
                         rounded_value = round(value, 2)
                    else:
                         rounded_value = value
                         
                    results[taxa][param_type][stat_name] = rounded_value
                    print(f"    - {stat_name}: {rounded_value}")
                else:
                    print(f"    - {stat_name}: {value}")
    
    print("\n" + "-"*50 + "\n")

    # 2. Priprava DataFrame-ov z dodanimi statistikami
    
    # Dodana P_Value in Std_Error v strukturo DataFrame-a
    df_K50_data = {
        "Pollen taxa": [], "Trend": [], "P_Value": [], "Std_Error": [], "R2": [],
        "min(mids)": [], "avg(mids)": [], "max(mids)": []
    }

    df_CP_data = {
        "Pollen taxa": [], "Trend": [], "P_Value": [], "Std_Error": [], "R2": [],
        "min(CP)": [], "avg(CP)": [], "max(CP)": []
    }

    for taxa in results:
        # Podatki za K50 (mids)
        df_K50_data["Pollen taxa"].append(taxa)
        mids_stats = results[taxa]["mids"]
        
        # Pomožna funkcija za vstavljanje podatkov, če niso NaN
        def get_stat(stat_key, stats):
             return stats.get(stat_key, np.nan)
        
        df_K50_data["Trend"].append(get_stat("Trend", mids_stats))
        df_K50_data["P_Value"].append(get_stat("P_Value", mids_stats))
        df_K50_data["Std_Error"].append(get_stat("Std_Error", mids_stats))
        df_K50_data["R2"].append(get_stat("R2", mids_stats))
        df_K50_data["min(mids)"].append(get_stat("min", mids_stats))
        df_K50_data["avg(mids)"].append(get_stat("avg", mids_stats))
        df_K50_data["max(mids)"].append(get_stat("max", mids_stats))

        # Podatki za CP
        df_CP_data["Pollen taxa"].append(taxa)
        cp_stats = results[taxa]["CP"]
        
        df_CP_data["Trend"].append(get_stat("Trend", cp_stats))
        df_CP_data["P_Value"].append(get_stat("P_Value", cp_stats))
        df_CP_data["Std_Error"].append(get_stat("Std_Error", cp_stats))
        df_CP_data["R2"].append(get_stat("R2", cp_stats))
        df_CP_data["min(CP)"].append(get_stat("min", cp_stats))
        df_CP_data["avg(CP)"].append(get_stat("avg", cp_stats))
        df_CP_data["max(CP)"].append(get_stat("max", cp_stats))

    # 3. Ustvarjanje DataFrame-ov
    df_K50 = pd.DataFrame(df_K50_data)
    df_CP = pd.DataFrame(df_CP_data)

    # Tiskanje končnih tabel za lažjo vizualizacijo
    print("## 📊 Tabela K50/mids Trendi")
    print(df_K50.to_string())
    print("\n## 📊 Tabela CP Trendi")
    print(df_CP.to_string())
    
    return df_K50, df_CP
