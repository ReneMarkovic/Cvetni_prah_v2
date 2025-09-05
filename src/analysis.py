# src/analysis.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

from src.utils import moving_average, path_for_export


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

def type_specific_activation(df: pd.DataFrame, location:str, ma:int = 7):
    dg = df.groupby("Type")
    cmap = plt.get_cmap("RdBu")#"tab20c")

    results = {}
    dg = df.groupby("Type")

    lw_1 = 0.5
    lw_2 = 1
    w = ma
    for i,ii in dg:
        list_y = []
        list_y_smooth = []
        list_y_cumsum = []
        list_y_cumsum_norm = []
        K50 = {}
        max_CP = {}
        years = sorted(ii["Year"].unique())  # Get a sorted list of years for consistency
        colors = {year: cmap(i / len(years)) for i, year in enumerate(years)} 
        
        fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(10, 6), gridspec_kw={'width_ratios': [1, 1, 1]},dpi=150, facecolor='lightblue')
        #add spacing between subplots
        plt.subplots_adjust(wspace=0.5)
        x_standard = [k for k in range(365)]
        
        for j,jj in ii.groupby("Year"):
            jj = jj.sort_values("Date")
            x = [k for k in range(len(jj["Date"]))][0:365]
            y = jj["Value"].values[0:365]
            list_y.append(y)
            
            ax[0,0].plot(x_standard,y,label=f"{j}",alpha = 0.5, color=colors[j],linewidth = lw_1)
            
            y_smooth = moving_average(y, w)
            xs = [k+w//2 for k in range(len(y_smooth))]
            list_y_smooth.append(y_smooth)
            ax[0,1].plot(xs,y_smooth,label=f"{j}",alpha = 0.5, color=colors[j],linewidth = lw_1)
            
            xx = [k for k in range(len(y))]
            yy = np.array([np.nansum(y[0:v]) for v in range(len(y))])
            list_y_cumsum.append(yy)
            ax[0,2].plot(xx,yy,label=f"{j}",alpha = 0.5, color=colors[j],linewidth = lw_1)
            
            norm = np.nanmax(yy)
            if norm == 0:
                norm = 1.0
            yy_norm = yy/norm
            selected_list = np.where(yy_norm>0.5)
            if len(selected_list[0]) > 0:
                K50[j] = np.where(yy_norm>0.5)[0][0]
            else:
                K50[j] = np.nan
            max_CP[j] = np.nanmax(yy)
            list_y_cumsum_norm.append(yy_norm)
            ax[1,0].plot(np.array(xx),yy_norm,label=f"{j}",alpha = 0.5, color=colors[j],linewidth = lw_1)
        
        mean_y = np.nanmean(list_y,axis=0)
        ax[0,0].plot(x,mean_y,label="Povprečje",c = "black",linewidth = lw_2)
        mean_y_smooth = np.nanmean(list_y_smooth,axis=0)
        ax[0,1].plot(mean_y_smooth,label="Povprečje",c = "black",linewidth = lw_2)
        
        mean_y_cumsum = np.nanmean(list_y_cumsum,axis=0)
        ax[0,2].plot(mean_y_cumsum,label="Povprečje",c = "black",linewidth = lw_2)
        
        mean_y_cumsum = np.nanmean(list_y_cumsum_norm,axis=0)
        mean_y_cumsum = [np.nanmean(np.array(list_y_cumsum_norm)[:,k]) for k in range(365)]
        
        ax[1,0].plot(mean_y_cumsum,label="Povprečje",c = "black",linewidth = lw_2)
        
        handles, labels = ax[0,0].get_legend_handles_labels()
        unique_labels = list(dict.fromkeys(labels))  # Limit the number of legend entries
        fig.legend(handles[:len(unique_labels)], unique_labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=12, fontsize=4)
        
        x = [k for k in K50]
        y_50 = [K50[leto] for leto in K50]
        #linear regression
        m, b = np.polyfit(x, y_50, 1)
        # R^2
        results[i]={}
        results[i]["K50"]={"Trend":float(m),
                        "Intercept":float(b),
                        "R2":float(np.corrcoef(x, y_50)[0, 1]**2),
                        "min(K50)":int(np.nanmin(y_50)),
                        "avg(K50)":int(np.nanmean(y_50)),
                        "max(K50)":int(np.nanmax(y_50))}
        r2 = np.corrcoef(x, y_50)[0, 1]**2
        ax[1,1].text(0.1, 1.15, fr"$y = {m:.2f}x + {b:.2f}$", transform=ax[1,1].transAxes, fontsize = 8)
        ax[1,1].text(0.1, 1.05, fr"$(R^2 = {r2:.3f})$", transform=ax[1,1].transAxes, fontsize = 8)
        ax[1,1].plot(x, m*np.array(x) + b, c = "red",linewidth = 2)
        ax[1,1].scatter([leto for leto in K50], [K50[leto] for leto in K50], color=[colors[leto] for leto in K50])
        
        x = [k for k in K50]
        y = [max_CP[leto] for leto in K50]
        #linear regression
        m, b = np.polyfit(x, y, 1)
        r2 = np.corrcoef(x, y)[0, 1]**2
        results[i]["CP"]={"Trend":float(m),
                        "Intercept":float(b),
                        "R2":float(np.corrcoef(x, y)[0, 1]**2),
                        "min(CP)":float(min(y)),
                        "avg(CP)":float(np.mean(y)),
                        "max(CP)":float(max(y))}
        ax[1,2].text(0.1, 1.15, fr"$y = {m:.2f}x + {b:.2f}$", transform=ax[1,2].transAxes, fontsize = 8)
        ax[1,2].text(0.1, 1.05, fr"$(R^2 = {r2:.3f})$", transform=ax[1,2].transAxes, fontsize = 8)
        ax[1,2].plot(x, m*np.array(x) + b, c = "red",linewidth = 2)
        ax[1,2].scatter([leto for leto in K50], [max_CP[leto] for leto in K50], color=[colors[leto] for leto in K50])
        
        for ix in range(0,2):
            for iy in range(0,3):
                ax[ix,iy].set_xlim(0,365)
        
        
        ax[0,0].set_title(i,fontsize = 8)
        ax[0,0].set_ylabel("Količina cvetnega prahu [?]",fontsize = 6)
        ax[0,0].set_xlabel("Dan v letu",fontsize = 6)
        
        ax[0,1].set_title(f"{w}-dneno povprečje cvetnega prahu",fontsize = 8)
        ax[0,1].set_ylabel("Količina cvetnega prahu [?]",fontsize = 6)
        ax[0,1].set_xlabel("Dan v letu",fontsize = 6)
        
        ax[0,2].set_title("Kumulativna vsota cvetnega prahu",fontsize = 8)
        ax[0,2].set_ylabel("Količina cvetnega prahu [?]",fontsize = 6)
        ax[0,2].set_xlabel("Dan v letu",fontsize = 6)
        
        ax[1,0].set_title("Normirana kumulativna vsota cvetnega prahu",fontsize = 8)
        ax[1,0].set_ylabel("Delež od max",fontsize = 6)
        ax[1,0].set_xlabel("Dan v letu",fontsize = 6)
        ax[1,0].hlines(0.5,0,365,linestyles="--",color="green")
        ax[1,0].text(np.mean(y_50)-50,0.51,"50%",color="green")
        ax[1,0].set_ylim(-0.01,1.01)
        ax[1,0].set_xlim(np.nanmean(y_50)-50,np.nanmean(y_50)+50)
        
        ax[1,1].set_xlim(2002,2023)
        ax[1,1].set_ylabel("K50",fontsize = 6)
        ax[1,1].set_xlabel("Dan v letu",fontsize = 6)
        
        ax[1,2].set_xlim(2002,2023)
        ax[1,2].set_ylabel("Skupna koncentracija",fontsize = 6)
        ax[1,2].set_xlabel("Dan v letu",fontsize = 6)
        
        plt.tight_layout()
        file_path = path_for_export(lv1 = "results", lv2  = f"{location}", name = f"02_{i}.png")
        plt.savefig(file_path)
        plt.close()
        
    res_2 = {"Type":[],
            "Year":[],
            "Start":[],
            "End":[],
            "Sart-End interval":[],
            "K50":[],
            "rate":[],
            "max_CP":[]}
    
    dg = df.groupby("Type")
    for i,ii in dg:
        for j,jj in ii.groupby("Year"):
            jj = jj.sort_values("Date")
            y = jj["Value"].values
            x = [ix for ix in range(len(y))]
            jj_cum_sum = [np.nansum(y[0:v]) for v in range(len(y[0:365]))]
            norm = np.nanmax(jj_cum_sum)
            total_sum = np.nansum(y) # Calculate total annual sum for the current year
            threshold_5_percent = 0.05 * total_sum # Calculate the 5% threshold

            # Ensure cumulative sum is normalized properly, handle cases where no values > 0.1, 0.5, or 0.9
            if norm == 0:
                norm = 1.0  # Prevent division by zero

            jj_cum_sum_norm = np.array(jj_cum_sum) / norm

            # Find K10, K50, K90 with proper handling for empty results
            # The start of the season is now when cumulative sum exceeds the 5% threshold
            Start_index = np.where(np.array(jj_cum_sum) > threshold_5_percent)[0]
            End_index = np.where(jj_cum_sum_norm > 0.975)[0]
            K50_index = np.where(jj_cum_sum_norm > 0.5)[0]

            # Assign default values if no indices are found
            K10 = int(Start_index[0]) if len(Start_index) > 0 else np.nan
            K50 = int(K50_index[0]) if len(K50_index) > 0 else np.nan
            K90 = int(End_index[0]) if len(End_index) > 0 else np.nan

            # Add these values to your results
            res_2["Start"].append(K10)
            res_2["K50"].append(K50)
            res_2["End"].append(K90)

            # Only calculate interval and rate if all K-values are valid (not NaN)
            if not (np.isnan(K10) or np.isnan(K50) or np.isnan(K90)):
                interval = int(K90 - K10)
                rate = float((jj_cum_sum[K90] - jj_cum_sum[K10]) / interval)
                
                res_2["Sart-End interval"].append(interval)
                res_2["rate"].append(rate)
            else:
                # If any K-values are NaN, append NaN for interval and rate
                res_2["Sart-End interval"].append(np.nan)
                res_2["rate"].append(np.nan)

            # Append max_CP for each year
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
    
def determine_season_by_reference_year(df: pd.DataFrame, location: str, pollen_type: str, reference_year: int):
    """
    Determines the start (5%) and end (95%) of the pollen season
    for a specific pollen type using a reference year's total sum.

    Args:
        df (pd.DataFrame): The processed DataFrame containing pollen data.
        location (str): The location for which the analysis is performed.
        pollen_type (str): The specific type of pollen to analyze.
        reference_year (int): The year used to determine the reference total annual sum.

    Returns:
        pd.DataFrame: A DataFrame with the start and end days for each year.
    """
    # Filter data for the specified location and pollen type
    df_filtered = df[(df["Type"] == pollen_type)].copy()
    if df_filtered.empty:
        print(f"Napaka: Podatki za vrsto '{pollen_type}' niso na voljo.")
        return None

    # Calculate the total annual sum for the reference year
    df_ref_year = df_filtered[df_filtered["Year"] == reference_year]
    if df_ref_year.empty:
        print(f"Napaka: Referenčno leto {reference_year} ni najdeno za vrsto '{pollen_type}'.")
        return None
        
    reference_total_sum = df_ref_year["Value"].sum()

    if reference_total_sum == 0:
        print(f"Napaka: Vsota cvetnega prahu za referenčno leto {reference_year} je 0. Ne morem izračunati pragov.")
        return None

    # Calculate the 5% and 95% thresholds
    threshold_5_percent = 0.05 * reference_total_sum
    threshold_95_percent = 0.95 * reference_total_sum
    
    # Dictionary to store results
    season_results = {"Year": [], "Start_Day": [], "End_Day": []}

    # Iterate through each year and determine season start and end
    for year, group in df_filtered.groupby("Year"):
        if year == reference_year:
            # We already have the reference sum, just need to find the dates
            pass

        group = group.sort_values("Date")
        cumulative_sum = np.array([np.nansum(group["Value"].values[:i+1]) for i in range(len(group))])
        
        # Find the day when the cumulative sum exceeds the thresholds
        start_day_index = np.where(cumulative_sum >= threshold_5_percent)[0]
        end_day_index = np.where(cumulative_sum >= threshold_95_percent)[0]

        start_day = int(start_day_index[0]) if len(start_day_index) > 0 else np.nan
        end_day = int(end_day_index[0]) if len(end_day_index) > 0 else np.nan

        season_results["Year"].append(year)
        season_results["Start_Day"].append(start_day)
        season_results["End_Day"].append(end_day)

    return pd.DataFrame(season_results)


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
        df_K50["Trend"].append(f'{results[i]["K50"]["Trend"]:.1f}')
        #df_K50["Intercept"].append(results[i]["K50"]["Intercept"])
        df_K50["R2"].append(f'{results[i]["K50"]["R2"]:.1f}')
        df_K50["min(K50)"].append(results[i]["K50"]["min(K50)"])
        df_K50["avg(K50)"].append(f'{results[i]["K50"]["avg(K50)"]:.1f}')
        df_K50["max(K50)"].append(results[i]["K50"]["max(K50)"])
        
        df_CP["Vrsta"].append(i)
        df_CP["Trend"].append(f'{results[i]["CP"]["Trend"]:.1f}')
        #df_CP["Intercept"].append(results[i]["CP"]["Intercept"])
        df_CP["R2"].append(f'{results[i]["CP"]["R2"]:.1f}')
        df_CP["min(CP)"].append(results[i]["CP"]["min(CP)"]),
        df_CP["avg(CP)"].append(f'{results[i]["CP"]["avg(CP)"]:.1f}')
        df_CP["max(CP)"].append(results[i]["CP"]["max(CP)"])

    df_K50 = pd.DataFrame(df_K50)
    df_CP = pd.DataFrame(df_CP)