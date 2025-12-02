import os
import pandas as pd
from tqdm import tqdm
import datetime
from prophet import Prophet
import numpy as np

rastline_skupine = {
    'JELŠA': 'Drevesa',
    'BREZA': 'Drevesa',
    'CIPRESOVKE': 'Grmičevje in pelinovke',
    'TRAVE': 'Trave in zelišča',
    'AMBROZIJA': 'Alergeni pleveli',
    'LESKA': 'Drevesa',
    'JESEN': 'Drevesa',
    'PLATANA': 'Drevesa',
    'KOPRIVOVKE': 'Grmičevje in pelinovke',
    'GABER': 'Drevesa',
    'HRAST': 'Drevesa',
    'BOR': 'Drevesa',
    'PRAVI KOSTANJ': 'Drevesa',
    'TRPOTEC': 'Trave in zelišča',
    'PELIN': 'Grmičevje in pelinovke',
    'BUKEV': 'Drevesa',
    'KISLICA': 'Trave in zelišča',
    'OLJKA': 'Drevesa'
}

def impute_missing_with_prophet(dates, y):
    """
    Imputes missing values (NaN) in a time series using Prophet,
    ensuring that the imputed values are non-negative.
    """
    df = pd.DataFrame({'ds': dates, 'y': y})
    mask = ~np.isnan(y)
    df_train = df[mask].copy() # Use .copy() to avoid SettingWithCopyWarning

    # --- Fallback to Mean Imputation ---
    # Prophet requires at least 5 points, but a more robust minimum
    # is often better for fitting seasonality (e.g., 20% of a year)
    if len(df_train) < 5: # Changed condition to a more robust minimum
        # Fallback to mean imputation if not enough data
        mean_y = np.nanmean(y)
        # Ensure the mean is non-negative before imputation
        mean_y = max(0, mean_y)
        y[np.isnan(y)] = mean_y
        return y

    # --- Prophet Modeling and Forecasting ---
    m = Prophet(yearly_seasonality = True, daily_seasonality = False)
    m.fit(df_train)
    
    future = df[['ds']]
    forecast = m.predict(future)
    y_pred = forecast['yhat'].values
    
    # --- Non-Negativity Constraint Implemented Here ---
    # Only the predictions for the missing spots need to be constrained.
    # Identify the indices where 'y' was originally missing
    missing_indices = np.where(np.isnan(y))[0]
    
    # Extract predictions for the missing spots
    predictions_for_missing = y_pred[missing_indices]
    
    # Apply the non-negativity constraint: replace any negative prediction with 0.
    # This is a simple and effective way to ensure valid pollen counts.
    constrained_predictions = np.maximum(0, predictions_for_missing)
    
    # --- Imputation ---
    # Apply the constrained predictions back to the original 'y' array
    # Note: 'y' is modified in place, which is fine as it's typically a
    # numpy array passed by reference.
    y[missing_indices] = constrained_predictions
    
    return y

def apply_prophet_imputation(group_df):
    """Applies prophet imputation to a single pollen type time series."""
    # Ensure columns are sorted by date for correct time series order
    group_df = group_df.sort_values(by="Date").reset_index(drop=True)
    
    # Get the dates (ds) and values (y) as arrays
    dates = group_df["Date"].values
    y_values = group_df["Value"].values.copy() # Use .copy() for safety

    # Apply the imputation function
    imputed_values = impute_missing_with_prophet(dates, y_values)
    
    # Assign the results back to the DataFrame
    group_df["Value"] = imputed_values
    
    return group_df

def load_raw_data(path):
    df_raw = pd.ExcelFile(path)
    dates = []
    types = []
    vales = []
    N = len(df_raw.sheet_names)
    for sheet in tqdm(df_raw.sheet_names,total = N, desc = "Reading sheets"):
        print(sheet)
        df = df_raw.parse(sheet, skiprows=0)
        if "Datum" in df.columns:
            pass
        else:
            df = df_raw.parse(sheet, skiprows=0)
        df["Datum"] = df["Datum"].apply(lambda x:str(x)[5:10])
        for year in range(2002,2024,1):
            dfs = df[["Datum",year]]
            for i, ii in dfs.iterrows():
                dates.append(ii["Datum"]+f"-{year}")
                types.append(sheet)
                vales.append(ii[year])
    print(dates[0:20])
    print(vales[0:20])
    print(types[0:20])
    draw_data = {"Date":dates,"Type":types,"Value":vales}
    draw_data= pd.DataFrame(draw_data)
    # Convert "Date" column to datetime and filter out invalid dates
    try:
        draw_data["Date"] = pd.to_datetime(draw_data["Date"], format="%m-%d-%Y", errors='coerce')
    except Exception as e:
        raise ValueError(f"Error converting 'Date' column to datetime: {e}")
    # Clean "Value" column
    draw_data["Value"] = draw_data["Value"].astype(str).str.replace("x", "").str.replace(",", "")
    draw_data["Value"] = pd.to_numeric(draw_data["Value"], errors="coerce")
    return draw_data

def load_data(path, Maribor = False):
    df_raw = pd.ExcelFile(path)
    dates = []
    types = []
    vales = []
    N = len(df_raw.sheet_names)
    for sheet in tqdm(df_raw.sheet_names,total = N, desc = "Reading sheets"):
        print(f"\n{sheet}")
        df = df_raw.parse(sheet, skiprows=0)
        if "Datum" in df.columns:
            pass
        else:
            df = df_raw.parse(sheet, skiprows=0)
        
        df["Datum"] = df["Datum"].apply(lambda x:str(x)[5:10])
        for year in range(2002,2025,1):
            dfs = df[["Datum",year]]
            for i, ii in dfs.iterrows():
                dates.append(ii["Datum"]+f"-{year}")
                types.append(sheet)
                vales.append(ii[year])
    df_procesed = {"Date":dates,"Type":types,"Value":vales}
    df_processed = pd.DataFrame(df_procesed)
    
    remove = []
    for i,ii in tqdm(df_processed.iterrows(),total = len(df_processed),desc="Checking dates"):
        try: # Try to convert a string to a date
            datetime.datetime.strptime(ii["Date"],"%m-%d-%Y")
        except: # If it fails, then it is a leap year
            remove.append(i)

    tqdm.pandas(desc="Removing non-leap 29/02/")
    df_processed = df_processed.drop(remove)

    tqdm.pandas(desc="Converting dates")
    df_processed["Date"] = pd.to_datetime(df_processed["Date"],format="%m-%d-%Y")
    df_processed["Year"] = df_processed["Date"].dt.year

    tqdm.pandas(desc="Cleaning values")
    df_processed["Value"] = df_processed["Value"].progress_apply(lambda x: str(x).replace("x",""))
    df_processed["Value"] = df_processed["Value"].progress_apply(lambda x: str(x).replace(",",""))
    df_processed["Value"] = pd.to_numeric(df_processed["Value"],errors="coerce")
# ----------------------------------------------------------------------
    # 💥 Imputation Step 💥
    # ----------------------------------------------------------------------
    print("\nStarting Prophet Imputation...")
    
    # Group by 'Type' and apply the imputation function to each group
    df_processed = df_processed.groupby("Type", group_keys=False).progress_apply(
        apply_prophet_imputation
    ).reset_index(drop=True)

    print("Imputation complete.")
    # ----------------------------------------------------------------------
    
    # The NaNs have now been filled by Prophet's predictions (or mean fallback),
    # so this line is likely no longer necessary, but if your Prophet model 
    # failed completely or you still have NaNs for some other reason, 
    # it provides a final safety net.
    # We will keep it but it should only apply to cases where Prophet could not run.
    df_processed["Value"] = df_processed["Value"].fillna(0) 

    df_processed["Skupina"] = df_processed["Type"].progress_apply(lambda x: rastline_skupine[x])
    if Maribor:
        #postavi vse vrednosti v januarju, februarju, novembru in decembru na 0
        df_processed.loc[df_processed["Date"].dt.month.isin([1,2,11,12]),"Value"] = 0
    return df_processed

def process_data(location, Maribor = False):
    """
    Manages the data loading and processing for a given location.
    """
    if location not in ["Ljubljana", "Maribor", "Primorje"]:
        raise ValueError("Data for the specified location is not available.")

    path = os.path.join("data", "raw", f"{location}2024.xlsx")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at {path}")
    #check if os.path.join(output_path, f"{location}_processed.csv") exists, if so, load it instead of processing again
    processed_file_path = os.path.join("data", "processed", f"{location}_processed.csv")
    if os.path.exists(processed_file_path):
        print(f"Loading processed data from {processed_file_path}...")
        df_processed = pd.read_csv(processed_file_path)
        df_processed["Date"] = pd.to_datetime(df_processed["Date"], format="%Y-%m-%d")
    else:
        print(f"Processing raw data from {path}...")
        if location == "Maribor":
            df_processed = load_data(path, Maribor = True)
        else:
            df_processed = load_data(path)
    
        # Save processed data to a CSV file
        output_path = os.path.join("data", "processed")
        os.makedirs(output_path, exist_ok=True)
        df_processed.to_csv(os.path.join(output_path, f"{location}_processed.csv"), index=False)
    return df_processed