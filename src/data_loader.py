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
    df = pd.DataFrame({'ds': dates, 'y': y})
    mask = ~np.isnan(y)
    df_train = df[mask]
    # Prophet requires at least 5 points to fit
    if len(df_train) < 5:
        y[np.isnan(y)] = np.nanmean(y)
        return y
    m = Prophet(yearly_seasonality=True, daily_seasonality=False)
    m.fit(df_train)
    future = df[['ds']]
    forecast = m.predict(future)
    y_pred = forecast['yhat'].values
    y[np.isnan(y)] = y_pred[np.isnan(y)]
    return y

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

def load_data(path):
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

    # --- Impute missing daily values with Prophet for each Type/Year ---
    processed_frames = []
    for pollen_type in tqdm(df_processed["Type"].unique(), desc="Prophet imputation"):
        for year in df_processed["Year"].unique():
            sub = df_processed[(df_processed["Type"] == pollen_type) & (df_processed["Year"] == year)].copy()
            if len(sub) == 0:
                continue
            dates_arr = sub["Date"].values
            y_arr = sub["Value"].values.astype(float)
            if np.any(np.isnan(y_arr)):
                y_arr = impute_missing_with_prophet(dates_arr, y_arr)
                sub["Value"] = y_arr
            processed_frames.append(sub)
    df_final = pd.concat(processed_frames, ignore_index=True)

    # Now fill any remaining NaN with zero (should be rare)
    df_final["Value"] = df_final["Value"].fillna(0)

    # Add group info
    tqdm.pandas(desc="Assigning Skupina")
    df_final["Skupina"] = df_final["Type"].progress_apply(lambda x: rastline_skupine.get(x, "Unknown"))

    return df_final

def process_data(location):
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
        df_processed = load_data(path)
    
        # Save processed data to a CSV file
        output_path = os.path.join("data", "processed")
        os.makedirs(output_path, exist_ok=True)
        df_processed.to_csv(os.path.join(output_path, f"{location}_processed.csv"), index=False)
    return df_processed