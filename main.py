# main.py
import pandas as pd
import src.data_loader as dl
import src.analysis as an
import src.plotting as pl
import os

S1 = True
S2 = True
S3 = True
S4 = True
S5 = True
S6 = True
S7 = True


def main():
    """
    Main function to run the complete pollen data analysis workflow.
    It iterates through a list of locations, processes the data for each,
    generates all analyses and plots, and saves the results.
    """
    # List of locations to be analyzed
    locations = ["Ljubljana", "Maribor", "Primorje"]
    
    # A list to store processed dataframes for later cross-regional analysis
    processed_dfs = []

    print("Starting the pollen analysis workflow...")

    for location in locations:
        try:
            print(f"\n{'='*50}")
            print(f"--- PROCESSING LOCATION: {location.upper()} ---")
            print(f"{'='*50}\n")

            # Step 1: Load and process data for the current location.
            print(f"Step 1: Loading and processing data for {location}...")
            df_processed = dl.process_data(location)
            
            # If data processing is successful, proceed with analysis
            if df_processed is not None:
                processed_dfs.append(df_processed) # Add processed df to the list
                print("Data processing complete.")

                # Step 2: Generate and save global overview plots.
                if S2:
                    step_name = "Step_2_Global_Overview"
                    print(f"Step 2: Generating global data plots to '{step_name}'...")
                    pl.plot_global_data(df_processed, location, step_name)
                    print("Global data plots saved.")

                # Step 3: Generate and save data completeness analysis plots.
                if S3:
                    step_name = "Step_3_Completeness"
                    print(f"Step 3: Analyzing and plotting data completeness to '{step_name}'...")
                    pl.plot_completeness_analysis(location, step_name)
                    print("Completeness analysis plots saved.")
                
                if S4:
                    # Step 4: Perform detailed, type-specific analysis.
                    step_name = "Step_4_Type_Specific"
                    print(f"Step 4: Performing type-specific activation analysis...")
                    results_1, results_2, colors = an.type_specific_activation(df_processed, location, step_name=step_name)
                    an.auc_and_ci_start_stop(df_processed, results_1, location, step_name=step_name)
                    print("Type-specific analysis complete.")
                

                # Step 5: Display summary results and generate detailed plots.
                step_name = "Step_5_Detailed_Plots"
                print(f"Step 5: Showing summary results and generating detailed plots to '{step_name}'...")
                # The show_results function prints tables and returns them as DataFrames.
                df_k50, df_cp = an.show_results(results_1)
                
                print("\nK50 Trend Analysis Summary:")
                print(df_k50)
                
                print("\nCumulative Pollen (CP) Trend Analysis Summary:")
                print(df_cp)

                # Plot the results of the second analysis part (start, end, rate, etc.)
                pl.plot_auc_and_ci(results_2, colors, location, step_name)
                print(f"Detailed analysis plots for {location} saved.")
            else:
                print(f"Data processing failed for {location}. Skipping subsequent steps.")

            print(f"\n--- SUCCESSFULLY FINISHED PROCESSING FOR {location.upper()} ---")

        except FileNotFoundError as e:
            print(f"ERROR for {location}: {e}. Skipping this location.")
        except Exception as e:
            print(f"An unexpected error occurred while processing {location}: {e}. Skipping this location.")

    print(f"\n{'='*50}")
    step_name = "Step_6_Cross_Regional_Correlation"
    print(f"Izvajanje korelacijske analize za vse regije v mapo '{step_name}'...")
    
    # Step 6: Perform cross-regional correlation analysis after all data is processed
    correlation_results = an.perform_cross_regional_correlation(processed_dfs, locations)

    # Save correlation results to a structured JSON file
    if correlation_results:
        an.save_correlation_results_to_json(correlation_results, step_name)

    # Step 7: Plot the correlation heatmaps
    if correlation_results:
        pl.plot_correlation_with_time_series(correlation_results, step_name)
    
    print(f"\n{'='*50}")
    print("Workflow finished for all locations.")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()