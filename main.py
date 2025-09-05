# main.py
import pandas as pd
import src.data_loader as dl
import src.analysis as an
import src.plotting as pl

def main():
    """
    Main function to run the complete pollen data analysis workflow.
    It iterates through a list of locations, processes the data for each,
    generates all analyses and plots, and saves the results.
    """
    # List of locations to be analyzed
    locations = ["Ljubljana", "Maribor", "Primorje"]

    print("Starting the pollen analysis workflow...")

    for location in locations:
        try:
            print(f"\n{'='*50}")
            print(f"--- PROCESSING LOCATION: {location.upper()} ---")
            print(f"{'='*50}\n")

            # Step 1: Load and process data for the current location.
            # The 'rastline_skupine' dictionary is defined within the data_loader module.
            print(f"Step 1: Loading and processing data for {location}...")
            df_processed = dl.process_data(location)
            print("Data processing complete.")

            # Step 2: Generate and save global overview plots.
            print("Step 2: Generating global data plots...")
            pl.plot_global_data(df_processed, location)
            print("Global data plots saved.")

            # Step 3: Generate and save data completeness analysis plots.
            print("Step 3: Analyzing and plotting data completeness...")
            pl.plot_completeness_analysis(location)
            print("Completeness analysis plots saved.")

            # Step 4: Perform detailed, type-specific analysis.
            print("Step 4: Performing type-specific activation analysis...")
            results_1, results_2, colors = an.type_specific_activation(df_processed, location)
            print("Type-specific analysis complete.")

            # Step 5: Display summary results and generate detailed plots.
            print("Step 5: Showing summary results and generating detailed plots...")
            # The show_results function prints tables and returns them as DataFrames.
            df_k50, df_cp = an.show_results(results_1)
            
            print("\nK50 Trend Analysis Summary:")
            print(df_k50)
            
            print("\nCumulative Pollen (CP) Trend Analysis Summary:")
            print(df_cp)

            # Plot the results of the second analysis part (start, end, rate, etc.)
            pl.plot_auc_and_ci(results_2, colors, location)
            print(f"Detailed analysis plots for {location} saved.")

            print(f"\n--- SUCCESSFULLY FINISHED PROCESSING FOR {location.upper()} ---")

        except FileNotFoundError as e:
            print(f"ERROR for {location}: {e}. Skipping this location.")
        except Exception as e:
            print(f"An unexpected error occurred while processing {location}: {e}. Skipping this location.")

    print(f"\n{'='*50}")
    print("Workflow finished for all locations.")
    print(f"{'='*50}")


if __name__ == "__main__":
    # This block ensures the main function is called only when the script is executed directly
    main()