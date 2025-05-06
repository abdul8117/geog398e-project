import pandas as pd
import matplotlib.pyplot as plt
import json
import requests
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import re #using regex match pattern for phenophase matching

class PollenDataAnalyzer:
    """
    A class to load, clean, and visualize pollen intensity observation data.

    Methods
    -------
    load_data()
        Loads raw data from CSV and mapping JSON, applies initial filtering.
    plot_intensity_counts()
        Plots counts of 'high' and 'low' intensity observations by day of year.
    plot_normalized_intensity()
        Plots normalized proportions of 'high' and 'low' intensity observations by day of year.
    plot_monthly_observations()
        Plots total number of observations per month as a bar chart.
    """
    def __init__(self, data_path: str, mapping_path: str, Phenophase_path:str, land_cover_path: str, project_data_path: str):
        """
        Initialize the analyzer with paths to the dataset and intensity mapping.

        Parameters 
        ----------
        data_path : str
            Path to the raw CSV file containing status and intensity observations.
        mapping_path : str
            Path to the JSON file mapping raw intensity labels to normalized categories.
        Phenophase_path : str
            Path to JSON file mapping Phenophases
        land_cover_path : str
            Path to where our land cover data is stored for every county in our ROI
        project_data_path : str
            Directory where to store output dataframes
        df: Table
            current table that we are manipulating 
        intensity_mapping: Table
            table that was 
          
        

        pollen_df: Table
            table that store only the Reproductive phenophases (include pollen) related data 

        fips_df : Table
            pollen_df with added 'county_fips' column based on coordinates.

        self.final_df: Table
            final table with the 

        load_mapping: ...?
        load_and_clean_mapping: ...?
        """
        self.data_path = data_path 
        self.mapping_path = mapping_path
        self.Phenophase_path = Phenophase_path
        self.land_cover_path = land_cover_path
        self.project_data_path = project_data_path

        self.df = None
        self.intensity_mapping = None
        self.pollen_df = None
        self.fips_df = None
        self.final_df = None
        self.fips_cache = {}
        self._load_mapping()
        self.load_and_clean_dataset()
        
    def _load_mapping(self):
        """
        Loads the intensity mapping JSON into memory and sets up both binary and numerical mappings.
        """
        # Load the original binary mapping
        with open(self.mapping_path, 'r') as f:
            self.binary_mapping = json.load(f)
        
        # Create numerical mapping (scale 1-10)
        self.numerical_mapping = {
            # Count-based categories
            "Less than 3": 0, 
            "3 to 10": 1,
            "11 to 100": 2,
            "101 to 1,000": 4,
            "1,001 to 10,000": 6,
            "More than 10,000": 8,
            "More than 10": 2,
            "More than 1,000": 5,
            
            # Percentage-based categories
            "Less than 5%": 0,
            "5-24%": 1,
            "25-49%": 3,
            "50-74%": 5,
            "75-94%": 7,
            "95% or more": 9,
            
            # Qualitative categories
            "Little": 1,
            "Some": 3,
            "Lots": 6,
            "Peak flower": 8,
            "Peak opening": 8,
            "Peak pollen": 8
        }
        
        # Set the default mapping to binary
        self.intensity_mapping = self.binary_mapping

    def load_and_clean_dataset(self, use_numerical_mapping=True):
        """
        Reads the raw CSV, filters out invalid values, drops unneeded columns,
        and maps raw intensity labels to normalized values.
        
        Parameters:
        -----------
        use_numerical_mapping : bool, default=True
            If True, uses a more granular numerical intensity mapping (1-10 scale)
            If False, uses the original binary mapping (high/low)
        """
        
        df = pd.read_csv(self.data_path)
        
        if "cleaned" not in self.data_path:
            # Filter out invalid values
            df = df[df['Intensity_Value'].astype(str) != '-9999']
            
            # Columns to drop
            cols_to_drop = [
                'Update_Datetime', 'State', 'Plant_Nickname', 'ObservedBy_Person_ID', 'Site_ID', 
                'Elevation_in_Meters', 'Genus', 'Species', 'Common_Name', 'Kingdom', 
                'Phenophase_Status', 'Abundance_Value'
            ]

            df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

            # Set which mapping to use based on parameter
            mapping_to_use = self.numerical_mapping if use_numerical_mapping else self.binary_mapping
            
            # Apply the selected mapping
            df['Intensity_Value'] = df['Intensity_Value'].map(mapping_to_use)
            #trying to dropp the empty/na intensity/it seem like it doesn't work
            df.dropna(subset=['Intensity_Value'])
            df = df[df['Intensity_Value'].astype(str).str.strip() != '']
            
            # Create a new column to store the original binary mapping for reference
            # if use_numerical_mapping:
            #     df['Intensity_Binary'] = df['Intensity_Value'].map(self.binary_mapping)
            
        # Save the cleaned data
        output_file = "cleaned_status_intensity_observation_data.csv"
        # if use_numerical_mapping:
        #     output_file = "cleaned_numerical_intensity_data.csv"
        
        df.to_csv(os.path.join(self.project_data_path, output_file))
        self.df = df
        
        print(f"Data cleaned with {'numerical (1-10)' if use_numerical_mapping else 'binary (high/low)'} intensity mapping.")
        if use_numerical_mapping:
            print("Intensity value distribution:")
            print(df['Intensity_Value'].value_counts().sort_index())
    def pollen_only(self):
        """
        Filters the dataset to include only rows with reproductive phenophases,
        including "Open flowers (lilac)" and other variations.
        """
        # Check if dataset is loaded
        if self.df is None:
            raise ValueError("Dataset not loaded. Please run load_and_clean_dataset() first.")
        
        # Load phenophase categories from JSON
        with open(self.Phenophase_path, 'r') as f:
            phenophases = json.load(f)
        
        reproductive_phenophases = phenophases.get("Reproductive", [])
        
        # Create a more flexible pattern that matches the base phrases anywhere in the string
        # This will catch "Open flowers (lilac)" and similar variations
        patterns = []
        for p in reproductive_phenophases:
            # Create pattern that matches the base term regardless of what follows in parentheses
            patterns.append(rf'{re.escape(p)}(\s*\([^)]*\))?')
        
        # Combine patterns with OR operator
        combined_pattern = '|'.join(patterns)
        
        # Filter using str.contains instead of str.match to find the pattern anywhere in the string
        mask = self.df['Phenophase_Description'].str.contains(combined_pattern, regex=True, na=False)
        
        # Apply the mask to filter the dataframe
        pollen_df = self.df[mask].copy()
        
        # Save the filtered DataFrame to CSV
        pollen_df.to_csv(self.project_data_path + "pollen_only_data.csv", index=False)
        
        # Update the pollen_df attribute
        self.pollen_df = pollen_df
        
        # Print information about the resulting filtered dataset
        print("Pollen-only dataset created with", len(pollen_df), "rows.")
        print("Original dataset has", len(self.df), "rows.")
        print("Unique phenophase descriptions in pollen dataset:")
        print(pollen_df["Phenophase_Description"].unique())

  
    def get_fips(self, lat, lon):
      """
      Fetch county FIPS code using latitude and longitude, caching results in-memory
      to avoid redundant API calls.
      """
      key = (round(lat, 4), round(lon, 4))

      # Check if FIPS for this coordinate pair is already cached
      if key in self.fips_cache:
          return self.fips_cache[key]

      try:
          url = f'https://geo.fcc.gov/api/census/area?lat={lat}&lon={lon}&censusYear=2020&format=json'
          r = requests.get(url, timeout=5)
          data = r.json()
          fips = data['results'][0]['county_fips'] if data['results'] else None

          # Cache the result
          self.fips_cache[key] = fips

          return fips

      except Exception as e:
        #   print(f"Error fetching FIPS for ({lat}, {lon}): {e}")
          return None

    def lat_long_to_county(self):
        """
        Adds a 'county_fips' column to self.pollen_df based on latitude and longitude
        using hashmap caching for FIPS codes, and stores the result in self.fips_df.
        """
        if self.pollen_df is None:
          raise ValueError("Pollen-only dataset not created. Run pollen_only() first.")

        df = self.pollen_df.copy()  

        if df is None:
            raise ValueError("Pollen-only dataset not created. Run pollen_only() first.")
        if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
            raise ValueError("DataFrame must contain 'Latitude' and 'Longitude' columns.")

        print("Fetching county FIPS codes...")

        results = []

        # Using tqdm to create a progress bar
        for lat, lon in tqdm(zip(df['Latitude'], df['Longitude']), 
                            total=len(df), desc="Fetching FIPS", ncols=100):
            fips = self.get_fips(lat, lon)  # Get FIPS from cache or API
            results.append(fips)

        df['county_fips'] = results

        # Drop all 'Unnamed' columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        self.fips_df = df
        self.df = df

        print("Finished fetching county FIPS codes.")
        print(self.df)

        # Save to the provided file path
        self.df.to_csv(
            self.project_data_path + "cleaned_countyflips_status_intensity_observation_data.csv",
            index=False
        )


    def add_land_cover_info(self):
        """
        Adds land cover info to self.fips_df by matching county_fips to GEOID in the land cover table.
        Stores the result in self.df.

        The land cover file can be a CSV or Excel file.
        """
        # Determine file extension and load accordingly
        if self.land_cover_path.endswith('.csv'):
            land_cover_df = pd.read_csv(self.land_cover_path)
        elif self.land_cover_path.endswith(('.xlsx', '.xls')):
            land_cover_df = pd.read_excel(self.land_cover_path)
        else:
            raise ValueError("Unsupported land cover file format. Please use .csv or .xlsx/.xls")

        # Check required columns
        if "GEOID" not in land_cover_df.columns or "Max_LCC_Name" not in land_cover_df.columns:
            raise ValueError("Land cover file must contain 'GEOID' and 'Max_LCC_Name' columns.")

        # Build hash table: GEOID → land cover type
        land_cover_dict = {
            str(row["GEOID"]).zfill(5): row["Max_LCC_Name"]
            for _, row in land_cover_df.iterrows()
        }

        # Add land cover info to dataframe
        df = self.df.copy()
        df["land_cover_type"] = df["county_fips"].astype(str).map(land_cover_dict)

        print("Land cover types added using hash table.")
        print(df[["county_fips", "land_cover_type"]].head())

        # Save updated dataframe
        output_path = os.path.join(self.project_data_path, "cleaned_data.csv")
        df.to_csv(output_path, index=False)

        self.df = df
        self.final_df = df


    def split_observation_date(self):
        """
        Splits 'Observation_Date' into separate 'Year', 'Month', and 'Day' columns.
        Updates self.final_df and saves the result as 'final_df.csv'.
        """
        if self.final_df is None:
            raise ValueError("final_df is not loaded. Please ensure it is initialized before calling this method.")

        # Show progress bar while splitting
        tqdm.pandas(desc="Splitting Observation_Date")

        # Convert Observation_Date to datetime
        self.final_df['Observation_Date'] = pd.to_datetime(self.final_df['Observation_Date'], errors='coerce')

        # Extract year, month, day with progress bar
        self.final_df['Year'] = self.final_df['Observation_Date'].progress_apply(lambda x: x.year if pd.notnull(x) else None)
        self.final_df['Month'] = self.final_df['Observation_Date'].progress_apply(lambda x: x.month if pd.notnull(x) else None)
        self.final_df['Day'] = self.final_df['Observation_Date'].progress_apply(lambda x: x.day if pd.notnull(x) else None)

        # Save to CSV
        output_path = os.path.join(self.project_data_path, "final_df.csv")
        self.final_df.to_csv(output_path, index=False)
        print(f"Updated final_df with Year, Month, Day columns and saved to {output_path}")

    def plot_intensity_by_year(self, year):
        """
        Plot combined, smoothed intensity counts by day for a specific year using final_df,
        overlaid with a grey line showing daily Tmax.
        """
        df = self.final_df.copy()
        df['Observation_Date'] = pd.to_datetime(df['Observation_Date'], errors='coerce')
        df_yr = df[df['Observation_Date'].dt.year == year]

        # intensity counts
        counts = df_yr.groupby(['Day_of_Year','Intensity_Value']).size().unstack(fill_value=0).sort_index()
        total_counts = counts.sum(axis=1)
        smooth_counts = total_counts.rolling(window=7, center=True, min_periods=1).mean()

        # daily Tmax
        if 'Tmax' not in df_yr.columns:
            raise KeyError("'Tmax' column not found in final_df")
        
        daily_tmax = df_yr.groupby('Day_of_Year')['Tmax'].mean().sort_index()
        smooth_daily_tmax = daily_tmax.rolling(window=7, center=True, min_periods=1).mean()

        daily_tmin = df_yr.groupby('Day_of_Year')['Tmin'].mean().sort_index()
        smooth_daily_tmin = daily_tmin.rolling(window=7, center=True, min_periods=1).mean()

        plt.figure(figsize=(12,6))

        # grey line for Tmax
        plt.plot(smooth_daily_tmax.index, smooth_daily_tmax.values, color='grey', alpha=0.4, linewidth=1, label='Tmax (°C)')
        
        # line for tmin
        plt.plot(smooth_daily_tmin.index, smooth_daily_tmin.values, color='brown', alpha=0.4, linewidth=1, label='Tmin (°C)')

        # smooth intensity line
        plt.plot(smooth_counts.index, smooth_counts.values, color='red', linewidth=2, label='Observations (smoothed)')

        plt.title(f'Reproductive Phenophases Observations & Tmax in {year}')
        plt.xlabel('Day of Year')
        plt.ylabel('Count / Tmax (°C)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_intensity_counts(self):
        """
        Plots the raw counts of high and low intensity observations for each day of the year.
        """
        
        df = self.final_df.copy()

        counts = df[df['Intensity_Value'].notna()] \
            .groupby(['Day_of_Year', 'Intensity_Value']) \
            .size() \
            .unstack(fill_value=0) \
            .sort_index()
        
        plt.figure(figsize=(12, 6))

        for level, color in [('high', 'red'), ('low', 'green')]:
            if level in counts.columns:
                plt.plot(counts.index, counts[level], label=level.capitalize(), color=color)

        plt.title('Pollen Intensity Observations Throughout the Year')
        plt.xlabel('Day of Year')
        plt.ylabel('Number of Observations')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_normalized_intensity(self):
        """
        Plots the proportion of high and low intensity observations for each day of the year.
        """

        df = self.final_df.copy()

        daily = df[df['Intensity_Value'].notna()] \
            .groupby(['Day_of_Year', 'Intensity_Value']) \
            .size() \
            .unstack(fill_value=0)
        
        totals = daily.sum(axis=1)
        proportions = daily.div(totals, axis=0)

        plt.figure(figsize=(12, 6))

        for level, color in [('high', 'red'), ('low', 'green')]:
            if level in proportions.columns:
                plt.plot(proportions.index, proportions[level], label=level.capitalize(), color=color)

        plt.title('Proportion of Pollen Intensity Observations Throughout the Year')
        plt.xlabel('Day of Year')
        plt.ylabel('Proportion of Observations')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_monthly_observations(self):
      """
      Plots the total number of observations for each month as a bar chart.
      """

      df = self.final_df.copy()

      df['Observation_Date'] = pd.to_datetime(df['Observation_Date'], errors='coerce')
      df = df.dropna(subset=['Observation_Date'])
      df['Month'] = df['Observation_Date'].dt.month

      monthly_counts = df['Month'].value_counts().sort_index()

      plt.figure(figsize=(10, 5))
      monthly_counts.plot(kind='bar')
      plt.title('Total Observations Per Month')
      plt.xlabel('Month')
      plt.ylabel('Number of Observations')
      plt.xticks(ticks=range(1, 13), labels=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], rotation=45)
      plt.grid(axis='y')
      plt.tight_layout()
      plt.show()

    def plot_pollen_cone_counts(self):
        """
        Plots counts of fresh pollen cones by day of year.
        Handles descriptions like 'Pollen cones (conifers)'.
        """
        df = self.final_df
        if 'Phenophase_Description' in df.columns:
            mask = df['Phenophase_Description'].str.contains(r"^Pollen cones", na=False)
            df_pc = df[mask]
        else:
            # fallback to ID-based filtering
            pid = self.phenophase_ids.get('pollen_cones')
            if pid is None:
                raise KeyError("Phenophase_Description column missing and no ID provided for 'pollen_cones'.")
            df_pc = df[df['Phenophase_ID'] == pid]
        counts = df_pc.groupby('Day_of_Year').size().sort_index()
        plt.figure(figsize=(12,6))
        plt.plot(counts.index, counts.values, marker='o')
        plt.title('Fresh Pollen Cone Counts by Day of Year')
        plt.xlabel('Day of Year'); plt.ylabel('Count of Fresh Pollen Cones')
        plt.grid(True); plt.tight_layout(); plt.show()


