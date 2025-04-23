import pandas as pd
import matplotlib.pyplot as plt
import json
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        Loads the intensity mapping JSON into memory.
        """
        with open(self.mapping_path, 'r') as f:
            self.intensity_mapping = json.load(f)

    def load_and_clean_dataset(self):
        """
        Reads the raw CSV, filters out invalid values, drops unneeded columns,
        and maps raw intensity labels to normalized values.
        """
        
        df = pd.read_csv(self.data_path)
        
        if "cleaned" not in self.data_path:
          
          df = df[df['Intensity_Value'] != '-9999'] 
          
          cols_to_drop = [
              'Update_Datetime', 'Site_ID', 'Elevation_in_Meters', 'Genus',
              'Species', 'Common_Name', 'Kingdom', 'Phenophase_Status', 'Abundance_Value'
          ]

          df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

          df['Intensity_Value'] = df['Intensity_Value'].map(self.intensity_mapping)

        df.to_csv(self.project_data_path + "cleaned_status_intensity_observation_data.csv") #csv 

        self.df = df

    def pollen_only(self):
      """
      Filters the dataset to include only rows with reproductive phenophases
      loaded from the JSON file provided in self.Phenophase_path.
      """

      # Check if dataset is loaded
      if self.df is None:
          raise ValueError("Dataset not loaded. Please run load_and_clean_dataset() first.")
      
      # Load phenophase categories from JSON
      with open(self.Phenophase_path, 'r') as f:
          phenophases = json.load(f)
      
      reproductive_phenophases = phenophases.get("Reproductive phenophases", [])

      # Filter the dataframe to only reproductive phenophases
      pollen_df = self.df[self.df['Phenophase_Description'].isin(reproductive_phenophases)].copy()

      pollen_df.to_csv(self.project_data_path + "pollen_only_data.csv", index=False)

      self.pollen_df = pollen_df

      print("Pollen-only dataset created with", len(pollen_df), "rows.")
      print("Original dataset has", len(self.df), "rows.")
  
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
        if 'Latitude' not in self.pollen_df.columns or 'Longitude' not in self.pollen_df.columns:
            raise ValueError("DataFrame must contain 'Latitude' and 'Longitude' columns.")

        print("Fetching county FIPS codes...")

        results = []

        # Using tqdm to create a progress bar
        for lat, lon in tqdm(zip(self.pollen_df['Latitude'], self.pollen_df['Longitude']), 
                             total=len(self.pollen_df), desc="Fetching FIPS", ncols=100):
            fips = self.get_fips(lat, lon)  # Get FIPS from cache or API
            results.append(fips)

        self.fips_df = self.pollen_df.copy()
        self.fips_df['county_fips'] = results

        # Drop all 'Unnamed' columns
        self.fips_df = self.fips_df.loc[:, ~self.fips_df.columns.str.contains('^Unnamed')]

        print("Finished fetching county FIPS codes.")
        print(self.fips_df)

        # Save to the provided file path
        self.fips_df.to_csv(
            self.project_data_path + "cleaned_countyflips_status_intensity_observation_data.csv",
            index=False
        )

    def add_land_cover_info(self):
      """
      Adds land cover info to self.fips_df by matching county_fips to GEOID in the land cover table.
      Stores the result in self.final_df.

      Parameters
      ----------
      land_cover_path : str
          Path to the land cover CSV file with 'GEOID' and 'Max_LCC_Name' columns.
      """
      # Load Brooke's land cover table
      land_cover_df = pd.read_excel(self.land_cover_path)

      # Check required columns exist
      if "GEOID" not in land_cover_df.columns or "Max_LCC_Name" not in land_cover_df.columns:
          raise ValueError("Land cover file must contain 'GEOID' and 'Max_LCC_Name' columns.")

      # Create hash table: key = county (GEOID), value = land cover type
      land_cover_dict = {}
      for _, row in land_cover_df.iterrows():
          geoid = str(row["GEOID"]).zfill(5)  # ensure FIPS are 5-digit strings
          land_cover = row["Max_LCC_Name"]
          land_cover_dict[geoid] = land_cover

      # Create a new DataFrame with land cover info added
      self.final_df = self.fips_df.copy()
      self.final_df["land_cover_type"] = self.final_df["county_fips"].astype(str).map(land_cover_dict)

      print("Land cover types added using hash table.")
      print(self.final_df[["county_fips", "land_cover_type"]].head())

      self.final_df.to_csv(
          self.project_data_path + "cleaned_data.csv",
          index=False
      )


    def plot_intensity_by_year(self, year):
        """
        Plot combined, smoothed intensity counts by day for a specific year using final_df,
        overlaid with a translucent line showing daily Tmax.
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

        plt.figure(figsize=(12,6))
        # translucent line for Tmax
        plt.plot(smooth_daily_tmax.index, smooth_daily_tmax.values, color='grey', alpha=0.4, linewidth=1, label='Tmax (°C)')
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