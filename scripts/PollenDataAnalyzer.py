import pandas as pd
import matplotlib.pyplot as plt
import json
import requests
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
    def __init__(self, data_path: str, mapping_path: str):
        """
        Initialize the analyzer with paths to the dataset and intensity mapping.

        Parameters
        ----------
        data_path : str
            Path to the raw CSV file containing status and intensity observations.
        mapping_path : str
            Path to the JSON file mapping raw intensity labels to normalized categories.
        """
        self.data_path = data_path
        self.mapping_path = mapping_path
        self.df = None
        self.intensity_mapping = None
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

        df.to_csv("/Applications/Home/2025 Spring/GEOG398E/Project data/dataset-for-roi/cleaned_status_intensity_observation_data.csv") #csv 

        self.df = df

    def pollen_only(self):
      """
      Filters the dataset to include only rows with reproductive phenophases.
      """
      reproductive_phenophases = [
        "Flowers or flower buds",
        "Open flowers",
        "Pollen release",
        "Pollen cones",
        "Open pollen cones"
      ]

      # Check if self.df exists
      if not hasattr(self, 'df'):
          raise ValueError("Dataset not loaded. Please run load_and_clean_dataset() first.")
      
      # Filter based on Phenophase_Description
      pollen_df = self.df[self.df['Phenophase_Description'].isin(reproductive_phenophases)].copy()
      
      # Optional: Save to CSV [for kathy]
      pollen_df.to_csv("/Applications/Home/2025 Spring/GEOG398E/Project data/dataset-for-roi/pollen_only_data.csv", index=False)

      # Store or return
      self.pollen_df = pollen_df
      print("Pollen-only dataset created with", len(pollen_df), "rows.")
      print("original dataset has", len(self.df), "row.")

    def lag_long_to_county(self):
        """
        Adds a 'county_fips' column to self.df based on latitude and longitude using multithreading,
        and prints the updated DataFrame.
        """
        def get_fips(lat, lon):
            try:
                url = f'https://geo.fcc.gov/api/census/area?lat={lat}&lon={lon}&censusYear=2020&format=json'
                r = requests.get(url, timeout=5)
                data = r.json()
                return data['results'][0]['county_fips'] if data['results'] else None
            except Exception as e:
                print(f"Error fetching FIPS for ({lat}, {lon}): {e}")
                return None

        if 'Latitude' not in self.df.columns or 'Longitude' not in self.df.columns:
            raise ValueError("DataFrame must contain 'Latitude' and 'Longitude' columns.")
        
        print("Fetching county FIPS codes in parallel...")

        coords = list(zip(self.df['Latitude'], self.df['Longitude']))
        results = [None] * len(coords)

        # Use ThreadPoolExecutor to parallelize API requests
        with ThreadPoolExecutor(max_workers=20) as executor:
            future_to_index = {
                executor.submit(get_fips, lat, lon): i
                for i, (lat, lon) in enumerate(coords)
            }
            for future in as_completed(future_to_index):
                i = future_to_index[future]
                try:
                    results[i] = future.result()
                except Exception as exc:
                    print(f"Error at index {i}: {exc}")
                    results[i] = None

        self.df['county_fips'] = results

        print(self.df)
        #[for kathy]
        self.df.to_csv("/Applications/Home/2025 Spring/GEOG398E/Project data/dataset-for-roi/cleaned_countyflips_status_intensity_observation_data.csv")


      


    def plot_intensity_counts(self):
        """
        Plots the raw counts of high and low intensity observations for each day of the year.
        """
        
        df = self.df.copy()

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

        df = self.df.copy()

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

      df = self.df.copy()

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
        df = self.df
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

    def plot_open_pollen_cones(self):
        df = self.df
        if 'Phenophase_Description' in df.columns:
            pc_mask = df['Phenophase_Description'].str.contains(r"^Pollen cones", na=False)
            op_mask = df['Phenophase_Description'].str.contains(r"^Open pollen cones", na=False)
            df_pc = df[pc_mask]
            df_op = df[op_mask]
        else:
            pid_pc = self.phenophase_ids.get('pollen_cones')
            pid_op = self.phenophase_ids.get('open_pollen_cones')
            if pid_pc is None or pid_op is None:
                raise KeyError("IDs missing for pollen_cones or open_pollen_cones.")
            df_pc = df[df['Phenophase_ID'] == pid_pc]
            df_op = df[df['Phenophase_ID'] == pid_op]
        total = df_pc.groupby('Day_of_Year').size()
        opened = df_op.groupby('Day_of_Year').size()
        pct = (opened / total).fillna(0).sort_index()
        plt.figure(figsize=(12,6))
        plt.plot(pct.index, pct.values, marker='s', color='orange')
        plt.title('Proportion of Open Pollen Cones by Day of Year')
        plt.xlabel('Day of Year'); plt.ylabel('Proportion Open')
        plt.ylim(0,1); plt.grid(True); plt.tight_layout(); plt.show()

    def plot_pollen_release_intensity(self):
        df = self.df
        if 'Phenophase_Description' in df.columns:
            mask = df['Phenophase_Description'].str.contains(r"^Pollen release", na=False)
            df_pr = df[mask]
        else:
            pid = self.phenophase_ids.get('pollen_release')
            if pid is None:
                raise KeyError("ID missing for 'pollen_release'.")
            df_pr = df[df['Phenophase_ID'] == pid]
        counts = df_pr.groupby(['Day_of_Year','Intensity_Value']).size().unstack(fill_value=0).sort_index()
        plt.figure(figsize=(12,6))
        for level, style in [('Little','-.'), ('Some','--'), ('Lots','-')]:
            if level in counts.columns:
                plt.plot(counts.index, counts[level], linestyle=style, label=level)
        plt.title('Pollen Release Intensity by Day of Year')
        plt.xlabel('Day of Year'); plt.ylabel('Number of Observations')
        plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()
