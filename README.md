# GEOG398E Project: Predictive Modeling of Seasonal Pollen Exposure on the U.S. East Coast ☘️

📌 **Project Overview:** 
This project aims to build a predictive AI model that forecasts seasonal pollen intensity levels using historical weather
land cover, and phenophase data. Our region of interest (ROI) is the Southeastern United States, focusing on states including Maryland, West Virginia, Virginia, North Carolina, South Carolina, and Georgia.

🧹 **Data Cleaning Summary:** 
Standardized time ranges and key attributes across all datasets, Removed redundant or irrelevant variables, Merged datasets based on geographic and temporal alignment, Added land cover class per county from ArcGIS, Visualized seasonal trends with matplotlib to observe correlations between phenophases and pollen release

> ⚠️ **Note:** The datasets folder is not uploaded onto this repository due to GitHub's file size limit of 100 MB.
> 
> Download the datasets through this [link](https://drive.google.com/drive/folders/1g7zUb9xDMP870bw_ZHhOE8FMozYhIDdV?usp=sharing).

# Getting Started
Follow the steps below to set up and run the project on your local machine.

### 1. Clone the Repository

```bash
git clone https://github.com/abdul8117/geog398e-project.git
cd geog398e-project
```

### 2. Create a Virtual Environment
Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

macOS or Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Required Packages
```bash
pip install -r requirements.txt
```
On macOS or Linux, use ``pip3`` instead of ``pip``.

# Running the Code
To run any Python script in the terminal, use:
```bash
python script_name.py
```

On macOS or Linux, use ``python3`` instead of ``python``.

