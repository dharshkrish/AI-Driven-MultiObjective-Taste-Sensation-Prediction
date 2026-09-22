
# AI Driven Multi-Taste Sensation Predictor

A Streamlit-based AI application that predicts and analyzes taste sensations from food items and chemical compounds. The application provides taste profiles for Bitter, Sweet, Umami, and Other sensations, along with visualizations and AI-based suggestions.

## Tech Stack

### Frontend / User Interface

* Streamlit
* HTML/CSS styling
* Matplotlib

### Programming Language

* Python

### Data Processing

* Pandas
* NumPy

### Database / Data Sources

* CSV-based Food Taste Database
* CSV-based Food Compounds Database

### Tools

* Jupyter Notebook
* VS Code / PyCharm
* Git
* GitHub

## Features

* Supports two prediction modes:

  * **Food Lookup (DB-first)**
  * **Chemical Compound Mode**
* Allows users to enter multiple food items at once.
* Looks up food items from an external food taste database.
* Identifies the dominant taste sensation of a food.
* Provides taste profiles for:

  * Bitter
  * Sweet
  * Umami
  * Other

* Generates taste prediction scores for chemical compounds.
* Identifies the dominant predicted taste.
* Provides AI-based compound suggestions based on the predicted taste.
* Provides healthy alternatives and notes for food items.
* Displays prediction results in a structured table.
* Provides graphical visualization of taste scores.
* Includes a dedicated AI Suggestions page.
* Uses Streamlit session state to preserve prediction results between page changes.
* Supports database loading from multiple possible project locations.

## Application Modes

### 1. Food Lookup Mode

The Food Lookup mode allows users to enter one or more food items.

The application searches the food taste database and retrieves:

* Food Item
* Compound
* Taste Profile
* Dominant Taste
* Healthy Alternative
* Additional Notes

Example:

```text
Apple
Coffee
Chocolate
Cheese
```

The application then generates a taste profile for each food item.

### 2. Chemical Compound Mode

The Chemical Compound mode allows users to enter chemical structures 

Example:

```text
CCO
CC(=O)OC1=CC=CC=C1C(=O)O
```

For each compound, the application generates taste scores for:

```text
Bitter
Sweet
Umami
Other
```

The application also identifies the dominant taste and provides a corresponding suggestion.

## Application Pages

### Input & Table

Displays the prediction results in a structured table.

The table changes according to the selected mode and contains the relevant food or chemical compound information.

### Graph Visualization

Displays a bar graph comparing the predicted taste scores for:

* Bitter
* Sweet
* Umami
* Other

This provides a visual representation of the taste profile for each input.

### AI Suggestions

Displays taste-based suggestions in a card-style interface.

For food items, the application displays:

* Food item
* Dominant taste
* Healthy alternative
* Additional notes

For chemical compounds, the application displays:


* Predicted dominant taste
* Compound-based suggestion



This database is used for compound-related taste suggestions.

## Project Structure


MultiTaste/
│
├── app.py
│
├── data/
│   └── ExternalDBs/
│       ├── food_taste_database_extended.csv
│       └── food_compounds_db.csv
│
├── README.md
│
└── screenshots/
    ├── food_lookup.png
    ├── prediction_results.png
    ├── graph_visualization.png
    └── ai_suggestions.png




## Running the Application (Local Setup)



### Install Required Libraries

Make sure Python is installed on your system.

```bash
pip install streamlit pandas numpy matplotlib
```

### Run the Streamlit Application

If your main Python file is `app.py`:

```bash
streamlit run app.py
```

The application will start locally and can be accessed at:

```text
http://localhost:8501
```

## Using the Application

### Step 1 — Select Prediction Mode


### Step 2 — Enter Input

For **Food Lookup**, enter food items one per line.

For **Compound Mode**, enter SMILES strings one per line.

### Step 3 — Run Prediction

Click:

```text
🔍 Lookup Taste
```

for Food Lookup mode.

Or:

```text
🔮 Predict Taste
```

for Chemical Compound mode.

### Step 4 — View Results

Use the sidebar to switch between:

```text
Input & Table
Graph Visualization
AI Suggestions
```

## Output Screenshots


### Prediction Results
<img width="1919" height="920" alt="Screenshot 2025-11-10 000940" src="https://github.com/user-attachments/assets/a4f45e18-1906-4dd7-81db-3b4f01eef3dc" />



### Graph Visualization

<img width="1918" height="904" alt="Screenshot 2025-11-10 001043" src="https://github.com/user-attachments/assets/6cae79c6-b3d7-4297-9552-103d110016c4" />


### AI Suggestions

<img width="1913" height="891" alt="Screenshot 2025-11-10 001105" src="https://github.com/user-attachments/assets/e8fe2bb1-3321-4dc1-a13c-be5d17f27000" />


### Chemical Compound Prediction

<img width="1750" height="898" alt="Screenshot 2025-11-09 210518" src="https://github.com/user-attachments/assets/46a50aba-cbd9-452f-8bc7-6ce8608e6348" />






## Applications

The AI Driven Multi-Taste Sensation Predictor can be used for:

* Food and flavor analysis
* Taste classification
* Food product research
* Chemical compound taste analysis
* Sensory science experiments
* Food technology projects
* Educational machine learning applications
* Exploring relationships between food compounds and taste sensations

