# Country Aid Clustering

An end-to-end data science project that uses socioeconomic, health, demographic, and structural indicators to identify patterns of vulnerability across countries and support data-driven aid prioritisation.

## Business Problem

HELP International has approximately $10 million available for international aid allocation. With limited resources and many countries facing different combinations of economic and human-development challenges, the organisation needs a systematic way to identify countries with the greatest overall need.

The objective of this project is to group countries with similar development profiles using unsupervised machine learning and identify the clusters representing higher levels of vulnerability. These insights can then support, rather than replace, humanitarian and policy judgment when prioritising aid.

## Key Results

- The final solution uses **K-Means with 3 clusters** to separate countries into different vulnerability profiles.
- The **high-vulnerability cluster contains 51 countries** requiring the strongest further humanitarian review.
- Hierarchical clustering identified a **42-country core high-need group**, fully contained within the K-Means high-vulnerability cluster, providing additional validation of the prioritisation pattern.
- The highest-priority countries identified in the final ranking include **Haiti, Central African Republic, Sierra Leone, Chad, and Niger**.
- The model is intended to support aid prioritisation, not replace humanitarian or policy judgment.

## Dataset

The dataset used in this project was provided as part of the Scaler Data Science program. It contains socioeconomic, health, demographic, and trade indicators for 167 countries and corresponds to the publicly available **“Unsupervised Learning on Country Data”** dataset associated with the HELP International aid-allocation case study.

A public version of the same dataset is available on Kaggle: **Unsupervised Learning on Country Data**.

### Dtaset Features 

| Feature | Description |
|---|---|
| `child_mort` | Deaths of children under five per 1,000 live births |
| `exports` | Exports of goods and services as a percentage of GDP |
| `health` | Health expenditure as a percentage of GDP |
| `imports` | Imports of goods and services as a percentage of GDP |
| `income` | Net income per person |
| `inflation` | Annual growth rate of consumer prices |
| `life_expec` | Average life expectancy |
| `total_fer` | Average number of children per woman |
| `gdpp` | GDP per capita |

The features collectively represent different aspects of a country's economic capacity, human-development conditions, demographic pressure, and external economic structure.

## Project Workflow

The project was developed as an end-to-end machine learning workflow:

1. **Data Understanding and Audit**
   - Reviewed the dataset structure, data types, missing values, duplicates, and country-level uniqueness.
   - Checked extreme values to distinguish genuine socioeconomic outliers from possible data-quality issues.

2. **Exploratory Data Analysis**
   - Analysed feature distributions using histograms, box plots, and violin plots.
   - Examined skewness and influential observations.
   - Studied relationships between economic, health, demographic, and trade indicators using correlation analysis and scatter plots.
   - Used Tableau dashboards to explore human-development and economic vulnerability patterns interactively.

3. **Statistical Analysis**
   - Tested selected business-relevant relationships using Pearson and Spearman correlation.
   - Used the statistical results to support and validate patterns observed during EDA.

4. **Preprocessing**
   - Applied transformations to reduce skewness in selected variables.
   - Preserved the same preprocessing logic for both model training and inference.
   - Standardised the final modelling features before clustering.

5. **Clustering and Country Prioritisation**
   - Applied K-Means clustering to group countries with similar socioeconomic and development profiles.
   - Interpreted the resulting clusters using their feature characteristics.
   - Mapped the clusters to vulnerability levels to support aid prioritisation.

6. **Model Serving**
   - Saved the fitted scaler, clustering model, and cluster metadata as reusable model artifacts.
   - Built a Flask API with input validation, preprocessing, and prediction logic.

7. **Containerisation**
   - Packaged the API, preprocessing code, dependencies, and model artifacts into a Docker image.
   - Verified both successful predictions and validation errors inside the Docker container.

   ## Key EDA Findings

Exploratory analysis revealed several strong relationships and important distributional characteristics:

- **Income and GDP per capita were strongly positively correlated**, showing that countries with higher income levels generally also had greater economic output per person.
- **Child mortality and life expectancy showed a strong negative relationship**, indicating that countries with higher child mortality generally had substantially lower life expectancy.
- **Child mortality and total fertility were strongly positively related**, suggesting that countries with higher fertility rates also tended to experience greater child mortality.
- Several variables, including income, GDP per capita, inflation, exports, and imports, were strongly skewed and contained extreme observations.
- Extreme values were not automatically removed because many represented genuine country-level socioeconomic conditions rather than data errors.
- The relationship between income and life expectancy showed diminishing returns, with improvements becoming smaller at higher income levels.
- Health expenditure as a percentage of GDP showed only a weak direct relationship with life expectancy, indicating that expenditure level alone does not explain health outcomes.

These findings guided the preprocessing strategy and helped interpret the vulnerability profiles produced by the clustering model.

## Preprocessing and Feature Selection

The preprocessing strategy was designed to reduce the effect of highly skewed distributions while preserving meaningful country-level differences.

The following transformations were applied:

- `child_mort` → `log1p` transformation
- `gdpp` → `log1p` transformation
- `exports` → square-root transformation
- `imports` → square-root transformation
- `inflation` → Yeo-Johnson transformation

The Yeo-Johnson transformation for `inflation` uses the lambda value learned from the training dataset and reuses the same value during inference. This ensures that new observations are transformed consistently with the data used to train the clustering model.

After transformation, the modelling features were standardised before applying K-Means clustering so that variables measured on different numerical scales did not dominate the distance calculations.

### Final Modelling Features

The final clustering model uses:

- `child_mort`
- `exports`
- `health`
- `imports`
- `inflation`
- `life_expec`
- `total_fer`
- `gdpp`

`income` was explored extensively during EDA but was excluded from the final clustering feature set. Its strong relationship with GDP per capita introduced substantial overlap in the economic information represented by the model, so GDP per capita was retained as the primary economic-capacity feature.

## Clustering Results and Business Interpretation

K-Means clustering was used to group countries based on similarities across the final standardised socioeconomic and development indicators.

The resulting clusters were interpreted by comparing their underlying feature profiles rather than relying only on the numerical cluster labels. Each cluster was then mapped to a vulnerability level for easier business interpretation.

### Vulnerability Interpretation

The **higher-vulnerability group** is characterised by a combination of conditions such as:

- higher child mortality
- lower life expectancy
- higher fertility
- lower GDP per capita
- greater economic and developmental pressure across multiple indicators

Countries in this group represent the strongest candidates for further humanitarian review and potential aid prioritisation.

Countries in the other clusters generally demonstrate progressively stronger economic and human-development conditions and therefore lower relative vulnerability within this dataset.

### Aid Prioritisation

The clustering output is intended to provide a **decision-support framework**, not an automatic aid-allocation rule.

HELP International could use the high-vulnerability cluster as an initial shortlist and then incorporate additional information such as:

- current humanitarian emergencies
- population size and number of people affected
- conflict and political conditions
- existing international assistance
- feasibility of delivering aid
- country-specific needs not represented in the dataset

This allows the model to support systematic prioritisation while keeping the final allocation decision with humanitarian and policy experts.

## Architecture

```text
Raw Country Data
        ↓
EDA + Statistical Analysis
        ↓
Feature Transformation
        ↓
StandardScaler
        ↓
K-Means Clustering
        ↓
Cluster Interpretation
        ↓
Saved Model Artifacts
        ↓
Flask Prediction API
        ↓
Docker Container
```

During inference, the API follows the same preprocessing and scaling steps used during model development before assigning a country to a vulnerability cluster.

## Tech Stack

- **Python** — data analysis, preprocessing, modelling, and API logic
- **Pandas / NumPy** — data manipulation and numerical processing
- **SciPy** — statistical analysis and transformations
- **Scikit-learn** — scaling and K-Means clustering
- **Matplotlib / Seaborn** — exploratory visualisation
- **Tableau** — interactive analytical dashboards
- **Flask** — prediction API
- **Joblib** — model artifact persistence
- **Docker** — reproducible containerisation of the API and runtime environment
- **Git / GitHub** — version control and project repository

## Project Structure

```text
country-aid-clustering/
│
├── app/                  # Flask API
├── data/                 # Raw project data
├── models/               # Saved scaler, K-Means model, and metadata
├── notebooks/            # EDA, statistical analysis, and modelling notebooks
├── reports/              # Supporting project outputs
├── src/                  # Preprocessing, training, and validation code
├── Tableau/              # Tableau dashboard files
│
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── requirements-docker.txt
└── README.md
```

The project separates experimentation, reusable source code, model artifacts, API logic, and deployment files so that the workflow can be reproduced and maintained more easily.

## Running the API with Docker

Build the Docker image:

```bash
docker build -t country-aid-api .
```

Run the container:

```bash
docker run --rm -p 5001:5001 country-aid-api
```

The API will be available locally at:

```text
http://127.0.0.1:5001
```

### Example Prediction Request

```bash
curl -X POST http://127.0.0.1:5001/predict \
-H "Content-Type: application/json" \
-d '{
  "child_mort": 90,
  "exports": 20,
  "health": 7,
  "imports": 30,
  "inflation": 12,
  "life_expec": 55,
  "total_fer": 5,
  "gdpp": 1200
}'
```

Example response:

```json
{
  "cluster": 1,
  "vulnerability": "High Vulnerability"
}
```

## Limitations

- The dataset represents a limited snapshot of country-level socioeconomic and health indicators and may not reflect current conditions.
- K-Means identifies similarity patterns but does not determine causality or guarantee that every country within a cluster has the same level of need.
- The model does not include important real-world factors such as conflict, natural disasters, population size, political stability, existing aid, or delivery feasibility.
- Cluster labels such as `High Vulnerability` are interpretation layers added after modelling; the underlying K-Means cluster numbers themselves have no inherent meaning.
- The API should therefore be treated as a decision-support tool rather than an automated aid-allocation system.

