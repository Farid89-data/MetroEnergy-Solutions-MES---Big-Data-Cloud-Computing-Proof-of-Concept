# MetroEnergy Solutions (MES) - Big Data & Cloud Computing Proof of Concept

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Apache Spark](https://img.shields.io/badge/Apache-Spark-orange.svg)
![PySpark](https://img.shields.io/badge/PySpark-MLlib-red.svg)
![License](https://img.shields.io/badge/License-Academic-green.svg)

## Overview

This repository contains the Proof of Concept (PoC) developed for the **COM7020 – Big Data and Cloud Computing** module at **Arden University**.

The project demonstrates the design and implementation of a cloud-enabled big data pipeline for a fictional energy provider, **MetroEnergy Solutions (MES)**. It showcases how Apache Spark can be used to process large-scale smart meter data, perform data quality management, aggregate energy consumption, and build a machine learning model for peak demand forecasting.

The solution follows the architecture proposed in the accompanying technical report and represents the **processing** and **forecasting** layers of a modern cloud-based energy analytics platform.

---

# Project Objectives

The project aims to demonstrate:

- Generation of realistic synthetic smart meter data
- Batch ingestion of large datasets
- Distributed data processing using Apache Spark
- Data quality validation and fault detection
- Energy consumption aggregation
- Peak electricity demand forecasting using Spark MLlib
- Export of analytical datasets and visualisations

---

# Architecture

The implementation follows the architecture proposed in the report:

```
Data Sources
      │
      ▼
Batch Ingestion
      │
      ▼
Cloud Storage (Data Lake)
      │
      ▼
Apache Spark Processing
      │
      ▼
Machine Learning (Spark MLlib)
      │
      ▼
Reports & Dashboards
```

The PoC focuses primarily on the **Batch Processing** and **Machine Learning** components.

---

# Features

## Synthetic Smart Meter Dataset

The project generates a realistic synthetic dataset representing:

- Smart electricity meters
- Solar generation
- EV charging
- Weather conditions
- Voltage monitoring

The generated data models:

- Daily consumption cycles
- Weekly usage patterns
- Seasonal temperature changes
- Household variation
- Solar production
- EV charging behaviour
- Sensor faults
- Random measurement noise

Dataset characteristics:

- 5 regions
- 40 smart meters per region
- Hourly readings
- 365 days of data
- Approximately **1.75 million records**

---

## Data Processing

Apache Spark is used to:

- Load large CSV datasets
- Apply schema validation
- Clean invalid sensor readings
- Separate faulty data into a quarantine dataset
- Generate analytical features
- Aggregate daily energy consumption
- Create hourly regional load profiles

---

## Machine Learning

Spark MLlib Linear Regression is used to predict:

- Daily peak grid demand

Model features include:

- Temperature
- Day of week
- Month
- Weekend indicator
- Total daily energy consumption

Performance metrics generated:

- RMSE
- R² Score
- Feature coefficients
- Model intercept

---

## Visualisations

The project automatically generates:

- Daily energy consumption trend
- Hourly regional load profile
- Predicted vs Actual peak demand

---

# Technologies Used

- Python
- Apache Spark
- PySpark
- Spark SQL
- Spark MLlib
- Pandas
- Matplotlib
- Google Cloud Storage (GCS)

---

# Project Structure

```
.
├── mes_poc.py
├── outputs/
│   ├── daily_region_summary.csv
│   ├── hourly_load_profile.csv
│   ├── peak_demand_predictions.csv
│   ├── peak_demand_per_day.csv
│   ├── model_metrics.txt
│   ├── chart_daily_consumption_trend.png
│   ├── chart_hourly_load_profile.png
│   └── chart_prediction_vs_actual.png
├── README.md
└── MES_Big_Data_Cloud_Report.pdf
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/mes-bigdata-poc.git

cd mes-bigdata-poc
```

Install dependencies:

```bash
pip install pyspark pandas matplotlib
```

---

# Running the Project

Execute:

```bash
python mes_poc.py
```

Apache Spark will:

- Load the dataset
- Clean faulty readings
- Aggregate consumption data
- Train the forecasting model
- Export results
- Generate charts

---

# Outputs

The script generates:

| Output | Description |
|---------|-------------|
| raw_smart_meter_readings.csv | Synthetic smart meter dataset |
| daily_region_summary.csv | Daily regional energy summary |
| hourly_load_profile.csv | Hourly consumption profile |
| peak_demand_per_day.csv | Daily peak demand |
| peak_demand_predictions.csv | ML predictions |
| model_metrics.txt | RMSE, R² and model coefficients |
| chart_daily_consumption_trend.png | Daily consumption chart |
| chart_hourly_load_profile.png | Regional hourly profile |
| chart_prediction_vs_actual.png | Forecast evaluation |

---

# Academic Context

This project was developed as part of the **COM7020 – Big Data and Cloud Computing** module.

The accompanying technical report evaluates:

- Big Data challenges
- Cloud deployment models
- Hybrid cloud architecture
- Batch vs Stream Processing
- Data governance
- GDPR compliance
- Energy-sector regulations
- Big data architecture design
- Apache Spark implementation
- Future recommendations

---

# Limitations

This Proof of Concept uses:

- Synthetic data rather than real utility data
- Local Spark execution instead of a distributed cluster
- Linear Regression as a baseline forecasting model

Future work may include:

- Spark Structured Streaming
- Kafka integration
- Real cloud deployment
- Predictive maintenance
- Advanced forecasting models (XGBoost, LSTM, Prophet)
- Real-time dashboards

---

# Author

**Farid Negahbani**

Student Number: **24154844**

MSc Data Science

Arden University

---

# Academic Supervisor

**Dr. Ahmed Hassan**

Module Tutor

COM7020 – Big Data and Cloud Computing

Arden University

---

# Acknowledgements

This project was completed as part of the MSc Data Science programme at Arden University. It demonstrates the practical application of Big Data, Cloud Computing, Apache Spark, and Machine Learning concepts within the context of modern smart energy systems.

---

# License

This repository is provided for **academic and educational purposes only**.

© 2026 Farid Negahbani | Arden University