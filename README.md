# Machines Remaining Useful Life Prediction

## Objective
The objective of this project is to develop a robust and accurate predictive model for estimating the Remaining Useful Life (RUL) of machines. By leveraging advanced machine learning techniques, this project aims to enhance predictive maintenance by providing timely and accurate predictions of machinery failure, thereby reducing downtime and maintenance costs. Additionally, it seeks to improve operational efficiency by optimizing the operational lifespan of machinery, ensuring continuous and efficient operation. Another key goal is to increase safety by minimizing the risk of unexpected machinery failures, thus enhancing the safety of operations and reducing the likelihood of accidents. This project is designed to be a comprehensive solution for industries seeking to implement predictive maintenance strategies and improve the reliability and efficiency of their machinery.

## Data Description
This dataset was available as a part of Azure AI Notebooks for Predictive Maintenance. But as of 15th Oct, 2020 the notebook (link) is no longer available. However, the data can still be downloaded using the following links:

https://azuremlsampleexperiments.blob.core.windows.net/datasets/PdM_telemetry.csv
https://azuremlsampleexperiments.blob.core.windows.net/datasets/PdM_errors.csv
https://azuremlsampleexperiments.blob.core.windows.net/datasets/PdM_maint.csv
https://azuremlsampleexperiments.blob.core.windows.net/datasets/PdM_failures.csv
https://azuremlsampleexperiments.blob.core.windows.net/datasets/PdM_machines.csv

The data consists of the following datasets:



- Telemetry Time Series Data (PdM_telemetry.csv): It consists of hourly average of voltage, rotation, pressure, vibration collected from 100 machines for the year 2015.

- Error (PdM_errors.csv): These are errors encountered by the machines while in operating condition. Since, these errors don't shut down the machines, these are not considered as failures. The error date and times are rounded to the closest hour since the telemetry data is collected at an hourly rate.

- Maintenance (PdM_maint.csv): If a component of a machine is replaced, that is captured as a record in this table. Components are replaced under two situations: 1. During the regular scheduled visit, the technician - replaced it (Proactive Maintenance) 2. A component breaks down and then the technician does an unscheduled maintenance to replace the component (Reactive Maintenance). This is considered as a failure and corresponding data is captured under Failures. Maintenance data has both 2014 and 2015 records. This data is rounded to the closest hour since the telemetry data is collected at an hourly rate.

- Failures (PdM_failures.csv): Each record represents replacement of a component due to failure. This data is a subset of Maintenance data. This data is rounded to the closest hour since the telemetry data is collected at an hourly rate.

- Machines Characteristics (PdM_Machines.csv): Model type & age of the Machines.




## Methodology

Data collected from different sensors and machines characteristics will be analyzed and then we will perform feature engineering and data preprocessing. for the modeling, The project will utilize a hybrid model combining an LSTM network with an MLP network to achieve high prediction accuracy.


## Machine Learning 

### Machine Learning Models

Models used in this project are the following :

    - Dummy Regressor (baseline model)
    - Fine Tuned MobileNetV2 Model

### Models Evaluation & Results

The following are the accuracy scores for each model on the validation dataset:

    - Simple CNN model: 0.92
    - Fine Tuned MobileNetV2 model: 0.98

<p align = "center"> 
  <img src = "https://github.com/Mahdi-Kriaa/Casting_Defect_Detection/blob/main/images/confusion_matrix.png">
</p>

This is the confusion matrix for the fine Tuned MobileNetV2 model on the validation dataset
## Recommendations
- The fine tuned MobileNetV2 model can detect the majority of defective pieces but it still not perfect, so it is not reliable to trust only on it.
- Images must represent the frontal piece view and must be taken in a good brigthness level.

## Limitations & Next Steps

The model is not reliable for detecting all the casting defects so other image preprocessing techniques or models can be tried to improve the model performance.


 
