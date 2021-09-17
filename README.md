# Project 4 - West Nile Virus Prediction (Kaggle Competition)

This project was my final group project (part of a [Kaggle competition](https://www.kaggle.com/c/predict-west-nile-virus/overview)) for the [Data Science Immersive course at General Assembly](https://generalassemb.ly/education/data-science-immersive/singapore).

## Executive Summary

The West Nile Virus is most commonly spread to humans through infected mosquitos. Around 20% of people who become infected with the virus develop symptoms ranging from a persistent fever, to serious neurological illnesses that can result in death.
West Nile Virus-related hospitalizations and follow-ups in the United States costed [$778 million](https://www.medicinenet.com/script/main/art.asp?articlekey=176668) in health care expenses and lost productivity from 1999 through 2012.

In 2002, the first human cases of the West Nile virus were reported in Chicago. By 2004 the City of Chicago and the Chicago Department of Public Health (CDPH) had established a comprehensive surveillance and control program, which is still in effect today. Since the implementation of comprehensive surveillances and control programmes, occurrences of the West Nile Virus has been depleting. 

As part of a Kaggle competition, our team aims to build a classifier model to predict the presence of the West Nile Virus in Chicago. The following models were tested and compared: Logistic Regression, Random Forest, Gradient Boosting, Decision Tree, Extra Trees, AdaBoost, and SVC. Evaluation was performed primarily based on the AUC ROC, recall and precision scores. Finally, we looked into potential interventions derived from our model and performed a cost-benefit analysis for a proposal to The Chicago Department of Public Health (CDPH). 

## Problem Statement

Recognising that the West Nile Virus could develop into an endemic, we aim to improve the cost-effectiveness of existing strategies to control adult mosquito populations and mitigate the spread of the virus.

Capitalising on data on past weather conditions and locations where West Nile Virus were found, we look to develop a machine learning model to predict the presence of the West Nile Virus at a particular location facing specific weather conditions. This prediction tool will be useful as it allows for targeted spraying of specific neighbourhoods facing higher threats of the West Nile Virus. We hope to aid Chicago in achieving cost-savings through efficient resource-management towards preventing the transmission of the West Nile Virus.

## Data Dictionary

The datasets obtained from [Kaggle](https://www.kaggle.com/c/predict-west-nile-virus/data) are are as follows. 

|Data|Description|
|:---:|:---|
|[`train.csv`](/assets/train.csv)|There are 10,505 raw observations. Each observation contains location details on the traps, the number of mosquitoes collected and an indication of whether the West Nile Virus is present. The data is for traps tested in years 2007, 2009, 2011 and 2013.| 
|[`test.csv`](/assets/test.csv)|There are 116,293 raw observations. Each observation contains location details on the traps, the number of mosquitoes collected and an indication of whether the West Nile Virus is present. The data is for traps tested in years 2008, 2010, 2012, and 2014.| 
|[`weather.csv`](/assets/weather.csv)|Weather data collected from 2 weather stations between 1 May and 31 Oct from 2007 to 2014. For further descriptions, see [pdf](/assets/noaa_weather_qclcd_documentation.pdf).|
|[`spray.csv`](/assets/spray.csv)|Data contains the date, time and location of spray in 2011 and 2013. This dataset will only be used for the benefit and costs analysis of pesticide use (see Book 5) and will not be used to build the prediction model.|

The data dictionary of the newly engineered features is found below. 

|Feature|Type|Description|
|:---|:---:|:---|
|Tmin_L0|*float*|Average minimum temperature of the weather station closest to the trap within the same month.| 
|Tmin_L1|*float*|Average minimum temperature of the weather station closest to the trap in the previous month.| 
|Tmax_L0|*float*|Average maximum temperature of the weather station closest to the trap within the same month.| 
|Tmax_L1|*float*|Average maximum temperature of the weather station closest to the trap in the previous month.| 
|SeaLevel_L0|*float*|Average atmospheric pressure of the weather station closest to the trap within the same month.| 
|SeaLevel_L1|*float*|Average atmospheric pressure of the weather station closest to the trap in the previous month.| 
|RelativeHumidity|*float*|Relative humidity computed based on dewpoint and average temperature using the formula given at [link](https://www.calcunation.com/calculator/humidity-calculator.php).| 
|RelativeHumidity_L0|*float*|Average relative humidity of the weather station closest to the trap within the same month.| 
|RelativeHumidity_L1|*float*|Average relative humidity of the weather station closest to the trap in the previous month.| 
|PrecipTotal_L0|*float*|Average total precipitation reading of the weather station closest to the trap within the same month.| 
|PrecipTotal_L1|*float*|Average total precipitation reading of the weather station closest to the trap in the previous month.| 
|DewPoint_L0|*float*|Average dewpoint temperature reading of the weather station closest to the trap within the same month.| 
|DewPoint_L1|*float*|Average dewpoint temperature reading of the weather station closest to the trap in the previous month.| 
|AvgSpeed_L0|*float*|Average wind speed reading of the weather station closest to the trap within the same month.| 
|AvgSpeed_L1|*float*|Average wind speed reading of the weather station closest to the trap in the previous month.| 
|PIPIENS|*integer*|Dummy variable = 1 if Pipiens species is found within the trap and = 0 if otherwise.| 
|RESTUANS|*integer*|Dummy variable = 1 if Restauns species is found within the trap and = 0 if otherwise.| 
|WnvRisk_very low|*integer*|Dummy variable = 1 if between 0 and 2 cases of WNV were detected in the past and = 0 if otherwise.| 
|WnvRisk_low|*integer*|Dummy variable = 1 if between 2 and 6 cases of WNV were detected in the past and = 0 if otherwise.| 
|WnvRisk_medium|*integer*|Dummy variable = 1 if between 6 and 10 cases of WNV were detected in the past and = 0 if otherwise.| 
|WnvRisk_high|*integer*|Dummy variable = 1 if more than 10 cases of WNV were detected in the past and = 0 if otherwise.| 
|Rain|*integer*|Dummy variable = 1 if there was an event of either torrential storm, rain, drizzle or shower and = 0 if otherwise.| 
|Mist|*integer*|Dummy variable = 1 if there was an event of either fog, mist, haze or smoke and = 0 if otherwise.| 

## Analysis & Results

7 models were evaluated in the modelling stage. The models considered were Gradient Boosting, Adaboost, Support Vector Machine, Logistic Regression, Decision Trees, Random Forest Classifier & Extra Trees Classifier. Prior to model training, SMOTE was applied on the training dataset to fix the severe class imbalance. Next, a 2-stage pipeline was built for standard scaling on the train data before model training.

During the model training process, cross validation was performed on the train data set, optimizing on the ROC AUC score. The models were then evaluated based on the cross-validated ROC AUC score, precision score and recall score. Although the Gradient Boosting model had the strongest cross-validated ROC AUC score, its recall score (0.527) pales in comparison to that of Adaboost (0.606). This means that we are likely to have fewer False Negatives using Adaboost. Support Vector Classification is also a possible consideration, but it fared worse in terms of the cross-validated ROC AUC score and precision score compared to Adaboost. 

AdaBoost, with a test AUC of 0.837 and recall score of 0.606, seems like the best model as it is important to ensure a relatively high recall score that does not compromise the ROC AUC and/or precision score.

## Conclusions & Recommendations

With the model and the cost-benefit analysis, we concluded that we should be able to achieve significant cost savings However, the WNV prediction rate could be better. More data points would be helpful.

The cost analysis was over-simplified and not performed on a macro level. For example, the cost calculation on spraying did not account for logistics / manhour costs. Some traps were within a 700ft radius between one another; the spraying of traps overlaps and the actual required insecticide is less. In addition, the cost calculation for hospitalization fees was also generalized. To perform a proper cost analysis, we would need to look at the most recent data for an accurate assessment of the current status. Only then can a fair comparison be made.

In addition, understanding the CDPH's basis of operations will also help in bridging the current situation towards their desired KPIs. They mentioned about other efforts including larviciding catch basins, which involves dropping tablets in storm drains along the public way that slowly dissolve over a five-month period to prevent mosquito larvae from hatching, and eliminating standing water by ensuring that swimming pools and construction sites are regularly maintained. These additional efforts beyond spraying and trapping could be explored, and data can be collected on those variables to build a more robust model.

## Team

Jeffrey Prasetio | https://www.linkedin.com/in/jeffreyprasetio/
Raymond Onn | https://www.linkedin.com/in/raymondonn/
Samuel Ng | https://www.linkedin.com/in/samuelngme/
Shi Min Lee | https://www.linkedin.com/in/shi-minlee/

## References

- https://www.cdc.gov/westnile/statsmaps/cumMapsData.html
- https://www.cdc.gov/westnile/statsmaps/cumMapsData.html
- https://www.calcunation.com/calculator/humidity-calculator.php
- https://www.chicago.gov/city/en/depts/cdph/provdrs/healthy_communities/news/2020/august/city-to-spray-insecticide-thursday-to-kill-mosquitoes0.html
- https://www.chicago.gov/content/dam/city/depts/cdph/Mosquito-Borne-Diseases/Zenivex.pdf
- https://www.callnorthwest.com/2019/05/how-long-does-a-mosquito-treatment-last/
- https://datasmart.ash.harvard.edu/news/article/predictive-analytics-guides-west-nile-virus-control-efforts-in-chicago-1152
- https://www.cdc.gov/mosquitoes/about/life-cycles/culex.html
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3322011/
- https://www.cityofchicago.org/dam/city/depts/cdph/comm_dis/general/Communicable_Disease/CD_CDInfo_Jun07_WNV.pdf
