# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is a Random Forest classifier trained to predict salary category based on categorical features.

**Model Type:** Random Forest Classifier
**Model Version:** 1.0
**Training Date:** August 2025
**Framework:** scikit-learn
**Developer:** Alex Barton  for the purpose of WGU/Udacity coursework

## Intended Use

**Primary Use Cases:**
- Educational purposes and ML pipeline demonstrations
- Research on income prediction models

**Primary Intended Users:**
- Data scientists and ML engineers
- Students learning ML pipeline deployment

**Out-of-Scope Use Cases:**
- Making real-world financial decisions
- Any use that could influence decisions affecting employment, housing, credit or other life impacts. Don't be evil. 
- Commercial applications

## Factors
The following demographic factors were used in the training of this model.
    ""workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",


## Metrics
**Model Performance Measures:**
- **Precision:** 0.7407 - Proportion of positive predictions that were correct
- **Recall:** 0.6320 - Proportion of actual positives correctly identified  
- **F1-Score:** 0.6820 - Harmonic mean balancing precision and recall

## Evaluation Data
**Dataset: census.csv**
20% of the dataset was used for evaluation.
The portion of evaluation data was choses as it is a standard amount used for model training.
Slice-based evaluation ensured fairness by identifying issues across demographic groups. 
One-hot encoding was used for categorical features along with conversion of labels to binary. 


## Training Data
**Dataset: census.csv**
80% of the dataset was used for evaluation.


## Quantitative Analyses
Performance was observed to vary across demographics, education performance was the best, showing clear correlation between education
and model performance (F1 scores).

## Ethical Considerations
Any disparities found between racial and sex groups should be carefully evaluated. The root cause of such disparities is likely 
related to historical and systemic differences that perpetuate inequity that crosses racial, sex and class bounds. 
Do not use data insights derived from racial or sex groups to influcence or harm any person or group such as 
hiring, firing, lending, housing etc. Would you be comforatable if someone made this decision about one of your
loved ones without knowing them? Treat the use of this data as if it effects you personally. Don't be evil. 


## Caveats and Recommendations
This model and any outputs are inteded for education. No financial or practical
decisions of any kind should be made using this model. 