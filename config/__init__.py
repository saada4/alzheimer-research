'''Has all the global variables and imports for the project

This module contains all the global variables and imports that are used throughout the project. This includes the following:
- Imports: All the libraries that are used in the project
- Global Variables: All the global variables that are used in the project
- Constants: All the constants that are used in the project

Imports:
    -  dask: Used to parallelize the data processing in the project
    -  pandas: Used to read and process the data in the project
    -  numpy: Used for numerical computations in the project
    -  sklearn: Used for building the machine learning model in the project
    -  pickle: Used for saving and loading the data and model in the project
    -  matplotlib: Used for plotting the data in the project
    -  plotly: Used for interactive plotting in the project
    -  graphviz: Used for plotting the decision tree in the project

Global Variables:
    -  dataset_pickle: The path to the pickle file containing the dataset
    -  dataset_pop_pickle: The path to the pickle file containing the population data
    -  model_data_pickle: The path to the pickle file containing the model data
    -  model_pickle: The path to the pickle file containing the model

Constants:
    -  topics_to_use: The list of topics to use in the analysis
    -  regions_to_drop: The list of regions to drop from the analysis
    -  cols_to_drop: The list of columns to drop from the analysis
    -  population_years: The list of population years to use in the analysis
        - Used to weigh years of the population data
    -  pop_sex_to_stratified: The mapping of numerical sex to its textual representation
    -  pop_race_to_stratified: The mapping of numerical race to its textual representation
    -  pop_stratified_to_race: The mapping of textual representation of race to its numerical representation
    -  dementia_statistics: The statistics for dementia prevalence
        - Used in generating the synthetic data
    -  rng: The seeded random number generator
    -  choices: The choices for the random number generator (binary as of now, can be extended to multi-class)
'''

# data
import dask.dataframe as dd
from dask import array as da
from dask import delayed
import pandas as pd
import numpy as np
# model
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TunedThresholdClassifierCV
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, \
    confusion_matrix, ConfusionMatrixDisplay, classification_report
# misc
import pickle
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import graphviz

dataset_pickle = "storage/data_20240725.pickle"
dataset_pop_pickle = "storage/population_data.pickle"
model_data_pickle = "storage/model_data.pickle"
model_pickle = "storage/model.pickle"

# constants
reduction_factor = 2 ** 10 # How much to shrink the population data by
alpha = 2 ** 3 # Laplace Smoothing for the stimulated population (accounts for regions with little data)
alpha_dementia = .3 # 30% smoothing on the population which has dementia

topics_to_use = [
    "Frequent mental distress",
    "Prevalence of sufficient sleep",
    "Eating 2 or more fruits daily",
    "Eating 3 or more vegetables daily",
    "Lifetime diagnosis of depression",
    "Obesity",
    "Fall with injury within last year",
    "Oral health:  tooth retention"]

cols_to_drop  = [
    "YearEnd", 
    "Class", 
    "Question", 
    "Data_Value_Type", 
    "Data_Value_Footnote_Symbol", 
    "Data_Value_Footnote", 
    "Geolocation", 
    "ClassID", 
    "TopicID", 
    "QuestionID", 
    "StratificationCategoryID2"]

regions_to_drop = [
    "United States, DC & Territories",
    "West",
    "South",
    "Midwest",
    "Northeast",
    "Puerto Rico",
    "Virgin Islands",
    "Guam"]

population_years = [
    "POPESTIMATE2015", 
    "POPESTIMATE2016",
    "POPESTIMATE2017", 
    "POPESTIMATE2018", 
    "POPESTIMATE2019", 
    "POPESTIMATE2020", 
    "POPESTIMATE2021", 
    "POPESTIMATE2022"]

pop_sex_to_stratified = {
    0: "Overall",
    1: "MALE",
    2: "FEMALE"
}

pop_race_to_stratified = {
    -1: "HIS",
    1: "WHT",
    2: "BLK",
    3: "NAA",
    4: "ASN",
    5: "NAA"
}

pop_stratified_to_race = {v: k for k, v in pop_race_to_stratified.items()}

'''
Dementia statistics from these papers

https://jamanetwork.com/journals/jamaneurology/fullarticle/2781919
Hendriks S, Peetoom K, Bakker C, et al. Global Prevalence of Young-Onset Dementia: A Systematic Review and Meta-analysis. 
    JAMA Neurol. 2021;78(9):1080–1090. doi:10.1001/jamaneurol.2021.2161
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9013315/
Rajan KB, Weuve J, Barnes LL, McAninch EA, Wilson RS, Evans DA. 
    Population estimate of people with clinical Alzheimer's disease and mild cognitive impairment in the United States (2020-2060). 
    Alzheimers Dement. 2021 Dec;17(12):1966-1975. doi: 10.1002/alz.12362. Epub 2021 May 27. PMID: 34043283; PMCID: PMC9013315.
'''
# add 30% padding to account for unbalanced class data
dementia_statistics = {
    "Overall" : (1 - .1 / 100_000 - alpha_dementia, .1 / 100_000 + alpha_dementia),
    "50-64 years": (1 - np.mean((1.6, 6.7, 23.3)) / 100_000 - alpha_dementia, np.mean((1.6, 6.7, 23.3)) / 100_000 + alpha_dementia),
    "65 years or older": (1 - 11.3 / 100 - alpha_dementia, 11.3 / 100 + alpha_dementia)
}

non_patient_probabilities = {
    "Frequent mental distress": (1 - 15.9 / 100, 15.9 / 100),
        # source: https://www.americashealthrankings.org/explore/measures/mental_distress
    "Prevalence of sufficient sleep": (1 - 72.3 / 100, 72.3 / 100),
        # source: https://www.cdc.gov/nchs/fastats/sleep-health.htm
    "Eating 2 or more fruits daily": (1 - 12.3 / 100, 12.3 / 100),
        # source: https://www.cdc.gov/mmwr/volumes/71/wr/mm7101a1.htm
    "Eating 3 or more vegetables daily": (.9, .1),
        # source: https://www.cdc.gov/mmwr/volumes/71/wr/mm7101a1.htm
    "Lifetime diagnosis of depression": (1 - 18.4 / 100, 18.4 / 100),
        # source: https://www.cdc.gov/mmwr/volumes/72/wr/mm7224a1.htm
    "Obesity": (1 - 41.9 / 100, 41.9 / 100),
        # https://www.cdc.gov/obesity/php/data-research/adult-obesity-facts.html
    "Fall with injury within last year": (1 - 27.6 / 100, 27.6 / 100),
        # https://www.cdc.gov/mmwr/volumes/72/wr/mm7235a1.htm
    "Oral health:  tooth retention": (1 - 27.6  / 100, 27.6  / 100)
        # https://www.nidcr.nih.gov/research/data-statistics/tooth-loss/adults
}

# High quality initial entropy
entropy = 0x3f8524baec58cd3338c2ba5503ebade2
base_bitgen = np.random.PCG64(entropy)
generators = base_bitgen.spawn(12)
current_generator = generators[0]

rng = np.random.default_rng(current_generator)

choices = (0, 1)