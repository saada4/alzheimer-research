from config import dd, da, delayed, pickle, pd, np, \
    dataset_pickle, dataset_pop_pickle, model_data_pickle, \
    topics_to_use, regions_to_drop, cols_to_drop, population_years, pop_sex_to_stratified, pop_race_to_stratified, non_patient_probabilities,\
        rng, dementia_statistics, choices, alpha, reduction_factor
# imports by lines are: libs, files, constants, and stimulation misc

len_zero_cols = len(topics_to_use)


def load_data(mode: str="dask", data_set: str="brfss") -> dd.DataFrame | pd.DataFrame:
    '''Loads data from a pickle file and returns a dask dataframe
    
    Args:
        mode (str): The mode to load the data in. Can be "dask" or "pandas"
            - defaults to "dask"
        data_set (str): The dataset to load. Can be "brfss" or "census"
            - defaults to "brfss"
    '''

    dataset = dataset_pickle if data_set == "brfss" else dataset_pop_pickle
    match mode:
        case "pandas":
            return pickle.load(open(dataset, "rb"))
        case _:
            return dd.from_pandas(pickle.load(open(dataset, "rb")), npartitions=16)
        
def drop_unwanted(data: dd.DataFrame | pd.DataFrame) -> dd.DataFrame | pd.DataFrame:
    '''Drops unwanted columns and rows from the data [brfss]'''

    # reason its separated is for the indexing
    data = data[data.Topic.isin(topics_to_use)].reset_index(drop=True)
    data = data[~data.LocationDesc.isin(regions_to_drop)]
    return data.dropna(subset=['Data_Value']).reset_index(drop=True)

def get_stratified_data(data: dd.DataFrame) -> list[pd.DataFrame]:
    '''Gets stratified data (based on the brfss data) for the topics to use'''

    stratified_data = data.drop(columns=cols_to_drop) \
        .groupby(["Topic", "LocationDesc", "Stratification1", "StratificationID2", "YearStart"]).agg("mean").reset_index().groupby("Topic")
    return [stratified_data.get_group((topic,)).compute() for topic in topics_to_use]

def generate_stimulated_base(brfss_data: dd.DataFrame, pop_data: dd.DataFrame) -> pd.DataFrame:
    '''Generates stimulated data for the population [census] data'''

    year_weight = brfss_data.YearEnd.value_counts().compute().sort_index() / sum(brfss_data.YearStart.value_counts())

    # add constant amount to avoid zeroing
    def apply_year_weight(pop_data_row) -> int:
        return int(
                (pop_data_row[population_years].values * year_weight).sum()
            ) // reduction_factor + alpha
    
    def stimulate_population_base(pop_data_row: pd.Series) -> np.array:
        height = apply_year_weight(pop_data_row) + 10    
        return np.column_stack((
            # add dementia column
            rng.choice(choices, size=(height, ), p=dementia_statistics[pop_data_row["AGE_GROUP"]]),
            # add stratified data
            np.full((height, 4), pop_data_row[["AGE_GROUP", "NAME", "SEX", "RACE"]]),
            # add feature cols
            np.zeros((height, len_zero_cols)),        
        ))

    data_stimulated = delayed(pd.concat)(
        [
            delayed(pd.DataFrame)(result, columns=["HAS_ALZHEIMERS", "AGE_GROUP", "NAME", "SEX", "RACE", *topics_to_use]) 
            for result in pop_data.compute().apply(stimulate_population_base, axis=1)
        ]
    ).reset_index(drop=True)

    return data_stimulated.compute()

def stimulate_data(data_stimulated: pd.DataFrame, stratified_data: list[pd.DataFrame]=None, mode="both") -> dd.DataFrame:
    '''Stimulates the population data based on the brfss data
    
    Args:
        data_stimulated (pd.DataFrame): The stimulated population data that needs filling
        stratified_data (list[pd.DataFrame]): The stratified data for the topics to use
            - ONLY used in the "ad" mode
        mode (str): The mode to stimulate the data in. Can be "both", "non-ad", or "ad"
            - defaults to "non-ad"
    '''

    if mode == "non-ad" or mode == "both":
        n = len(data_stimulated[data_stimulated["HAS_ALZHEIMERS"] == 0])
        for key, prob in non_patient_probabilities.items():
            data_stimulated.loc[data_stimulated["HAS_ALZHEIMERS"] == 0, key] = rng.choice((0, 1), size=n, p=prob)
        if mode == "non-ad":
            return data_stimulated
    if mode == "ad" or mode == "both":
        # Define the meta information for the output DataFrame
        meta = pd.DataFrame({name: pd.Series(dtype='int')  for name in topics_to_use})
        # Define helper function to stimulate the population features
        def get_mean_and_std(question_data: pd.DataFrame, stimulated_row: pd.Series, sub_frame: pd.DataFrame) -> tuple[float, float]:
            # if not found, go by location and age_group, ignoring race and sex because they are less available
            if question_data.empty:
                question_data = sub_frame[
                    (sub_frame["LocationDesc"] == stimulated_row["NAME"]) &
                    (sub_frame["Stratification1"] == stimulated_row["AGE_GROUP"])
                ]
            try:
                return (
                    question_data["Data_Value"].mean(),
                    max(
                        (question_data["Data_Value"] - question_data["Low_Confidence_Limit"]).mean(),
                        (question_data["High_Confidence_Limit"] - question_data["Data_Value"]).mean()
                    ) / 1.645
                )
            except Exception as e:
                print(question_data, e)
                raise e        
        def stimulate_population_features(stimulated_row: pd.Series) -> np.ndarray:
            conditions = (
                (sub_frame["LocationDesc"] == stimulated_row["NAME"]) & (
                    #(sub_frame["StratificationID2"] == pop_race_to_stratified[stimulated_row["RACE"]]) |
                    (sub_frame["StratificationID2"] == stimulated_row["RACE"]) |
                    #(sub_frame["StratificationID2"] == pop_sex_to_stratified[stimulated_row["SEX"]])
                    (sub_frame["StratificationID2"] == stimulated_row["SEX"])
                ) & 
                (sub_frame["Stratification1"] == stimulated_row["AGE_GROUP"])
                for sub_frame in stratified_data
            )
            # pre-probability info for each question
            question_info = ([sub_frame[condition] for condition, sub_frame in zip(conditions, stratified_data)])
            # col1 are mean, col2 are std
            means_and_std = da.from_array([get_mean_and_std(question, stimulated_row, sub_frame) for question, sub_frame in zip(question_info, stratified_data)])    
            probabilities = rng.normal(means_and_std[:, 0], means_and_std[:, 1]).clip(0.005, 99.995)
            try:
                return [rng.choice(choices, p=(1 - probability / 100, probability / 100)) for probability in probabilities]
            except Exception as e:
                print(e, [rng.choice(choices, p=(1 - probability / 100, probability / 100)) for probability in probabilities], stimulated_row, probabilities)
                return np.zeros((1, 8))

        # Apply the function lazily using Dask
        f = dd.from_pandas(
            data_stimulated[data_stimulated["HAS_ALZHEIMERS"] == 1], 
            npartitions=64
            ).apply(stimulate_population_features, axis=1, meta=meta)
        data_stimulated.loc[data_stimulated["HAS_ALZHEIMERS"] == 1, topics_to_use] = np.vstack(f.compute(scheduler='processes').values)
        return data_stimulated
    
def encode_data(data_stimulated: dd.DataFrame | pd.DataFrame) -> dd.DataFrame | pd.DataFrame:
    '''Encodes the data's categorical columns
    
    Currently modifies "RACE", "SEX", "NAME" (i.e., state of residence), and "AGE_GROUP"
    '''

    def convert_sex_to_text(sex: int) -> str:
        return pop_sex_to_stratified[sex]
    def convert_race_to_text(race: int) -> str:
        return pop_race_to_stratified[race]
    
    try:
        data_stimulated['SEX'] = data_stimulated['SEX'].apply(convert_sex_to_text)
    except Exception as e:
        print("Sex already converted", e)
    try:
        data_stimulated['RACE'] = data_stimulated['RACE'].apply(convert_race_to_text)
    except Exception as e:
        print("Sex already converted", e)

    return pd.get_dummies(
        data_stimulated, 
        columns=['AGE_GROUP', 'NAME', 'SEX', 'RACE'], 
        prefix=['Age_Group', 'Location', 'Sex', 'Race_code']).astype(int)

def save_data(data_stimulated: dd.DataFrame | pd.DataFrame) -> None:
    '''Saves the data to a pickle file'''
    
    pickle.dump(data_stimulated, open(model_data_pickle, "wb"))

def load_model_data() -> pd.DataFrame:
    '''Loads the model data from a pickle file'''
    
    return pickle.load(open(model_data_pickle, "rb"))