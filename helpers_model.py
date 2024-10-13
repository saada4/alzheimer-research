from config import dd, pd, pickle, sklearn, \
    train_test_split, TunedThresholdClassifierCV, RandomForestClassifier, \
    model_pickle, rng

def randomized_train_test(data_stimulated: dd.DataFrame | pd.DataFrame) -> tuple:
    X = data_stimulated.drop(["HAS_ALZHEIMERS", "Oral health:  tooth retention"], axis=1) # , "Oral health:  tooth retention"
    y = data_stimulated["HAS_ALZHEIMERS"]

    return train_test_split(
        X, 
        y, 
        test_size=.1, 
        random_state=int(rng.random() * 4294967295),)

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> sklearn.base.BaseEstimator:
    '''Trains the model'''

    tuned_model = TunedThresholdClassifierCV(
        RandomForestClassifier(criterion="log_loss", n_jobs=-1,random_state=int(rng.random() * 4294967295,)),
        scoring="f1", 
        n_jobs=-1,
        store_cv_results=False
    )
    tuned_model.fit(X_train, y_train)
    
    return tuned_model

def load_model() -> sklearn.base.BaseEstimator:
    '''Loads the model from the pickle file'''

    return pickle.load(open(model_pickle, "rb"))

def save_model(model: sklearn.base.BaseEstimator) -> None:
    '''Saves the model to a pickle file'''

    pickle.dump(model, open(model_pickle, "wb"))