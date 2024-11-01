from config import *

sklearn.set_config(enable_metadata_routing=True)

def randomized_train_test(data_stimulated: dd.DataFrame | pd.DataFrame, test_size=.25) -> tuple:
    to_drop = [x for x in ["HAS_ALZHEIMERS", "Oral health:  tooth retention"] if x in data_stimulated.columns]
    X = data_stimulated.drop(to_drop, axis=1) # , "Oral health:  tooth retention"
    y = data_stimulated["HAS_ALZHEIMERS"]

    return train_test_split(
        X, 
        y, 
        test_size=test_size, 
        random_state=int(rng.random() * 4294967295),)

def train_model(X_train: pd.DataFrame, y_train: pd.Series, scoring="f1", tuned_args: dict={}, model_args: dict={}, tuned_params=None) -> sklearn.base.BaseEstimator:
    '''Trains the model'''

    if not tuned_params:
        inner_model = RandomForestClassifier(
            **{name: param for name, param in {
            "criterion": "log_loss",
            "n_jobs": -1,
            "random_state": int(rng.random() * 4294967295),
            **model_args
            }.items() if name not in model_args}
        )
    else:
        inner_model = RandomForestClassifier(
            **{name: param for name, param in {
            "criterion": "log_loss",
            "n_jobs": -1,
            "random_state": int(rng.random() * 4294967295),
            **model_args
            }.items() if name not in tuned_params}
        )
        inner_model.set_params(**tuned_args)

    inner_model.set_fit_request(sample_weight=True)

    tuned_model = TunedThresholdClassifierCV(
        inner_model,
        scoring=scoring, 
        n_jobs=-1,
        store_cv_results=False        
    )
    
    class_weights = {0: 1, 1: 1.5}
    
    # Normalize the weights so that the total weights for each class are equal
    total_weight_0 = len(y_train[y_train == 0])
    total_weight_1 = len(y_train[y_train == 1])
    
    normalization_factor = total_weight_0 / total_weight_1
    
    sample_weights = y_train.map(class_weights)
    sample_weights[y_train == 1] *= normalization_factor
    sample_weights[y_train == 1] *= 10
    tuned_model.fit(X_train, y_train, sample_weight=sample_weights)
    
    return tuned_model

def find_best_hyperparameters(X_train: pd.DataFrame, y_train: pd.Series, param_distributions: dict, n_iter: int = 100, scoring="f1", cv: int = 5) -> dict:
    '''Finds the best hyperparameters using RandomizedSearchCV'''

    inner_model = RandomForestClassifier(
        # criterion="log_loss",
        random_state=int(rng.random() * 4294967295)
    )

    random_search = RandomizedSearchCV(
        estimator=inner_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=int(rng.random() * 4294967295),
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)
    
    return random_search.best_params_

def load_model() -> sklearn.base.BaseEstimator:
    '''Loads the model from the pickle file'''

    return pickle.load(open(model_pickle, "rb"))

def save_model(model: sklearn.base.BaseEstimator) -> None:
    '''Saves the model to a pickle file'''

    pickle.dump(model, open(model_pickle, "wb"))

class NormalizingFlowModel(pl.LightningModule):
    def __init__(self, input_dim, n_flows=4, lr=1e-3, do_logging=False):
        super(NormalizingFlowModel, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.do_logging = do_logging
        
        self.base = nf.distributions.base.DiagGaussian(input_dim)
        self.flows = []
        for _ in range(n_flows):
            self.flows += [nf.flows.AutoregressiveRationalQuadraticSpline(input_dim, 2, 128)]
            self.flows += [nf.flows.LULinearPermute(input_dim)]
        
        self.model = nf.NormalizingFlow(self.base, self.flows)
        
    def forward(self, x):
        return self.model.log_prob(x)
    
    def training_step(self, batch, _):
        return self.model.forward_kld(batch[0])
    
    def validation_step(self, batch, _):
        if self.do_logging:
            loss = self.model.forward_kld(batch[0])
            self.log('val_loss', loss)
            return loss
        return self.model.forward_kld(batch[0])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
class NormalizingFlowClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model, threshold=0.0):
        '''Initialize the normalizing flow classifier'''

        self.model = model
        self.threshold = threshold

    def fit(self, X, y=None):
        '''No fitting required for the normalizing flow model'''

        return self

    def predict(self, X):
        log_probs = self.model(X)
        return (log_probs > self.threshold).float().numpy()

    def predict_proba(self, X):
        log_probs = self.model(X)
        probs = torch.exp(log_probs)
        return torch.stack([1 - probs, probs], dim=1).numpy()