from benchopt import BaseObjective, safe_import_context
from IPython import embed

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from sklearn.dummy import DummyClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.metrics import roc_auc_score
    import numpy as np


# The benchmark objective must be named `Objective` and
# inherit from `BaseObjective` for `benchopt` to work properly.
class Objective(BaseObjective):

    # Name to select the objective in the CLI and to display the results.
    name = "Prediction"
    url = "https://github.com/xyhan-github/stats-335_tabular_data"
    parameter_template = "test_size={test_size:.3f}"

    is_convex = False

    requirements = ["scikit-learn"]

    # List of parameters for the objective. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    # This means the OLS objective will have a parameter `self.whiten_y`.
    parameters = {
        'seed': [42],
        'test_size': [0.2],
    }

    # Minimal version of benchopt required to run this benchmark.
    # Bump it up if the benchmark depends on a new feature of benchopt.
    min_benchopt_version = "1.5"

    def set_data(self, X, y, preprocessor=None, prob_type = 'bin', num_classes = 2):
        """Set the data to be used to evaluate the ML algorithms.

        Parameters
        ----------
        X, y : ndarrays, (n_samples, n_features) and (n_samples,)
            The full data and labels, that will be split into train and test.
        preprocessor : sklearn transformer
            A transformer to preprocess the data before fitting the model.
            This part will be passed to the solver as is, and will be used to
            construct a `sklearn.Pipeline`.
        prob_type : str, 'bin', 'mult', or 'reg'
            The type of problem: binary classification, multiclass
            classification, or regression.
        num_classes : int
            The number of classes in the dataset. This is only used for classification.
        """
        rng = np.random.RandomState(self.seed)

        self.prob_type = prob_type

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=rng,
            stratify=y if self.prob_type != 'reg' else None # Stratify if classification; don't if regression
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=self.test_size,
            random_state=rng, stratify=y_train if self.prob_type != 'reg' else None # Stratify if classification; don't if regression
        )
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.X_test, self.y_test = X_test, y_test
        self.preprocessor = preprocessor
        self.num_classes  = num_classes

    def evaluate_result(self, model):
        """Evaluate a fitted model on the test set.

        Parameters
        ----------
        model : sklearn.Classifier
            The fitted model to evaluate. This model should be a pipeline with
            the preprocessor as first step and the fitted classifier
            as second step.

        Returns
        -------
        results : dict
            Dictionary containing the evaluation metrics:
            - `score_train`: accuracy on the train set
            - `score_test`: accuracy on the test set
            - `balanced_accuracy`: balanced accuracy on the test set
            - `roc_auc_score`: ROC AUC score on the test set
            - `value`: 1 - `score_test` (for convergence detection)
        """
        score_train = model.score(self.X_train, self.y_train)
        score_val = model.score(self.X_val, self.y_val)
        score_test = model.score(self.X_test, self.y_test)

        # Only calculate accuracy and ROC scores if classification (not regression)
        if self.prob_type != 'reg':
            bl_acc = balanced_accuracy_score(
                self.y_test, model.predict(self.X_test)
            )

            pred = model.predict_proba(self.X_test)
            if len(np.unique(self.y_test)) > 2:
                roc_score = roc_auc_score(self.y_test, pred, multi_class='ovr')
            else:
                roc_score = roc_auc_score(self.y_test, pred[:, 1])

        # This method can return many metrics in a dictionary. One of these
        # metrics needs to be `value` for convergence detection purposes.
        result =  dict(
            score_test=score_test,
            score_val=score_val,
            score_train=score_train,
            value=1-score_test
        )

        # Only save accuracy and ROC scores if classification (not regression)
        if self.prob_type != 'reg':
            result['balanced_accuracy'] = bl_acc
            result['roc_auc_score'] = roc_score

        return result

    def get_one_result(self):
        # Return one solution. The return value should be an object compatible
        # with `self.compute`. This is mainly for testing purposes.
        dummy_model = make_pipeline(self.preprocessor, DummyClassifier())
        return dict(model=dummy_model.fit(self.X_train, self.y_train))

    def get_objective(self):
        """Get the data to be passed to fit the solver.

        Returns
        -------
        X_train, y_train : ndarrays, (n_samples, n_features) and (n_samples,)
            The training data and labels, that will be used to fit the model.
        preprocessor : sklearn transformer
            A transformer to preprocess the data before fitting the model.
            This part should be used to construct a `sklearn.Pipeline`.
        """
        return dict(
            X_train=self.X_train,
            y_train=self.y_train,
            preprocessor=self.preprocessor,
            prob_type=self.prob_type,
            num_classes=self.num_classes,
        )
