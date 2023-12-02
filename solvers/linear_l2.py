from benchopt import safe_import_context, BaseSolver
from benchopt.stopping_criterion import SingleRunCriterion

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression, Ridge


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'linear_l2'

    install_cmd = 'conda'
    requirements = ['scikit-learn']

    parameters = {
        'reg_lambda': [1e-1, 1, 10],
        'num_rounds': [1000],
    }

    stopping_criterion = SingleRunCriterion()

    def set_objective(self, X_train, y_train, preprocessor, prob_type, num_classes):
        """Get the data to be passed to fit the solver.

        Parameters
        ----------
        X_train, y_train : ndarrays, (n_samples, n_features) and (n_samples,)
            The training data and labels, that will be used to fit the model.
        preprocessor : sklearn transformer
            A transformer to preprocess the data before fitting the model.
            This part should be used to construct a `sklearn.Pipeline`.
        prob_type : str, 'bin', 'mult', or 'reg'
            The type of problem: binary classification, multiclass
            classification, or regression.
        num_classes : int or None
            The number of classes in the dataset. This is only used for classification.
        """

        self.X_train, self.y_train = X_train, y_train

        if prob_type == 'reg':
            self.model = Ridge(alpha = self.reg_lambda, max_iter = self.num_rounds)
        elif prob_type == 'bin':
            self.model = LogisticRegression(C = 1/self.reg_lambda, max_iter = self.num_rounds)
        elif prob_type == 'mult':
            self.model = LogisticRegression(C = 1/self.reg_lambda, max_iter = self.num_rounds, multi_class = 'multinomial')

    def run(self, n_iter):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of iterations `n_iter`.
        self.model.fit(self.X_train, self.y_train)

    def get_result(self):
        """Return the fitted model to be evaluated."""
        return dict(model=self.model)
