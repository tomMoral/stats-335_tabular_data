from benchopt import BaseDataset, safe_import_context
from benchmark_utils import prepare_data

with safe_import_context() as import_ctx:
    from datasets import load_dataset
    import pandas as pd

class Dataset(BaseDataset):

    name = 'huggingface'

    install_cmd = 'conda'
    requirements = ["pip:datasets","pandas"]

    parameters = {
        "dataset": ["adult_income", "housing"]
    }

    def get_data(self):
        """Get the data to be used to evaluate the ML algorithms.

        Returns
        -------
        X, y : ndarrays, (n_samples, n_features) and (n_samples,)
            The full data and labels, that will be split into train and test
            in the objective.
        """
        # Load the necessary dataframes
        if self.dataset == 'adult_income':
            dataset = load_dataset("scikit-learn/adult-census-income")
            dataset = dataset['train']
            df = pd.DataFrame(dataset)
            y_df = df[['income']]
            X_df = df.drop('income', axis=1)

        elif self.dataset == 'housing':
            dataset = load_dataset("leostelon/california-housing")
            dataset = dataset['train']
            df = pd.DataFrame(dataset)
            y_df = df[['median_house_value']]
            X_df = df.drop('median_house_value', axis=1)
        else:
            raise Exception("Invalid Dataset Name!")

        # Prepare the data
        X, y, prob_type, num_classes = prepare_data(X_df, y_df)

        return dict(
            X=X,
            y=y,
            preprocessor=None,
            prob_type = prob_type,
            num_classes = num_classes,
        )
