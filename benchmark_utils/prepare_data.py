import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

#@title Define `prepare_data(X_df, y_df)` function
def prepare_data(X_df, y_df):

  """
  Preprocesses and prepares the input data and target labels for machine learning tasks.

  Parameters:
      X_df (pd.DataFrame): DataFrame containing the input features, including categorical columns.
      y_df (pd.DataFrame): DataFrame containing the target labels.

  Returns:
      X (np.ndarray): NumPy array of preprocessed input features after one-hot encoding categorical variables.
      y (np.ndarray): NumPy array of target labels, either as a numeric array (for regression) or encoded as
                      integers (for classification).
      obj_type (str): Indicates the type of the machine learning problem:
                      - 'reg' for regression (numeric target)
                      - 'bin' for binary classification
                      - 'mult' for multiclass classification
      num_classes (int or None): Number of classes for classification tasks. None for regression.

  The function first one-hot encodes categorical columns in X_df, then converts y_df into an appropriate format based on
  its data type:
  - If y_df contains numeric values, it's treated as a regression problem.
  - If y_df contains categorical values (including strings), it's treated as a classification problem. The target
    labels are encoded as integers, and the number of classes is determined. If there are 2 classes, it's binary
    classification; otherwise, it's multiclass classification.

  Example:
      X, y, obj_type, num_classes = prepare_data(X_df, y_df)
  """

  # Remove rows with NaNs
  X_df = X_df.dropna()
  y_df = y_df.loc[X_df.index]

  # Convert categorical columns into 0-1 variables
  object_cols = X_df.select_dtypes(include='object').columns
  X_df_encoded = pd.get_dummies(X_df, columns=object_cols)

  # Create data array
  X = X_df_encoded.values

  # Convert y into target array
  y_array = y_df.iloc[:, 0].to_numpy()

  # Create target vector
  if np.issubdtype(y_array.dtype, np.number):
      y = y_array
      obj_type = 'reg'
      num_classes = None
  else:
      # If y is categorical (including strings), use LabelEncoder for encoding
      encoder = LabelEncoder()
      y = encoder.fit_transform(y_array)
      num_classes = len(encoder.classes_)
      obj_type = 'bin' if num_classes == 2 else 'mult'

  return X, y, obj_type, num_classes