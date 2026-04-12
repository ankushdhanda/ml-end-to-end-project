# %% [markdown]
# This notebook contains :
# 1.) Load data
# 2.) Take quick look (head, info, describe)
# 3.) Visualize distributions

# %%
print(f"lets start with first machine learning project")

# %% [markdown]
# Task: Use California census data to build a model of housing prices in the state.

# %% [markdown]
# Objective: Model's output (a prediction to districts median housing price) will be essential to determine whether it is worth investing in a given area.
# Our models output will be fed to another machine learning system for investment analysis.

# %% [markdown]
# ## Problem Type
# 
# This is a supervised regression problem where the goal is to predict a continuous target variable (median_house_value).

# %%
import sys

assert sys.version_info >= (3, 10)

from packaging.version import Version
import sklearn

assert Version(sklearn.__version__) >= Version("1.6.1")

# %% [markdown]
# Step-1 Get the data

# %% [markdown]
# 
# Welcome to Machine Learning Housing Corp.! 
# Task is to predict median house values in Californian districts, given a number of features from these districts.

# %% [markdown]
# EDA = Understand your data BEFORE modeling so you don’t build garbage models

# %%
from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets", filter="data")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

housing_full = load_housing_data()

# %% [markdown]
# Whyyyyy?  housing_full.head()
# To understand:
# - What features exist
# - What kind of data (numerical, categorical)
# - What the target variable looks like

# %%
housing_full.head()

# %% [markdown]
# ## Initial Data Inspection
# 
# The dataset contains housing-related features such as location, income, population, and median house value (target).
# 
# We observe both numerical and categorical features (e.g., ocean_proximity).

# %% [markdown]
# Why? housing_full.info():
# To identify:
# - Missing values
# - Data types (int, float, object)
# - Number of entries

# %%
housing_full.info()

# %% [markdown]
# ## Data Structure
# 
# - Dataset has X entries and Y features
# - `total_bedrooms` contains missing values → needs handling later
# - `ocean_proximity` is categorical

# %%
housing_full["ocean_proximity"].value_counts()

# %% [markdown]
# To understand:
# - Range of values
# - Mean vs median (skew)
# - Outliers (very high max values)

# %%
housing_full.describe()

# %% [markdown]
# ## Statistical Summary
# 
# - Some features show capped values (e.g., housing_median_age)
# - Income is normalized rather than absolute
# - Presence of potential outliers in population and households

# %% [markdown]
# To detect:
# - Skewed data
# - Normal vs non-normal distribution
# - Feature scaling needs

# %%
import matplotlib.pyplot as plt

# extra code – the next 5 lines define the default font sizes
plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

housing_full.hist(bins=50, figsize=(12, 8))

plt.show()

# %% [markdown]
# ## Feature Distributions
# 
# - Many features are right-skewed (e.g., median_income)
# - Target variable is capped → may affect model predictions
# - Non-normal distributions suggest need for transformation later

# %% [markdown]
# ## Impact on ML Pipeline
# 
# Based on EDA, the following preprocessing steps will be required:
# 
# - Missing Value Handling  
#   `total_bedrooms` contains missing values → will require imputation
# 
# - Feature Scaling  
#   Features like population and income have different ranges → scaling needed
# 
# - Handling Skewness  
#   Right-skewed features may require transformation (e.g., log scaling)
# 
# - Categorical Encoding  
#   `ocean_proximity` is categorical → will require encoding (OneHotEncoding)
# 
# - Target Value Limitation  
#   Median house value is capped → may affect model performance and predictions

# %% [markdown]
# ## Create Test Set

# %%
import numpy as np

def shuffle_and_split_data(data, test_ratio, rng):
    shuffled_indices = rng.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

# %% [markdown]
# To ensure that this notebook's outputs remain the same every time we run it, we need to set the random seed:

# %%
rng = np.random.default_rng(seed=42)
train_set, test_set = shuffle_and_split_data(housing_full, 0.2, rng)
len(train_set)

# %%
len(test_set)

# %% [markdown]
# Sadly, this won't guarantee that this notebook will output exactly the same results as in the book, since there are other possible sources of variation. The most important is the fact that algorithms get tweaked over time when libraries evolve. So please tolerate some minor differences: hopefully, most of the outputs should be the same, or at least in the right ballpark.

# %% [markdown]
# Note: another source of randomness is the order of Python sets: it is based on Python's hash() function, which is randomly "salted" when Python starts up (this started in Python 3.3, to prevent some denial-of-service attacks). To remove this randomness, the solution is to set the PYTHONHASHSEED environment variable to "0" before Python even starts up. Nothing will happen if you do it after that. Luckily, if you're running this notebook on Colab, the variable is already set for you.
# 
# 
# 

# %%
from zlib import crc32

def is_id_in_test_set(identifier, test_ratio):
    return crc32(np.int64(identifier)) < test_ratio * 2**32

def split_data_with_id_hash(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

# %%
housing_with_id = housing_full.reset_index()  # adds an `index` column
train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "index")

# %%
housing_with_id["id"] = (housing_full["longitude"] * 1000
                         + housing_full["latitude"])
train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "id")

# %%
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing_full, test_size=0.2,
                                       random_state=42)

# %%
test_set["total_bedrooms"].isnull().sum()

# %% [markdown]
# To find the probability that a random sample of 1,000 people contains less than 49% female or more than 54% female when the population's female ratio is 51.6%, we use the binomial distribution. The cdf() method of the binomial distribution gives us the probability that the number of females will be equal or less than the given value.

# %%
# extra code – shows how to compute the 10.7% proba of getting a bad sample

from scipy.stats import binom

sample_size = 1000
ratio_female = 0.516
proba_too_small = binom(sample_size, ratio_female).cdf(490 - 1)
proba_too_large = 1 - binom(sample_size, ratio_female).cdf(540)
print(proba_too_small + proba_too_large)

# %%
# If you prefer simulations over maths, here's how you could get roughly the same result:
# extra code – shows another way to estimate the probability of bad sample

rng = np.random.default_rng(seed=42)
samples = (rng.random((100_000, sample_size)) < ratio_female).sum(axis=1)
((samples < 490) | (samples > 540)).mean()

# %% [markdown]
# ## Why Not Simple Random Split?
# 
# A purely random train-test split may not preserve the distribution of important features like median income.
# 
# This can lead to:
# - Biased training data
# - Misleading model evaluation
# 
# To ensure the dataset is representative, we use stratified sampling based on income categories.

# %%
housing_full["income_cat"] = pd.cut(housing_full["median_income"],
                                    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                    labels=[1, 2, 3, 4, 5])

# %% [markdown]
# ## Stratified Sampling Explanation
# 
# - Continuous feature `median_income` is converted into categorical bins using `pd.cut`
# - This allows us to perform stratified sampling
# - `StratifiedShuffleSplit` ensures that train and test sets maintain similar distributions
# - This prevents bias and improves model evaluation reliability

# %%
cat_counts = housing_full["income_cat"].value_counts().sort_index()
cat_counts.plot.bar(rot=0, grid=True)
plt.xlabel("Income category")
plt.ylabel("Number of districts")

plt.show()

# %%
from sklearn.model_selection import StratifiedShuffleSplit

splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
strat_splits = [] # created an array to store the splits, so we can check later that they are all the same
for train_index, test_index in splitter.split(housing_full,
                                              housing_full["income_cat"]):
    strat_train_set_n = housing_full.iloc[train_index]
    strat_test_set_n = housing_full.iloc[test_index]
    strat_splits.append([strat_train_set_n, strat_test_set_n])

# %%
strat_train_set, strat_test_set = strat_splits[0]

# %% [markdown]
# It's much shorter to get a single stratified split:

# %%
strat_train_set, strat_test_set = train_test_split(
    housing_full, test_size=0.2, stratify=housing_full["income_cat"],
    random_state=42)

# %%
strat_test_set["income_cat"].value_counts() / len(strat_test_set)

# %%
# extra code – computes the data for Figure 2–10

def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)

train_set, test_set = train_test_split(housing_full, test_size=0.2,
                                       random_state=42)

compare_props = pd.DataFrame({
    "Overall %": income_cat_proportions(housing_full),
    "Stratified %": income_cat_proportions(strat_test_set),
    "Random %": income_cat_proportions(test_set),
}).sort_index()
compare_props.index.name = "Income Category"
compare_props["Strat. Error %"] = (compare_props["Stratified %"] /
                                   compare_props["Overall %"] - 1)
compare_props["Rand. Error %"] = (compare_props["Random %"] /
                                  compare_props["Overall %"] - 1)
(compare_props * 100).round(2)

# %% [markdown]
# ## Stratified vs Random Sampling Comparison
# 
# Stratified sampling preserves the distribution of income categories much better than random sampling.
# 
# This ensures that both training and test sets are representative of the overall dataset.

# %%
def income_cat_proportions(data):
    return data["income_cat"].value_counts(normalize=True)

compare_props = pd.DataFrame({
    "Overall": income_cat_proportions(housing_full),
    "Stratified": income_cat_proportions(strat_train_set),
    "Random": income_cat_proportions(train_set),
}).sort_index()

compare_props

# %%
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# %% [markdown]
# ## Removing Temporary Feature
# 
# The `income_cat` feature was created only for stratified sampling and is removed to avoid unintended influence on the model.

# %% [markdown]
# ## Avoiding Data Leakage
# 
# After splitting the dataset, all further analysis and preprocessing is performed only on the training set.
# 
# This prevents information from the test set leaking into the model, ensuring unbiased evaluation.

# %%


# %% [markdown]
# ## Discover and Visualize the Data to Gain Insights

# %%
housing = strat_train_set.copy()

# %% [markdown]
# ## Visualizing Geographical Data

# %% [markdown]
# ## Geographical Visualization
# 
# We plot longitude vs latitude to understand spatial distribution of housing prices.
# 
# Observation:
# - High density clusters observed near coastal areas
# - Suggests location strongly influences house prices
# 
# Impact:
# - Location-based features will be important for model performance

# %%
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True)

plt.show()

# %%
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, alpha=0.2)

plt.show()

# %% [markdown]
# ## Geographical Distribution with Target Variable
# 
# This plot shows how house prices vary across locations.
# 
# Observations:
# - Higher prices are concentrated near coastal regions
# - Population density also correlates with price
# 
# Impact:
# - Location and population are strong predictors
# - Feature engineering around geography may improve model performance

# %%
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True,
            s=housing["population"] / 100, label="population",
            c="median_house_value", cmap="jet", colorbar=True,
            legend=True, sharex=False, figsize=(10, 7))

plt.show()

# %%
# extra code – this cell generates the first figure in the chapter

# Download the California image
filename = "california.png"
filepath = Path(f"my_{filename}")
if not filepath.is_file():
    homlp_root = "https://github.com/ageron/handson-mlp/raw/main/"
    url = homlp_root + "images/end_to_end_project/" + filename
    print("Downloading", filename)
    urllib.request.urlretrieve(url, filepath)

housing_renamed = housing.rename(columns={
    "latitude": "Latitude", "longitude": "Longitude",
    "population": "Population",
    "median_house_value": "Median house value (ᴜsᴅ)"})
housing_renamed.plot(
             kind="scatter", x="Longitude", y="Latitude",
             s=housing_renamed["Population"] / 100, label="Population",
             c="Median house value (ᴜsᴅ)", cmap="jet", colorbar=True,
             legend=True, sharex=False, figsize=(10, 7))

california_img = plt.imread(filepath)
axis = -124.55, -113.95, 32.45, 42.05
plt.axis(axis)
plt.imshow(california_img, extent=axis)

plt.show()

# %% [markdown]
# ## Looking for Correlations

# %%
corr_matrix = housing.corr(numeric_only=True)

# %%
corr_matrix["median_house_value"].sort_values(ascending=False)

# %% [markdown]
# ## Correlation Analysis
# 
# - `median_income` has strong positive correlation with house value
# - Some features show weak or negative correlation
# 
# Impact:
# - Income will be a key feature in modeling

# %%
from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))

plt.show()

# %%
housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1, grid=True)

plt.show()

# %%
housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["people_per_house"] = housing["population"] / housing["households"]

# %%
corr_matrix = housing.corr(numeric_only=True)
corr_matrix["median_house_value"].sort_values(ascending=False)

# %%


# %%


# %% [markdown]
# ## Why Not Fit on Test Data?
# 
# Scaling should be learned only from the training data.
# 
# Using `fit_transform` on the test set would compute new scaling parameters,
# leading to inconsistency between training and test data.
# 
# To ensure correct evaluation, we fit the scaler on training data and apply the same transformation to test data.

# %% [markdown]
# ## why not use fit_transform(test_data)? Interview Answer
# 
# Using fit_transform on test data is incorrect because it computes scaling parameters
# like mean and standard deviation from the test set, which leads to inconsistency.
# 
# The model is trained using features scaled with training data statistics,
# so test data must be transformed using the same parameters.
# 
# Therefore, we should fit the scaler on the training data and only transform
# the test data to ensure consistency and correct evaluation.

# %% [markdown]
# 

# %% [markdown]
# ## Prepare the Data for Machine Learning Algorithms

# %% [markdown]
# Let's revert to the original training set and separate the target (note that strat_train_set.drop() creates a copy of strat_train_set without the column, it doesn't actually modify strat_train_set itself, unless you pass inplace=True):

# %%
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"]

# %% [markdown]
# Note: The above cell resets the housing dataframe, also removing the custom features we just added. We are doing this because, later in the chapter, we will build an automated 'transformation pipeline' to handle all these steps at once.

# %% [markdown]
# ## Data Cleaning

# %% [markdown]
# 3 options are listed to handle the NaN values:
# 
# housing.dropna(subset=["total_bedrooms"], inplace=True)    # option 1
# 
# housing.drop("total_bedrooms", axis=1, inplace=True)       # option 2
# 
# median = housing["total_bedrooms"].median()  # option 3
# housing["total_bedrooms"] = housing["total_bedrooms"].fillna(median)
# 
# 
# For each option, we'll create a copy of housing and work on that copy to avoid breaking housing. We'll also show the output of each option, filtering on the rows that originally contained a NaN value.

# %%
null_rows_idx = housing.isnull().any(axis=1)
housing.loc[null_rows_idx].head()

# %%
housing_option1 = housing.copy()

housing_option1.dropna(subset=["total_bedrooms"], inplace=True)  # option 1

housing_option1.loc[null_rows_idx].head()

# %%
housing_option2 = housing.copy()

housing_option2.drop("total_bedrooms", axis=1, inplace=True)  # option 2

housing_option2.loc[null_rows_idx].head()

# %%
housing_option3 = housing.copy()

median = housing["total_bedrooms"].median()
housing_option3["total_bedrooms"] = housing_option3["total_bedrooms"].fillna(median)  # option 3

housing_option3.loc[null_rows_idx].head()

# %%
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")

# %% [markdown]
# Separating out the numerical attributes to use the "median" strategy (as it cannot be calculated on text attributes like ocean_proximity):

# %%
housing_num = housing.select_dtypes(include=[np.number])

# %%
imputer.fit(housing_num)

# %%
imputer.statistics_

# %% [markdown]
# Check that this is the same as manually computing the median of each attribute:

# %%
housing_num.median().values

# %% [markdown]
# Transform the training set

# %%
X = imputer.transform(housing_num)

# %%
imputer.feature_names_in_

# %%
housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing_num.index)

# %%
housing_tr.loc[null_rows_idx].head()

# %%
imputer.strategy

# %%
housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing_num.index)

# %%
housing_tr.loc[null_rows_idx].head()  # not shown in the book

# %% [markdown]
# Now let's drop some outliers:

# %%
from sklearn.ensemble import IsolationForest

isolation_forest = IsolationForest(random_state=42)
outlier_pred = isolation_forest.fit_predict(X)
outlier_pred

# %%
# if you want to remove the outliers, you can use the following code:
#housing = housing.iloc[outlier_pred == 1]
#housing_labels = housing_labels.iloc[outlier_pred == 1]

# %% [markdown]
# ## Handling Text and Categorical Attributes
# 
# Now let's preprocess the categorical input feature, ocean_proximity:

# %%
housing_cat = housing[["ocean_proximity"]]
housing_cat.head(8)

# %%
from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)

# %%
housing_cat_encoded[:8]

# %%
ordinal_encoder.categories_

# %%
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

# %%
housing_cat_1hot

# %% [markdown]
# By default, the OneHotEncoder class returns a sparse array, but we can convert it to a dense array if needed by calling the toarray() method:

# %%
housing_cat_1hot.toarray()

# %% [markdown]
# Alternatively, you can set sparse_output=False when creating the OneHotEncoder (note: the sparse hyperparameter was renamned to sparse_output in Scikit-Learn 1.2):

# %%
cat_encoder = OneHotEncoder(sparse_output=False)
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot

# %%
cat_encoder.categories_

# %%
df_test = pd.DataFrame({"ocean_proximity": ["INLAND", "NEAR BAY"]})
pd.get_dummies(df_test)

# %%
cat_encoder.transform(df_test)

# %%
df_test_unknown = pd.DataFrame({"ocean_proximity": ["<2H OCEAN", "ISLAND"]})
pd.get_dummies(df_test_unknown)

# %%
cat_encoder.handle_unknown = "ignore"
cat_encoder.transform(df_test_unknown)

# %%
cat_encoder.feature_names_in_

# %%
cat_encoder.get_feature_names_out()

# %%
df_output = pd.DataFrame(cat_encoder.transform(df_test_unknown),
                         columns=cat_encoder.get_feature_names_out(),
                         index=df_test_unknown.index)

# %%
df_output

# %% [markdown]
# ## Data Preprocessing Pipeline
# 
# We build a pipeline to automate data preprocessing steps:
# 
# - Missing values are handled using median imputation
# - Numerical features are scaled using standardization
# - Categorical features are encoded using OneHotEncoder
# 
# Using a pipeline ensures:
# - No data leakage
# - Consistent transformations for training and test data
# - Clean and reusable workflow

# %% [markdown]
# ## Feature Scaling

# %%
from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)

# %%
from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()
housing_num_std_scaled = std_scaler.fit_transform(housing_num)

# %%
# extra code – this cell generates plots showing the effect of log transformation on the population attribute
fig, axs = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
housing["population"].hist(ax=axs[0], bins=50)
housing["population"].apply(np.log).hist(ax=axs[1], bins=50)
axs[0].set_xlabel("Population")
axs[1].set_xlabel("Log of population")
axs[0].set_ylabel("Number of districts")

plt.show()

# %% [markdown]
# What if we replace each value with its percentile?

# %%
# extra code – just shows that we get a uniform distribution
percentiles = [np.percentile(housing["median_income"], p)
               for p in range(1, 100)]
flattened_median_income = pd.cut(housing["median_income"],
                                 bins=[-np.inf] + percentiles + [np.inf],
                                 labels=range(1, 100 + 1))
flattened_median_income.hist(bins=50)
plt.xlabel("Median income percentile")
plt.ylabel("Number of districts")
plt.show()
# Note: incomes below the 1st percentile are labeled 1, and incomes above the
# 99th percentile are labeled 100. This is why the distribution below ranges
# from 1 to 100 (not 0 to 100).

# %%
from sklearn.metrics.pairwise import rbf_kernel

age_simil_35 = rbf_kernel(housing[["housing_median_age"]], [[35]], gamma=0.1)

# %%
# extra code – this cell generates Figure 2–18

ages = np.linspace(housing["housing_median_age"].min(),
                   housing["housing_median_age"].max(),
                   500).reshape(-1, 1)
gamma1 = 0.1
gamma2 = 0.03
rbf1 = rbf_kernel(ages, [[35]], gamma=gamma1)
rbf2 = rbf_kernel(ages, [[35]], gamma=gamma2)

fig, ax1 = plt.subplots()

ax1.set_xlabel("Housing median age")
ax1.set_ylabel("Number of districts")
ax1.hist(housing["housing_median_age"], bins=50)

ax2 = ax1.twinx()  # create a twin axis that shares the same x-axis
color = "blue"
ax2.plot(ages, rbf1, color=color, label="gamma = 0.10")
ax2.plot(ages, rbf2, color=color, label="gamma = 0.03", linestyle="--")
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylabel("Age similarity", color=color)

plt.legend(loc="upper left")

plt.show()

# %%
from sklearn.linear_model import LinearRegression

target_scaler = StandardScaler()
scaled_labels = target_scaler.fit_transform(housing_labels.to_frame())

model = LinearRegression()
model.fit(housing[["median_income"]], scaled_labels)
some_new_data = housing[["median_income"]].iloc[:5]  # pretend this is new data

scaled_predictions = model.predict(some_new_data)
predictions = target_scaler.inverse_transform(scaled_predictions)

# %%
predictions

# %%
from sklearn.compose import TransformedTargetRegressor

model = TransformedTargetRegressor(LinearRegression(),
                                   transformer=StandardScaler())
model.fit(housing[["median_income"]], housing_labels)
predictions = model.predict(some_new_data)

# %%
predictions

# %% [markdown]
# ## Custom Transformers

# %% [markdown]
# To create simple transformers:

# %%
from sklearn.preprocessing import FunctionTransformer

log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
log_pop = log_transformer.transform(housing[["population"]])

# %%
rbf_transformer = FunctionTransformer(rbf_kernel,
                                      kw_args=dict(Y=[[35.]], gamma=0.1))
age_simil_35 = rbf_transformer.transform(housing[["housing_median_age"]])

# %%
age_simil_35

# %%
sf_coords = 37.7749, -122.41
sf_transformer = FunctionTransformer(rbf_kernel,
                                     kw_args=dict(Y=[sf_coords], gamma=0.1))
sf_simil = sf_transformer.transform(housing[["latitude", "longitude"]])

# %%
sf_simil

# %%
ratio_transformer = FunctionTransformer(lambda X: X[:, [0]] / X[:, [1]])
ratio_transformer.transform(np.array([[1., 2.], [3., 4.]]))

# %%
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

class StandardScalerClone(BaseEstimator, TransformerMixin):
    def __init__(self, with_mean=True):  # no *args or **kwargs!
        self.with_mean = with_mean

    def fit(self, X, y=None):  # y is required even though we don't use it
        X = check_array(X)  # checks that X is an array with finite float values
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.n_features_in_ = X.shape[1]  # every estimator stores this in fit()
        return self  # always return self!

    def transform(self, X):
        check_is_fitted(self)  # looks for learned attributes (with trailing _)
        X = check_array(X)
        assert self.n_features_in_ == X.shape[1]
        if self.with_mean:
            X = X - self.mean_
        return X / self.scale_

# %%
from sklearn.cluster import KMeans

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self  # always return self!

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]

# %% [markdown]
#  Let's cluster the districts based only on the latitude and longitude:

# %%
cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
similarities = cluster_simil.fit_transform(housing[["latitude", "longitude"]])

# %%
similarities[:3].round(2)

# %%
# extra code – this cell generates Figure 2–19

housing_renamed = housing.rename(columns={
    "latitude": "Latitude", "longitude": "Longitude",
    "population": "Population",
    "median_house_value": "Median house value (ᴜsᴅ)"})
housing_renamed["Max cluster similarity"] = similarities.max(axis=1)

housing_renamed.plot(kind="scatter", x="Longitude", y="Latitude", grid=True,
                     s=housing_renamed["Population"] / 100, label="Population",
                     c="Max cluster similarity",
                     cmap="jet", colorbar=True,
                     legend=True, sharex=False, figsize=(10, 7))
plt.plot(cluster_simil.kmeans_.cluster_centers_[:, 1],
         cluster_simil.kmeans_.cluster_centers_[:, 0],
         linestyle="", color="black", marker="X", markersize=20,
         label="Cluster centers")
plt.legend(loc="upper right")

plt.show()

# %% [markdown]
# ## Why Use Pipelines?
# 
# Using pipelines ensures that all preprocessing steps are applied consistently and in the correct order.
# 
# Benefits:
# - Prevents data leakage by fitting only on training data
# - Ensures same transformations are applied to test data
# - Improves code readability and maintainability
# - Makes the workflow production-ready

# %% [markdown]
# ## Transformation Pipelines

# %% [markdown]
# Now let's build a pipeline to preprocess the numerical attributes:

# %%
from sklearn.pipeline import Pipeline

num_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("standardize", StandardScaler()),
])

# %%
from sklearn.pipeline import make_pipeline

num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

# %%
from sklearn import set_config

set_config(display='diagram')

num_pipeline

# %%
housing_num_prepared = num_pipeline.fit_transform(housing_num)
housing_num_prepared[:2].round(2)

# %%
df_housing_num_prepared = pd.DataFrame(
    housing_num_prepared, columns=num_pipeline.get_feature_names_out(),
    index=housing_num.index)

# %%
df_housing_num_prepared.head(2)  # extra code

# %%
num_pipeline.steps

# %%
num_pipeline[1]

# %%
num_pipeline[:-1]

# %%
num_pipeline.named_steps["simpleimputer"]

# %%
num_pipeline.set_params(simpleimputer__strategy="median")

# %%
from sklearn.compose import ColumnTransformer

num_attribs = ["longitude", "latitude", "housing_median_age", "total_rooms",
               "total_bedrooms", "population", "households", "median_income"]
cat_attribs = ["ocean_proximity"]

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"))

preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
])

# %%
from sklearn.compose import make_column_selector, make_column_transformer

preprocessing = make_column_transformer(
    (num_pipeline, make_column_selector(dtype_include=np.number)),
    (cat_pipeline, make_column_selector(dtype_include=object)),
)

# %%
housing_prepared = preprocessing.fit_transform(housing)

# %%
# extra code – shows that we can get a DataFrame out if we want
housing_prepared_fr = pd.DataFrame(
    housing_prepared,
    columns=preprocessing.get_feature_names_out(),
    index=housing.index)
housing_prepared_fr.head(2)

# %%
def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]  # feature names out

def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler())

log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler())
cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                     StandardScaler())
preprocessing = ColumnTransformer([
        ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
        ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
        ("people_per_house", ratio_pipeline(), ["population", "households"]),
        ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                               "households", "median_income"]),
        ("geo", cluster_simil, ["latitude", "longitude"]),
        ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
    ],
    remainder=default_num_pipeline)  # one column remaining: housing_median_age

# %%
housing_prepared = preprocessing.fit_transform(housing)
housing_prepared.shape

# %%
preprocessing.get_feature_names_out()

# %% [markdown]
# ## Why Use Processed Data Instead of Raw Data?
# 
# Raw data cannot be used directly for training because:
# 
# - It contains missing values
# - Features are on different scales
# - Categorical variables are not numerical
# 
# The preprocessing pipeline transforms raw data into a clean and consistent numerical format (`housing_prepared`) that is suitable for machine learning models.
# 
# This ensures that the model receives standardized input during both training and inference.

# %% [markdown]
# The REAL insight (very important)
# 
# This is the key shift:
# ❌ Model is trained on “data”
# ✅ Model is trained on “processed feature matrix”
# 
# 
# 
# 🧠 Even deeper (system thinking — your strength)
# Think:
# Raw data → unreliable  
# Processed data → standardized input to ML system
# 
# 👉 This is like:
# API contract
# schema in data systems

# %% [markdown]
# 


