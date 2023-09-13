import sys
assert sys.version_info >= (3, 7), "Requires Python >= 3.7"
from packaging import version
import sklearn
assert version.parse(sklearn.__version__) >= version.parse("1.0.1")

from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
# from scipy.stats import binom
# from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.cluster import KMeans

def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

housing = load_housing_data()

""" Printing values from housing

print(housing.head())
print(housing.describe().to_string())
housing.hist(bins=50, figsize=(12,8))
plt.show()
"""

""" Shuffle and split data is a method that uses numpy to shuffle and split 
the housing data array. This is done by using numpy random function, more 
specifically permutation which changes the indices of the array (positions).
"""

def shuffle_and_split_data(data,test_ratio):
    shuffled_instances=np.random.permutation(len(data)) # Permutes the indices
    test_set_size = int(len(data)*test_ratio) # The size of the test set
    test_indices = shuffled_instances[:test_set_size] # The indices of the test set
    train_indices = shuffled_instances[test_set_size:] # The indices of the train set
    return data.iloc[train_indices], data.iloc[test_indices] 

""" Test cases
train_set, test_set = shuffle_and_split_data(housing, 0.2)
len(train_set), len(test_set)
"""

""" 
is_id_in_test_set is a function that performs a ciclycal redundacy check 
by summing the ids of the passed array. It only returns True if the ids are 
less than 2^32 times the passed test_ratio.
"""

def is_id_in_test_set(identifier, test_ratio):
    return crc32(np.int64(identifier)) < test_ratio * 2**32

def split_data_with_id_hash(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

""" 
I used this by the books example but in the end another more efective and 
preexistent implementation is suggested (I'll leave the functions untouched 
in the code so i can consult them if i need to get a glimpse of what the 
code is doing).

housing_with_id = housing.reset_index() # Adds an 'index' column
train_set, test_set = split_data_with_id_hash(housing, 0.2, "longitude")


housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "id")
"""

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
test_set["total_bedrooms"].isnull().sum()

# print(test_set["total_bedrooms"].isnull().sum())

""" 
# Extra code - shows how to compute the 10.7% probability of getting 
# a bad sample:

sample_size = 1000
ratio_female = 0.511
proba_too_small = binom(sample_size, ratio_female).cdf(485 - 1)
proba_too_large = 1 - binom(sample_size, ratio_female).cdf(535)
print(proba_too_small + proba_too_large)

"""

housing["income_cat"]= pd.cut(housing["median_income"],
bins=[0,1.5,3,4.5,6,np.inf],labels=[1,2,3,4,5]) 
# Segments and sorts the data into bins

""" 
Generating a bar plot from the sorted array

housing["income_cat"].value_counts().sort_index().plot.bar(rot=0,grid=True) 

plt.xlabel("Income Category")
plt.ylabel("Number of districts")
plt.show()

"""

""" Stratified testing technique by the books example. Here is where i define
the pertinent strata for the current test set. This is done by segmentating 
the data by income"""

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
strat_splits = []
for train_index, test_index in splitter.split(housing, housing["income_cat"]):
    strat_train_set_n = housing.iloc[train_index]
    strat_test_set_n = housing.iloc[test_index]
    strat_splits.append([strat_train_set_n, strat_test_set_n])

strat_train_set_n, strat_test_set_n = strat_splits[0]
strat_train_set_n, strat_test_set_n = train_test_split(housing, test_size=0.2, 
stratify=housing["income_cat"] ,random_state=42)

# print(strat_test_set_n["income_cat"].value_counts()/len(strat_test_set_n))

""" Since i wont be using the income_cat column again, 
i'll delete it from the current dataframe """

for set_ in (strat_train_set_n, strat_test_set_n):
    set_.drop("income_cat", axis=1, inplace=True)

""" Now i need to explore and visualize the current data. Therefore im going
to define a exploration set since the data is too big. Im going to do this by
just copying the dataframe. """

housing = strat_train_set_n.copy()

""" Plotting the data:
    Im going to generate the final example of the scatter plot from the 
    book's example. This takes into account latitude and longitude as well 
    as median house values in a predefined jet color map. 

    housing.plot(kind="scatter", x="longitude", y="latitude", grid=True,
            s=housing["population"] / 100, label="population",
            c="median_house_value", colorbar=True,
            legend=True, sharex=False, figsize=(10, 7))
plt.show()
"""

""" 
For some weird ass reason ocean proximity column interferes to find pearsons
correlation. I dropped said column. Error shown was of type conversion. 
"""

housing.drop("ocean_proximity", axis=1, inplace=True)

""" 
corr_matrix = housing.corr()    # Pearson's correlation.
corr_matrix["median_house_value"].sort_values(ascending=False) 
# Calculates how much each parameter correlates to median house value.
"""

""" 
Scatter plot matrix: This method plots every attribute in the passed 
array against every other numerical attribute.

attributes = ["median_house_value", "median_income", "total_rooms",
            "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))

plt.show()
"""

""" 
Income vs house value plot:

housing.plot(kind="scatter", x="median_income", y="median_house_value",
            alpha=0.1, grid=True)
plt.show() 
"""

""" 
Setting up new parameters (which are all attribute combinations) to see 
if they correlate with median house value.
"""

housing["rooms_per_house"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_ratio"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["people_per_house"] = housing["population"]/housing["households"]

# corr_matrix = housing.corr()
# print(corr_matrix["median_house_value"].sort_values(ascending=False))

""" Reverting to a clean training set: """

housing = strat_train_set_n.drop("median_house_value", axis=1)
housing_labels = strat_train_set_n["median_house_value"].copy()

""" Cleaning up the data: """

""" 

#Alternative 1

housing.dropna(subset=["total_bedrooms"],inplace=True)

#Alternative 2

housing.drop("total_bedrooms", axis=1)

#Alternative 3. This is the chosen one since is the least destructive 
for the dataset

"""

""" 
Here i replaced the missing values with the median so it does not bias the data.
Any other value than the median will affect the result by driving the fit towards
the extremes. this is achieved by the function SimpleImputer. This technique is
called imputation.

Then the Imputer calculated the mean and stored into an attribute called 
".statistics_".

Finally the data gets fitted and then transformed by the .transform() method.
"""


median = housing["total_bedrooms"].median() #3

housing["total_bedrooms"].fillna(median, inplace=True)

imputer = SimpleImputer(strategy="median")

housing_num = housing.select_dtypes(include=[np.number])

imputer.fit(housing_num)

imputer.statistics_
housing_num.median().values

X = imputer.transform(housing_num)

""" 
Finally the numpy array is converted back into a pandas dataframe by wrapping
X into a DataFrame using pandas and recovering the column names and index from
housing_num.

"""

""" Transformer:
    At last i get some sense of what the actual heck a transformer is. I mean 
    You hear about transformers in ML everywhere but i never really understood
    what they are. Here i get a pretty clear picture that a transformer's actual
    purpose is to transform the data (so far the only type of transformation 
    i've seen is of datatype from a matrix to a vector).
"""
housing_tr = pd.DataFrame(X, columns=housing_num.columns, 
                        index=housing_num.index)

housing_cat = housing[["ocean_proximity"]]
# print(housing_cat.head(8))

""" Data encoding:
    Here i encoded the data by assigning an ordinal number to each category.
    it is done by the OrdinalEncoder class preexistent in sklearn.
    Another option is to use the OneHotEncoder class. This is done by assinging
    a binary number to each category.
"""

ordinal_encoder = OrdinalEncoder()
housing_cat_encoded =  ordinal_encoder.fit_transform(housing_cat)

# print(housing_cat_encoded[:8]) # Prints some registries

# print(ordinal_encoder.categories_) # Prints all the categories

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

# print(housing_cat_1hot.__class__) # Prints the one hot type = matrix

housing_cat_1hot.toarray() # Converts the one hot matrix into array

""" Note:
    Just as clarification the usage of sparse matrices is justified in terms
    of efficiency in space. More specifically, lets say we have a sparse matrix
    it only stores the non-zero values and the location of the non-zero values.
    Therefore it only uses the instructions to fill the matrix and avoids the
    unnecesary usage of resources.
"""

# print(cat_encoder.categories_) # Prints all the categories

""" Typing this but not using it since it is not the best choice for the model
    and also, since the model is in production this will generate erroneous 
    results.
"""

df_test = pd.DataFrame({"ocean_proximity": ["INLAND", "NEAR BAY"]})
pd.get_dummies(df_test)

df_test_unknown = pd.DataFrame({"ocean_proximity": ["<2H OCEAN", "ISLAND"]})
pd.get_dummies(df_test_unknown)

cat_encoder.handle_unknown = "ignore"
cat_encoder.transform(df_test_unknown)

cat_encoder.feature_names_in_
cat_encoder.get_feature_names_out()
df_output=pd.DataFrame(cat_encoder.transform(df_test_unknown))

""" Scaling:
    
    Minmax scaling: This is more sensible to fitting bias. I.e if the data contains
    outliers, they will be scaled amongst the limits and could affect the 
    result. This is when you divide by the maximum value and substract the 
    minimum value in order to generate a 0-1 scale for the data.

    Standard scaling: This type of scaling begins by subtracting the mean value 
    (so standardized values have a zero mean), then it divides the result by 
    the standard deviation (so standardized values have a standard 
    deviation equal to 1).
"""

min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)
std_scaler = StandardScaler()
housing_num_std_scaled = std_scaler.fit_transform(housing_num)

""" Plotting:
# Plot the current population distribution bia Hist method

fig, axs = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
housing["population"].hist(ax=axs[0], bins=50)
housing["population"].apply(np.log).hist(ax=axs[1], bins=50)
axs[0].set_xlabel("Population")
axs[1].set_xlabel("Log of population")
axs[0].set_ylabel("Number of districts")
plt.show() 
"""

""" Bucketsizing: 
    This is a way to conglomerate the data into buckets. This 
    buckets are defined by some sort of encoding. What this achieves is to 
    smooth the data in order to prevent overfitting, as well as improve the
    model's learning capabilities.
"""

""" Plotting bucketsized data:

age_simil_35 = rbf_kernel(housing[["housing_median_age"]], [[35]], gamma=0.1)
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
"""

target_scaler=StandardScaler()
scaled_labels=target_scaler.fit_transform(housing_labels.to_frame())

model=LinearRegression()
model.fit(housing[["median_income"]],scaled_labels)
some_new_data=housing[["median_income"]].iloc[:5] # Pretend it is "new" data

scaled_predictions=model.predict(some_new_data)
predictions=target_scaler.inverse_transform(scaled_predictions)

model=TransformedTargetRegressor(regressor=LinearRegression(), 
                                transformer=StandardScaler())

model.fit(housing[["median_income"]],housing_labels)
predictions=model.predict(some_new_data)


""" Building custom transformers:
    
    Here i use a method called FunctionTransformer, it basically takes a
    function and applies it to the data. It is mainly used to build custom
    transformers.

    Later on a StandardScalerClone function is implemented.
    This one is written in the book mostly for illustrative purposes.
    It's usefull because it provides a blueprint class for constructing
    a custom transformer. But most importantly the book emphazises on the fact
    that any custom transformer should have the following methods:
    fit, transform and fit_transform.

    Write about: Cluster similarity transformer.
"""

log_transformer = FunctionTransformer(np.log,inverse_func=np.exp)
log_pop = log_transformer.transform(housing[["population"]])

rbf_transformer = FunctionTransformer(rbf_kernel,kw_args=dict(Y=[[35.]],gamma=0.1))
age_simil_35 = rbf_transformer.transform(housing[["housing_median_age"]])

ratio_transformer = FunctionTransformer(lambda X: X[:,[0]]/X[:,[1]])
ratio_transformer.transform(np.array([[1.,2.],[3.,4.]]))

class StandardScalerClone(BaseEstimator, TransformerMixin):
    def __init__(self, with_mean=True): # no *args or **kwargs!
        self.with_mean = with_mean
    def fit(self, X, y=None): # y is required even though we don't use it
        X = check_array(X) # checks that X is an array with finite float valuesself.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.n_features_in_ = X.shape[1] # every estimator stores this in fit()
        return self # always return self!
    def transform(self, X):
        check_is_fitted(self) # looks for learned attributes (with trailing _)
        X = check_array(X)
        assert self.n_features_in_ == X.shape[1]
        if self.with_mean:
            X = X - self.mean_
        return X / self.scale_


class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state
    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self # always return self!
    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]
    
""" 
    The cluster similarity class is a class that is implemented in the book in 
    order to categorize the data. The Kmeans algorithm fits the samples by 
    measuring a parameter known as "inertia". Said parameter is described 
    mathematically as the sum of squares. The difference between each cluster 
    is determined by the "proximity" of each of the samples variance to the 
    cluster value. The number of clusters is passed as an imput (hyperparameter
    since it's a ml algorithm in this case) on the function.
"""

""" 
    I really liked this explanation from the book so here it goes:
    "
    This code creates a ClusterSimilarity transformer, setting the number of
    clusters to 10. Then it calls fit_transform() with the latitude and 
    longitude of every district in the training set, weighting each district
    by its median house value. The transformer uses k-means to locate the
    clusters, then measures the Gaussian RBF similarity between each district
    and all 10 cluster centers. The result is a matrix with one row per district,
    and one column per cluster."
"""

cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
similarities = cluster_simil.fit_transform(housing[["latitude", "longitude"]],
sample_weight=housing_labels)

# print(similarities[:3].round(2))