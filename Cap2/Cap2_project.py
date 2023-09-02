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

