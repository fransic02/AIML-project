import os, sys
import urllib3
import pandas as pd

sys.path.insert(0, '../fair_classification/')  # the code for fair classification is in this directory
import utils as ut
import numpy as np
from random import seed, shuffle

SEED = 42
seed(SEED)  # set the random seed so that the random permutations can be reproduced again
np.random.seed(SEED)


def load_data(load_data_size=None, data_path=None):
    """
        if load_data_size is set to None (or if no argument is provided), then we load and return the whole data
        if it is a number, say 10000, then we will return randomly selected 10K examples
    """

    attrs = ['sex', 'age', 'age_cat', 'race', 'r_decile_score', 'score_text', 'priors_count', 'v_decile_score', 'v_score_text', 'c_charge_degree', 'is_recid']  # all attributes
    int_attrs = ['age', 'r_decile_score', 'priors_count', 'v_decile_score']  # attributes with integer values -- the rest are categorical
    sensitive_attrs = ['race']  # the fairness constraints will be used for this feature
    attrs_to_ignore = ['race', 'is_recid']  # race is sensitive feature so we will not use them in classification, we will not consider two_years_recid for classification since its computed externally and it highly predictive for the class
    attrs_for_classification = set(attrs) - set(attrs_to_ignore)

    dataset = pd.read_csv(data_path)[attrs]
    # remapping the is_recid feature to comply with the fair_classification algorithm
    dataset.loc[dataset.is_recid == 1, 'is_recid'] = -1
    dataset.loc[dataset.is_recid == 0, 'is_recid'] = 1

    X = []
    class_label = dataset.columns[-1]
    y = list(dataset[class_label].values)
    x_control = {}

    attrs_to_vals = {}  # will store the values for each attribute for all users
    for k in attrs:
        if k in sensitive_attrs:
            x_control[k] = []
        elif k in attrs_to_ignore:
            pass
        else:
            attrs_to_vals[k] = []

    for i in range(0, len(dataset.columns.values) - 1):
        attr_name = attrs[i]
        attr_vals = list(dataset[attr_name].values)

        if attr_name in sensitive_attrs:
            x_control[attr_name] = attr_vals
        elif attr_name in attrs_to_ignore:
            pass
        else:
            attrs_to_vals[attr_name] = attr_vals

    def convert_attrs_to_ints(d):  # discretize the string attributes
        for attr_name, attr_vals in d.items():
            if attr_name in int_attrs: continue
            uniq_vals = sorted(list(set(attr_vals)))  # get unique values

            # compute integer codes for the unique values
            val_dict = {}
            for i in range(0, len(uniq_vals)):
                val_dict[uniq_vals[i]] = i

            # replace the values with their integer encoding
            for i in range(0, len(attr_vals)):
                attr_vals[i] = val_dict[attr_vals[i]]
            d[attr_name] = attr_vals

    # convert the discrete values to their integer representations
    convert_attrs_to_ints(x_control)
    convert_attrs_to_ints(attrs_to_vals)

    # if the integer vals are not binary, we need to get one-hot encoding for them
    for attr_name in attrs_for_classification:
        attr_vals = attrs_to_vals[attr_name]
        if attr_name in int_attrs or attr_name in ['sex', 'c_charge_degree']:  # the way we encoded 'sex' and 'c_charge_degree', thay are binary now so no need to apply one hot encoding on it
            X.append(attr_vals)
        else:
            attr_vals, index_dict = ut.get_one_hot_encoding(attr_vals)
            for inner_col in attr_vals.T:
                X.append(inner_col)

    # convert to numpy arrays for easy handline
    X = np.array(X, dtype=float).T
    y = np.array(y, dtype=float)
    for k, v in x_control.items(): x_control[k] = np.array(v, dtype=float)

    # shuffle the data
    perm = list(range(0, len(y)))  # shuffle the data before creating each fold
    shuffle(perm)
    X = X[perm]
    y = y[perm]
    for k in x_control.keys():
        x_control[k] = x_control[k][perm]

    # see if we need to subsample the data
    if load_data_size is not None:
        print('Loading only %d examples from the data' % load_data_size)
        X = X[:load_data_size]
        y = y[:load_data_size]
        for k in x_control.keys():
            x_control[k] = x_control[k][:load_data_size]
    return X, y, x_control