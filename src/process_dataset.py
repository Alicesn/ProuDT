import numpy as np
import sklearn
from sklearn.model_selection import (
    train_test_split,
    ParameterGrid,
    ParameterSampler,
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    KFold,
)
from sklearn.tree import DecisionTreeClassifier, plot_tree, DecisionTreeRegressor

# from sklearn.metrics import accuracy_score, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    MinMaxScaler,
    LabelEncoder,
    OrdinalEncoder,
    StandardScaler,
    KBinsDiscretizer,
    RobustScaler,
    QuantileTransformer,
)
from category_encoders import LeaveOneOutEncoder

# from imblearn.over_sampling import SMOTE

from scipy.io.arff import loadarff

from collections import Counter

import os

# from tqdm.notebook import tqdm
from matplotlib import pyplot as plt

from IPython.display import Image
from IPython.display import display, clear_output

import pandas as pd
from ucimlrepo import fetch_ucirepo

import warnings

warnings.filterwarnings("ignore")
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# os.environ["PYTHONWARNINGS"] = "default"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"

import logging

np.seterr(all="ignore")

import time
import random

from itertools import product
from collections.abc import Iterable
import collections

from copy import deepcopy
import timeit


import functools

import scipy

from pathlib import Path
import csv


def flatten_list(l):

    def flatten(l):
        for el in l:
            if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
                yield from flatten(el)
            else:
                yield el

    flat_l = flatten(l)

    return list(flat_l)


def flatten_dict(d, parent_key="", sep="__"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def mergeDict(dict1, dict2):
    # Merge dictionaries and keep values of common keys in list
    newDict = {**dict1, **dict2}
    for key, value in newDict.items():
        if key in dict1 and key in dict2:
            if isinstance(dict1[key], dict) and isinstance(value, dict):
                newDict[key] = mergeDict(dict1[key], value)
            elif isinstance(dict1[key], list) and isinstance(value, list):
                newDict[key] = dict1[key]
                newDict[key].extend(value)
            elif isinstance(dict1[key], list) and not isinstance(value, list):
                newDict[key] = dict1[key]
                newDict[key].extend([value])
            elif not isinstance(dict1[key], list) and isinstance(value, list):
                newDict[key] = [dict1[key]]
                newDict[key].extend(value)
            else:
                newDict[key] = [dict1[key], value]
    return newDict


def normalize_data(
    X_data,
    normalizer_list=None,
    technique="min-max",
    low=-1,
    high=1,
    quantile_noise=1e-3,
    random_state=42,
):
    if normalizer_list is None:
        normalizer_list = []
        if isinstance(X_data, pd.DataFrame):
            if technique != "quantile":

                for column_name in X_data:
                    if technique == "min-max":
                        scaler = MinMaxScaler(feature_range=(low, high))
                    elif technique == "mean":
                        scaler = StandardScaler()

                    scaler.fit(X_data[column_name].values.reshape(-1, 1))

                    X_data[column_name] = scaler.transform(
                        X_data[column_name].values.reshape(-1, 1)
                    ).ravel()
                    normalizer_list.append(scaler)

            else:

                quantile_train = np.copy(X_data.values).astype(np.float64)
                if quantile_noise > 0:
                    np.random.seed(random_state)
                    stds = np.std(quantile_train, axis=0, keepdims=True)
                    noise_std = quantile_noise / np.maximum(stds, quantile_noise)
                    quantile_train += noise_std * np.random.randn(*quantile_train.shape)

                scaler = QuantileTransformer(
                    output_distribution="normal", random_state=random_state
                )
                scaler.fit(quantile_train)
                X_data = pd.DataFrame(
                    scaler.transform(X_data.values),
                    columns=X_data.columns,
                    index=X_data.index,
                )
                normalizer_list.append(scaler)

    else:
        if isinstance(X_data, pd.DataFrame):
            if technique != "quantile":
                for column_name, scaler in zip(X_data, normalizer_list):
                    X_data[column_name] = scaler.transform(
                        X_data[column_name].values.reshape(-1, 1)
                    ).ravel()
            else:
                X_data = pd.DataFrame(
                    normalizer_list[0].transform(X_data.values),
                    columns=X_data.columns,
                    index=X_data.index,
                )

    return X_data, normalizer_list


def split_train_test_valid(
    X_data, y_data, valid_frac=0.20, test_frac=0.20, seed=42, verbosity=0
):
    data_size = X_data.shape[0]
    test_size = int(data_size * test_frac)
    valid_size = int(data_size * valid_frac)

    X_train_with_valid, X_test, y_train_with_valid, y_test = train_test_split(
        X_data, y_data, test_size=test_size, stratify=y_data, random_state=seed
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_with_valid,
        y_train_with_valid,
        test_size=valid_size,
        stratify=y_train_with_valid,
        random_state=seed,
    )

    if verbosity > 0:
        print(X_train.shape, y_train.shape)
        print(X_valid.shape, y_valid.shape)
        print(X_test.shape, y_test.shape)

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def rebalance_data(
    X_train,
    y_train,
    balance_ratio=0.25,
    strategy="SMOTE",  #'SMOTE',
    seed=42,
    verbosity=0,
):  # , strategy='SMOTE'

    min_label = min(Counter(y_train).values())
    sum_label = sum(Counter(y_train).values())

    min_ratio = min_label / sum_label
    if verbosity > 0:
        print("Min Ratio: ", str(min_ratio))
    if min_ratio <= balance_ratio / (len(Counter(y_train).values()) - 1):
        from imblearn.over_sampling import (
            RandomOverSampler,
            SMOTE,
            SMOTEN,
            ADASYN,
            BorderlineSMOTE,
            KMeansSMOTE,
            SVMSMOTE,
            SMOTENC,
        )
        from imblearn.combine import SMOTETomek, SMOTEENN

        try:
            if strategy == "SMOTE":
                oversample = SMOTE()
                print(
                    f"implement SMOTE because-original min_ratio is: {min_ratio}<=criteria ratio is {balance_ratio / (len(Counter(y_train).values()) - 1)}"
                )
            else:
                oversample = RandomOverSampler(
                    sampling_strategy="auto", random_state=seed
                )

            X_train, y_train = oversample.fit_resample(X_train, y_train)
        except ValueError:
            oversample = RandomOverSampler(sampling_strategy="auto", random_state=seed)
            X_train, y_train = oversample.fit_resample(X_train, y_train)

        min_label = min(Counter(y_train).values())
        sum_label = sum(Counter(y_train).values())
        min_ratio = min_label / sum_label
        if verbosity > 0:
            print("Min Ratio: ", str(min_ratio))

    return X_train, y_train


def quantile_normalization(X_train, X_valid, X_test, config):
    X_train, normalizer_list = normalize_data(
        X_train,
        technique=config["preprocessing"]["normalization_technique"],
        quantile_noise=config["preprocessing"]["quantile_noise"],
    )
    X_valid, _ = normalize_data(
        X_valid,
        normalizer_list=normalizer_list,
        technique=config["preprocessing"]["normalization_technique"],
        quantile_noise=config["preprocessing"]["quantile_noise"],
    )
    X_test, _ = normalize_data(
        X_test,
        normalizer_list=normalizer_list,
        technique=config["preprocessing"]["normalization_technique"],
        quantile_noise=config["preprocessing"]["quantile_noise"],
    )
    return X_train, X_valid, X_test, normalizer_list


def load_dataset(name):
    X = pd.read_csv(f"datasets/{name}/X_{name}.csv")
    y = pd.read_csv(
        f"datasets/{name}/y_{name}.csv"
    ).squeeze()  # .squeeze() converts DataFrame to Series
    return X, y


def preprocess_data(
    X_data,
    y_data,
    nominal_features,
    ordinal_features,
    config,
    random_seed=42,
    verbosity=0,
):
    class_counts = y_data.value_counts()
    # print("original distribution", class_counts)

    # print("Shape of original X,y from loading:", X_data.shape, y_data.nunique())

    start_evaluate_network_complete = time.time()

    random.seed(random_seed)
    np.random.seed(random_seed)

    if verbosity > 0:
        print("Original Data Shape (selected): ", X_data.shape)

    # class_count = y_data.value_counts()
    # print(f"before splitting: original data structure is:{class_count}")
    (X_train, y_train, X_valid, y_valid, X_test, y_test) = split_train_test_valid(
        X_data, y_data, seed=random_seed, verbosity=verbosity
    )
    # class_train = y_train.value_counts()
    # class_test = y_test.value_counts()
    # print(f"after splitting: y_train is:{class_count}, y_test is: {class_test}")

    encoder = LeaveOneOutEncoder(cols=ordinal_features)

    X_train = encoder.fit_transform(X_train, y_train)
    X_valid = encoder.transform(X_valid)
    X_test = encoder.transform(X_test)

    excluded_features = flatten_list([nominal_features, ordinal_features])
    X_train, X_valid, X_test, normalizer_list = quantile_normalization(
        X_train, X_valid, X_test, config
    )

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test), normalizer_list


def get_preprocessed_dataset(identifier, random_seed=42, config=None, verbosity=0):
    if identifier == "WisconsinBreastCancer":

        originalData = fetch_ucirepo(id=17)
        X_data = originalData.data.features
        y = originalData.data.targets
        y_data = pd.Series(LabelEncoder().fit_transform(y), name="encoded_labels")

        nominal_features = []
        ordinal_features = []

    elif identifier == "Wine":
        data = originalData = fetch_ucirepo(id=109)
        X_data = originalData.data.features
        y = originalData.data.targets
        y_data = pd.Series(LabelEncoder().fit_transform(y), name="encoded_labels")

        nominal_features = []
        ordinal_features = []
    elif identifier == "ISOLET":
        X_data, y = load_dataset(identifier)
        nominal_features = []
        ordinal_features = []
        y_data = pd.Series(LabelEncoder().fit_transform(y), name="encoded_labels")

    elif identifier == "letter_recognition":
        X_data, y = load_dataset(identifier)
        nominal_features = []
        ordinal_features = []
        y_data = pd.Series(LabelEncoder().fit_transform(y), name="encoded_labels")

    elif identifier == "mnist":
        X_data, y = load_dataset(identifier)
        nominal_features = []
        ordinal_features = []
        y_data = pd.Series(LabelEncoder().fit_transform(y), name="encoded_labels")

    elif identifier == "pen_digits":
        X_data, y = load_dataset(identifier)
        nominal_features = []
        ordinal_features = []
        y_data = pd.Series(LabelEncoder().fit_transform(y), name="encoded_labels")

    elif identifier == "semeion":
        X_data, y = load_dataset(identifier)
        nominal_features = []
        ordinal_features = []
        y_data = pd.Series(LabelEncoder().fit_transform(y), name="encoded_labels")

    elif identifier == "Ecoli":
        data = originalData = fetch_ucirepo(id=39)
        X_data = originalData.data.features
        y = originalData.data.targets
        y_data = pd.Series(LabelEncoder().fit_transform(y), name="encoded_labels")
        nominal_features = []
        ordinal_features = []
    elif identifier == "Yeast":
        data = originalData = fetch_ucirepo(id=110)
        X_data = originalData.data.features
        y = originalData.data.targets
        y_data = pd.Series(LabelEncoder().fit_transform(y), name="encoded_labels")
        nominal_features = []
        ordinal_features = []
        X_data = X_data.reset_index(drop=True)
        y_data = y_data.reset_index(drop=True)

    elif identifier == "Statlog":
        data = originalData = fetch_ucirepo(id=148)
        X_data = originalData.data.features
        y = originalData.data.targets
        y_data = pd.Series(LabelEncoder().fit_transform(y), name="encoded_labels")
        nominal_features = []
        ordinal_features = []
    elif identifier == "MAGIC":
        data = originalData = fetch_ucirepo(id=159)
        X_data = originalData.data.features
        y = originalData.data.targets
        y_data = pd.Series(LabelEncoder().fit_transform(y), name="encoded_labels")
        nominal_features = []
        ordinal_features = []
    elif identifier == "Fertility":
        data = originalData = fetch_ucirepo(id=244)
        X_data = originalData.data.features
        y = originalData.data.targets
        y_data = pd.Series(LabelEncoder().fit_transform(y), name="encoded_labels")
        nominal_features = []
        ordinal_features = []
    elif identifier == "user knowledge":
        data = originalData = fetch_ucirepo(id=257)
        X_data = originalData.data.features
        y = originalData.data.targets
        y_data = pd.Series(LabelEncoder().fit_transform(y), name="encoded_labels")
        nominal_features = []
        ordinal_features = []
    elif identifier == "wholesaleCustomer":
        data = originalData = fetch_ucirepo(id=292)
        X_data = originalData.data.features
        y = originalData.data.targets
        y_data = pd.Series(LabelEncoder().fit_transform(y), name="encoded_labels")
        nominal_features = []
        ordinal_features = []
    elif identifier == "HTRU2":
        data = originalData = fetch_ucirepo(id=372)
        X_data = originalData.data.features
        y = originalData.data.targets
        y_data = pd.Series(LabelEncoder().fit_transform(y), name="encoded_labels")
        nominal_features = []
        ordinal_features = []
    elif identifier == "Phishing":
        data = originalData = fetch_ucirepo(id=379)
        X_data = originalData.data.features
        y = originalData.data.targets
        y_data = pd.Series(LabelEncoder().fit_transform(y), name="encoded_labels")
        nominal_features = []
        ordinal_features = []
    elif identifier == "iranian":
        data = originalData = fetch_ucirepo(id=563)
        X_data = originalData.data.features
        y = originalData.data.targets
        y_data = pd.Series(LabelEncoder().fit_transform(y), name="encoded_labels")
        nominal_features = []
        ordinal_features = []
    elif identifier == "BIN:Cervical Cancer":
        data = pd.read_csv(
            "./datasets/Cervical_Cancer/risk_factors_cervical_cancer.csv",
            index_col=False,
        )  # , names=feature_names

        features_select = [
            "Age",  # numeric
            "Number of sexual partners",  # numeric
            "First sexual intercourse",  # numeric
            "Num of pregnancies",  # numeric
            "Smokes",  # binary
            "Smokes (years)",  # numeric
            "Hormonal Contraceptives",  # binary
            "Hormonal Contraceptives (years)",  # numeric
            "IUD",  # binary
            "IUD (years)",  # numeric
            "STDs",  # binary
            "STDs (number)",  # numeric
            "STDs:condylomatosis",
            "STDs:cervical condylomatosis",
            "STDs:vaginal condylomatosis",
            "STDs:vulvo-perineal condylomatosis",
            "STDs:syphilis",
            "STDs:pelvic inflammatory disease",
            "STDs:genital herpes",
            "STDs:molluscum contagiosum",
            "STDs:AIDS",
            "STDs:HIV",
            "STDs:Hepatitis B",
            "STDs:HPV",
            "STDs: Number of diagnosis",  # numeric
            "STDs: Time since first diagnosis",  # numeric
            "STDs: Time since last diagnosis",  # numeric
            "Dx:Cancer",
            "Dx:CIN",
            "Dx:HPV",
            "Dx",
            "Biopsy",
        ]

        data = data[features_select]

        data["Number of sexual partners"][data["Number of sexual partners"] == "?"] = (
            data["Number of sexual partners"].mode()[0]
        )
        data["First sexual intercourse"][data["First sexual intercourse"] == "?"] = (
            data["First sexual intercourse"].mode()[0]
        )
        data["Num of pregnancies"][data["Num of pregnancies"] == "?"] = data[
            "Num of pregnancies"
        ].mode()[0]
        data["Smokes"][data["Smokes"] == "?"] = data["Smokes"].mode()[0]
        data["Smokes (years)"][data["Smokes (years)"] == "?"] = data[
            "Smokes (years)"
        ].mode()[0]
        data["Hormonal Contraceptives"][data["Hormonal Contraceptives"] == "?"] = data[
            "Hormonal Contraceptives"
        ].mode()[0]
        data["Hormonal Contraceptives (years)"][
            data["Hormonal Contraceptives (years)"] == "?"
        ] = data["Hormonal Contraceptives (years)"].mode()[0]
        data["IUD"][data["IUD"] == "?"] = data["IUD"].mode()[0]
        data["IUD (years)"][data["IUD (years)"] == "?"] = data["IUD (years)"].mode()[0]
        data["STDs"][data["STDs"] == "?"] = data["STDs"].mode()[0]
        data["STDs (number)"][data["STDs (number)"] == "?"] = data[
            "STDs (number)"
        ].mode()[0]
        data["STDs: Time since first diagnosis"][
            data["STDs: Time since first diagnosis"] == "?"
        ] = data["STDs: Time since first diagnosis"][
            data["STDs: Time since first diagnosis"] != "?"
        ].mode()[
            0
        ]
        data["STDs: Time since last diagnosis"][
            data["STDs: Time since last diagnosis"] == "?"
        ] = data["STDs: Time since last diagnosis"][
            data["STDs: Time since last diagnosis"] != "?"
        ].mode()[
            0
        ]

        data["STDs:condylomatosis"][data["STDs:condylomatosis"] == "?"] = data[
            "STDs:condylomatosis"
        ].mode()[0]
        data["STDs:cervical condylomatosis"][
            data["STDs:cervical condylomatosis"] == "?"
        ] = data["STDs:cervical condylomatosis"].mode()[0]
        data["STDs:vaginal condylomatosis"][
            data["STDs:vaginal condylomatosis"] == "?"
        ] = data["STDs:vaginal condylomatosis"].mode()[0]
        data["STDs:vulvo-perineal condylomatosis"][
            data["STDs:vulvo-perineal condylomatosis"] == "?"
        ] = data["STDs:vulvo-perineal condylomatosis"].mode()[0]
        data["STDs:syphilis"][data["STDs:syphilis"] == "?"] = data[
            "STDs:syphilis"
        ].mode()[0]
        data["STDs:pelvic inflammatory disease"][
            data["STDs:pelvic inflammatory disease"] == "?"
        ] = data["STDs:pelvic inflammatory disease"].mode()[0]
        data["STDs:genital herpes"][data["STDs:genital herpes"] == "?"] = data[
            "STDs:genital herpes"
        ].mode()[0]
        data["STDs:molluscum contagiosum"][
            data["STDs:molluscum contagiosum"] == "?"
        ] = data["STDs:molluscum contagiosum"].mode()[0]
        data["STDs:AIDS"][data["STDs:AIDS"] == "?"] = data["STDs:AIDS"].mode()[0]
        data["STDs:HIV"][data["STDs:HIV"] == "?"] = data["STDs:HIV"].mode()[0]
        data["STDs:Hepatitis B"][data["STDs:Hepatitis B"] == "?"] = data[
            "STDs:Hepatitis B"
        ].mode()[0]
        data["STDs:HPV"][data["STDs:HPV"] == "?"] = data["STDs:HPV"].mode()[0]

        data["Dx:Cancer"][data["Dx:Cancer"] == "?"] = data["Dx:Cancer"].mode()[0]
        data["Dx:CIN"][data["Dx:CIN"] == "?"] = data["Dx:CIN"].mode()[0]
        data["Dx:HPV"][data["Dx:HPV"] == "?"] = data["Dx:HPV"].mode()[0]
        data["Dx"][data["Dx"] == "?"] = data["Dx"].mode()[0]

        nominal_features = []

        ordinal_features = []

        X_data = data.drop(["Biopsy"], axis=1)
        y_data = pd.Series(
            OrdinalEncoder()
            .fit_transform(data["Biopsy"].values.reshape(-1, 1))
            .flatten(),
            name="Biopsy",
        )

    elif identifier == "Credit Card":
        data = pd.read_csv(
            "./datasets/UCI_Credit_Card/UCI_Credit_Card.csv", index_col=False
        )
        data = data.drop(["ID"], axis=1)

        features = [
            "LIMIT_BAL",  # numeric
            "SEX",  # binary
            "EDUCATION",  # categorical
            "MARRIAGE",  # binary
            "AGE",  # numeric
            "PAY_0",  # categorical
            "PAY_2",  # categorical
            "PAY_3",  # categorical
            "PAY_4",  # categorical
            "PAY_5",  # categorical
            "PAY_6",  # categorical
            "BILL_AMT1",  # numeric
            "BILL_AMT2",  # numeric
            "BILL_AMT3",  # numeric
            "BILL_AMT4",  # numeric
            "BILL_AMT5",  # numeric
            "BILL_AMT6",  # numeric
            "PAY_AMT1",  # numeric
            "PAY_AMT2",  # numeric
            "PAY_AMT3",  # numeric
            "PAY_AMT4",  # numeric
            "PAY_AMT5",  # numeric
            "PAY_AMT6",
        ]  # numeric

        nominal_features = []

        ordinal_features = []

        X_data = data.drop(["default.payment.next.month"], axis=1)
        y_data = (data["default.payment.next.month"] < 1) * 1

    elif identifier == "BIN:Absenteeism":

        data = pd.read_csv("datasets/Absenteeism/absenteeism.csv", delimiter=";")

        features_select = [
            "Seasons",  # nominal
            "Day of the week",  # nominal
            "Month of absence",  # nominal
            "Disciplinary failure",  # binary
            "Social drinker",  # binary
            "Social smoker",  # binary
            "Transportation expense",  # numeric
            "Distance from Residence to Work",  # numeric
            "Service time",  # numeric
            "Age",  # numeric
            "Work load Average/day ",  # numeric
            "Hit target",  # numeric
            "Education",  # categorical
            "Son",  # numeric
            "Pet",  # numeric
            "Weight",  # numeric
            "Height",  # numeric
            "Body mass index",  # numeric
            "Absenteeism time in hours",
        ]

        data = data[features_select]

        nominal_features = []

        ordinal_features = [
            "Education",
            "Seasons",
            "Month of absence",
        ]

        X_data = data.drop(["Absenteeism time in hours"], axis=1)
        y_data = (
            data["Absenteeism time in hours"] > 4
        ) * 1  # absenteeism_data['Absenteeism time in hours']

    elif identifier == "Adult":
        feature_names = [
            "Age",  # 0 numeric
            "Workclass",  # 1 nominal
            "fnlwgt",  # 2 numeric
            "Education",  # 3 nominal
            "Education-Num",  # 4 nominal
            "Marital Status",  # 5 nominal
            "Occupation",  # 6 nominal
            "Relationship",  # 7 nominal
            "Race",  # 8 nominal
            "Sex",  # 9 binary
            "Capital Gain",  # 10 numeric
            "Capital Loss",  # 11 numeric
            "Hours per week",  # 12 numeric
            "Country",  # 13 nominal
            "capital_gain",  # 14
        ]

        data = pd.read_csv(
            "./datasets/Adult/adult.data",
            names=feature_names,
            index_col=False,
        )

        # adult_data['Workclass'][adult_data['Workclass'] != ' Private'] = 'Other'
        # adult_data['Race'][adult_data['Race'] != ' White'] = 'Other'

        # adult_data.head()

        features_select = [
            "Sex",  # 9
            "Race",  # 8
            "Workclass",  # 1
            "Age",  # 0
            "fnlwgt",  # 2
            "Education",  # 3
            "Education-Num",  # 4
            "Marital Status",  # 5
            "Occupation",  # 6
            "Relationship",  # 7
            "Capital Gain",  # 10
            "Capital Loss",  # 11
            "Hours per week",  # 12
            "Country",  # 13
            "capital_gain",
        ]

        data = data[features_select]

        nominal_features = []
        ordinal_features = [
            "Race",
            "Workclass",
            "Education",
            "Marital Status",
            "Occupation",
            "Relationship",
            "Country",
            "Sex",
        ]

        X_data = data.drop(["capital_gain"], axis=1)
        y_data = (data["capital_gain"] != " <=50K") * 1

    elif identifier == "BIN:Titanic":
        data = pd.read_csv("./datasets/Titanic/train.csv")

        data["Age"].fillna(data["Age"].mean(), inplace=True)
        data["Fare"].fillna(data["Fare"].mean(), inplace=True)

        data["Embarked"].fillna("S", inplace=True)

        features_select = [
            #'Cabin',
            #'Ticket',
            #'Name',
            #'PassengerId'
            "Sex",  # binary
            "Embarked",  # nominal
            "Pclass",  # nominal
            "Age",  # numeric
            "SibSp",  # numeric
            "Parch",  # numeric
            "Fare",  # numeric
            "Survived",
        ]

        data = data[features_select]

        nominal_features = []  # [1, 2, 7]
        ordinal_features = ["Embarked", "Sex", "Pclass"]

        X_data = data.drop(["Survived"], axis=1)
        y_data = data["Survived"]

    elif identifier == "BIN:Loan House":
        data = pd.read_csv("datasets/Loan/loan-train.csv", delimiter=",")

        data["Gender"].fillna(data["Gender"].mode()[0], inplace=True)
        data["Dependents"].fillna(data["Dependents"].mode()[0], inplace=True)
        data["Married"].fillna(data["Married"].mode()[0], inplace=True)
        data["Self_Employed"].fillna(data["Self_Employed"].mode()[0], inplace=True)
        data["LoanAmount"].fillna(data["LoanAmount"].mean(), inplace=True)
        data["Loan_Amount_Term"].fillna(data["Loan_Amount_Term"].mean(), inplace=True)
        data["Credit_History"].fillna(data["Credit_History"].mean(), inplace=True)

        features_select = [
            #'Loan_ID',
            "Gender",  # binary
            "Married",  # binary
            "Dependents",  # numeric
            "Education",  # #binary
            "Self_Employed",  # binary
            "ApplicantIncome",  # numeric
            "CoapplicantIncome",  # numeric
            "LoanAmount",  # numeric
            "Loan_Amount_Term",  # numeric
            "Credit_History",  # binary
            "Property_Area",  # nominal
            "Loan_Status",
        ]

        data = data[features_select]

        # loan_data['Dependents'][loan_data['Dependents'] == '3+'] = 4
        # loan_data['Dependents'] = loan_data['Dependents'].astype(int)

        # loan_data['Property_Area'][loan_data['Property_Area'] == 'Rural'] = 0
        # loan_data['Property_Area'][loan_data['Property_Area'] == 'Semiurban'] = 1
        # loan_data['Property_Area'][loan_data['Property_Area'] == 'Urban'] = 2
        # loan_data['Property_Area'] = loan_data['Property_Area'].astype(int)

        nominal_features = []

        ordinal_features = [
            "Dependents",
            "Property_Area",
            "Education",
            "Gender",
            "Married",
            "Self_Employed",
        ]

        X_data = data.drop(["Loan_Status"], axis=1)
        y_data = (data["Loan_Status"] == "Y") * 1

    elif identifier == "Bank Marketing":

        data = pd.read_csv(
            "datasets/Bank Marketing/bank-full.csv", delimiter=";"
        )  # bank

        features_select = [
            "age",  # numeric
            "job",  # nominal
            "marital",  # nominal
            "education",  # nominal
            "default",  # nominal
            "housing",  # nominal
            "loan",  # nominal
            "contact",  # binary
            "month",  # nominal
            "day",  # nominal
            #'duration', #numeric
            "campaign",  # nominal
            "pdays",  # numeric
            "previous",  # numeric
            "poutcome",  # nominal
            "y",
        ]

        data = data[features_select]

        nominal_features = []
        ordinal_features = [
            "job",  # nominal
            "marital",  # nominal
            "education",  # nominal
            "default",  # nominal
            "housing",  # nominal
            "loan",  # nominal
            "contact",  # binary
            "month",  # nominal
            "day",  # nominal
            "campaign",  # nominal
            "pdays",  # numeric
            "poutcome",  # nominal
        ]

        X_data = data.drop(["y"], axis=1)
        y_data = pd.Series(
            OrdinalEncoder().fit_transform(data["y"].values.reshape(-1, 1)).flatten(),
            name="y",
        )

    elif identifier == "BIN:Wisconsin Diagnostic Breast Cancer":

        feature_names = [
            "ID number",
            "Diagnosis",
            "radius",  # (mean of distances from center to points on the perimeter)
            "texture",  # (standard deviation of gray-scale values)
            "perimeter",
            "area",
            "smoothness",  # (local variation in radius lengths)
            "compactness",  # (perimeter^2 / area - 1.0)
            "concavity",  # (severity of concave portions of the contour)
            "concave points",  # (number of concave portions of the contour)
            "symmetry",
            "fractal dimension",  # ("coastline approximation" - 1)
        ]
        # Wisconsin Diagnostic Breast Cancer
        data = pd.read_csv(
            "./datasets/Wisconsin Diagnostic Breast Cancer/wdbc.data",
            names=feature_names,
            index_col=False,
        )

        features_select = [
            #'ID number',
            "Diagnosis",
            "radius",  # numeric
            "texture",  # numeric
            "perimeter",  # numeric
            "area",  # numeric
            "smoothness",  # numeric
            "compactness",  # numeric
            "concavity",  # numeric
            "concave points",  # numeric
            "symmetry",  # numeric
            "fractal dimension",  # numeric
        ]

        data = data[features_select]

        nominal_features = []
        ordinal_features = []

        X_data = data.drop(["Diagnosis"], axis=1)
        y_data = pd.Series(
            OrdinalEncoder()
            .fit_transform(data["Diagnosis"].values.reshape(-1, 1))
            .flatten(),
            name="Diagnosis",
        )

    elif identifier == "BIN:Heart Disease":
        feature_names = [
            "age",  # numeric
            "sex",  # binary
            "cp",  # nominal
            "trestbps",  # numeric
            "chol",  # numeric
            "fbs",  # binary
            "restecg",  # nominal
            "thalach",  # numeric
            "exang",  # binary
            "oldpeak",  # numeric
            "slope",  # nominal
            "ca",  # numeric
            "thal",  # nominal
            "num",  #
        ]

        data = pd.read_csv(
            "./datasets/Heart Disease/processed.cleveland.data",
            names=feature_names,
            index_col=False,
        )  # , delimiter=' '

        data["age"][data["age"] == "?"] = data["age"].mode()[0]
        data["sex"][data["sex"] == "?"] = data["sex"].mode()[0]
        data["cp"][data["cp"] == "?"] = data["cp"].mode()[0]
        data["trestbps"][data["trestbps"] == "?"] = data["trestbps"].mode()[0]
        data["chol"][data["chol"] == "?"] = data["chol"].mode()[0]
        data["fbs"][data["fbs"] == "?"] = data["fbs"].mode()[0]
        data["restecg"][data["restecg"] == "?"] = data["restecg"].mode()[0]
        data["thalach"][data["thalach"] == "?"] = data["thalach"].mode()[0]
        data["exang"][data["exang"] == "?"] = data["exang"].mode()[0]
        data["oldpeak"][data["oldpeak"] == "?"] = data["oldpeak"].mode()[0]
        data["slope"][data["slope"] == "?"] = data["slope"].mode()[0]
        data["ca"][data["ca"] == "?"] = data["ca"].mode()[0]
        data["thal"][data["thal"] == "?"] = data["thal"].mode()[0]

        nominal_features = []

        ordinal_features = [
            "restecg",
            "slope",  # nominal
            "thal",  # nominal
            "cp",
        ]

        X_data = data.drop(["num"], axis=1)
        y_data = (data["num"] < 1) * 1

    elif identifier == "BIN:Heart Failure":

        data = pd.read_csv(
            "datasets/Heart Failure/heart_failure_clinical_records_dataset.csv",
            delimiter=",",
        )

        features = [
            "age",  # continuous #### age of the patient (years)
            "anaemia",  # binary #### decrease of red blood cells or hemoglobin (boolean)
            "high blood pressure",  # binary #### if the patient has hypertension (boolean)
            "creatinine phosphokinase (CPK)",  # continuous #### level of the CPK enzyme in the blood (mcg/L)
            "diabetes",  # binary #### if the patient has diabetes (boolean)
            "ejection fraction",  # continuous #### percentage of blood leaving the heart at each contraction (percentage)
            "platelets",  # continuous #### platelets in the blood (kiloplatelets/mL)
            "sex",  # binary ####  woman or man (binary)
            "serum creatinine",  # continuous #### level of serum creatinine in the blood (mg/dL)
            "serum sodium",  # continuous #### level of serum sodium in the blood (mEq/L)
            "smoking",  # binary #### if the patient smokes or not (boolean)
            "time",  # continuous #### follow-up period (days)
            "target",  # death event: if the patient deceased during the follow-up period (boolean)
        ]

        nominal_features = []
        ordinal_features = []

        X_data = data.drop(["DEATH_EVENT"], axis=1)
        y_data = (data["DEATH_EVENT"] > 0) * 1

    elif identifier == "Mushroom":
        feature_names = [
            "eadible",  #
            "cap-shape",  #: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
            "cap-surface",  #: fibrous=f,grooves=g,scaly=y,smooth=s
            "cap-color",  #: brown=n,buff=b,cinnamon=c,gray=g,green=r, pink=p,purple=u,red=e,white=w,yellow=y
            "ruises?",  #: bruises=t,no=f
            "odor",  #: almond=a,anise=l,creosote=c,fishy=y,foul=f, musty=m,none=n,pungent=p,spicy=s
            "gill-attachment",  #: attached=a,descending=d,free=f,notched=n
            "gill-spacing",  #: close=c,crowded=w,distant=d
            "gill-size",  #: broad=b,narrow=n
            "gill-color",  #: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e, white=w,yellow=y
            "stalk-shape",  #: enlarging=e,tapering=t
            "stalk-root",  #: bulbous=b,club=c,cup=u,equal=e, rhizomorphs=z,rooted=r,missing=?
            "stalk-surface-above-ring",  #: fibrous=f,scaly=y,silky=k,smooth=s
            "stalk-surface-below-ring",  #: fibrous=f,scaly=y,silky=k,smooth=s
            "stalk-color-above-ring",  #: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y
            "stalk-color-below-ring",  #: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y
            "veil-type",  #: partial=p,universal=u
            "veil-color",  #: brown=n,orange=o,white=w,yellow=y
            "ring-number",  #: none=n,one=o,two=t
            "ring-type",  #: cobwebby=c,evanescent=e,flaring=f,large=l, none=n,pendant=p,sheathing=s,zone=z
            "spore-print-color",  #: black=k,brown=n,buff=b,chocolate=h,green=r, orange=o,purple=u,white=w,yellow=y
            "population",  #: abundant=a,clustered=c,numerous=n, scattered=s,several=v,solitary=y
            "habitat",  #: grasses=g,leaves=l,meadows=m,paths=p, urban=u,waste=w,woods=d
        ]

        data = pd.read_csv(
            "./datasets/Mushroom/agaricus-lepiota.data",
            names=feature_names,
            index_col=False,
        )

        features_select = [
            "eadible",  #
            "cap-shape",  #: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
            "cap-surface",  #: fibrous=f,grooves=g,scaly=y,smooth=s
            "cap-color",  #: brown=n,buff=b,cinnamon=c,gray=g,green=r, pink=p,purple=u,red=e,white=w,yellow=y
            "ruises?",  #: bruises=t,no=f
            "odor",  #: almond=a,anise=l,creosote=c,fishy=y,foul=f, musty=m,none=n,pungent=p,spicy=s
            "gill-attachment",  #: attached=a,descending=d,free=f,notched=n
            "gill-spacing",  #: close=c,crowded=w,distant=d
            "gill-size",  #: broad=b,narrow=n
            "gill-color",  #: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e, white=w,yellow=y
            "stalk-shape",  #: enlarging=e,tapering=t
            "stalk-root",  #: bulbous=b,club=c,cup=u,equal=e, rhizomorphs=z,rooted=r,missing=?
            "stalk-surface-above-ring",  #: fibrous=f,scaly=y,silky=k,smooth=s
            "stalk-surface-below-ring",  #: fibrous=f,scaly=y,silky=k,smooth=s
            "stalk-color-above-ring",  #: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y
            "stalk-color-below-ring",  #: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y
            "veil-type",  #: partial=p,universal=u
            "veil-color",  #: brown=n,orange=o,white=w,yellow=y
            "ring-number",  #: none=n,one=o,two=t
            "ring-type",  #: cobwebby=c,evanescent=e,flaring=f,large=l, none=n,pendant=p,sheathing=s,zone=z
            "spore-print-color",  #: black=k,brown=n,buff=b,chocolate=h,green=r, orange=o,purple=u,white=w,yellow=y
            "population",  #: abundant=a,clustered=c,numerous=n, scattered=s,several=v,solitary=y
            "habitat",  #: grasses=g,leaves=l,meadows=m,paths=p, urban=u,waste=w,woods=d
        ]

        data = data[features_select]

        nominal_features = []
        ordinal_features = [
            "cap-shape",  #: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
            "cap-surface",  #: fibrous=f,grooves=g,scaly=y,smooth=s
            "cap-color",  #: brown=n,buff=b,cinnamon=c,gray=g,green=r, pink=p,purple=u,red=e,white=w,yellow=y
            "ruises?",  #: bruises=t,no=f
            "odor",  #: almond=a,anise=l,creosote=c,fishy=y,foul=f, musty=m,none=n,pungent=p,spicy=s
            "gill-attachment",  #: attached=a,descending=d,free=f,notched=n
            "gill-spacing",  #: close=c,crowded=w,distant=d
            "gill-size",  #: broad=b,narrow=n
            "gill-color",  #: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e, white=w,yellow=y
            "stalk-shape",  #: enlarging=e,tapering=t
            "stalk-root",  #: bulbous=b,club=c,cup=u,equal=e, rhizomorphs=z,rooted=r,missing=?
            "stalk-surface-above-ring",  #: fibrous=f,scaly=y,silky=k,smooth=s
            "stalk-surface-below-ring",  #: fibrous=f,scaly=y,silky=k,smooth=s
            "stalk-color-above-ring",  #: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y
            "stalk-color-below-ring",  #: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y
            "veil-type",  #: partial=p,universal=u
            "veil-color",  #: brown=n,orange=o,white=w,yellow=y
            "ring-number",  #: none=n,one=o,two=t
            "ring-type",  #: cobwebby=c,evanescent=e,flaring=f,large=l, none=n,pendant=p,sheathing=s,zone=z
            "spore-print-color",  #: black=k,brown=n,buff=b,chocolate=h,green=r, orange=o,purple=u,white=w,yellow=y
            "population",  #: abundant=a,clustered=c,numerous=n, scattered=s,several=v,solitary=y
            "habitat",  #: grasses=g,leaves=l,meadows=m,paths=p, urban=u,waste=w,woods=d
        ]

        X_data = data.drop(["eadible"], axis=1)
        y_data = pd.Series(
            OrdinalEncoder()
            .fit_transform(data["eadible"].values.reshape(-1, 1))
            .flatten(),
            name="number",
        )

    elif identifier == "BIN:Raisins":

        data = pd.read_excel("./datasets/Raisins/Raisin_Dataset.xlsx")

        nominal_features = []
        ordinal_features = []

        X_data = data.drop(["Class"], axis=1)
        y_data = pd.Series(
            OrdinalEncoder()
            .fit_transform(data["Class"].values.reshape(-1, 1))
            .flatten(),
            name="number",
        )

    elif identifier == "Rice":

        data = pd.read_excel("./datasets/Rice/Rice_Cammeo_Osmancik.xlsx")

        nominal_features = []
        ordinal_features = []

        X_data = data.drop(["Class"], axis=1)
        y_data = pd.Series(
            OrdinalEncoder()
            .fit_transform(data["Class"].values.reshape(-1, 1))
            .flatten(),
            name="number",
        )

    elif identifier == "BIN:Horse Colic":
        feature_names = [
            "surgery",
            "Age",
            "Hospital Number",
            "rectal temperature",
            "pulse",
            "respiratory rate",
            "temperature of extremities",
            "peripheral pulse",
            "mucous membranes",
            "capillary refill time",
            "pain",
            "peristalsis",
            "abdominal distension",
            "nasogastric tube",
            "nasogastric reflux",
            "nasogastric reflux PH",
            "rectal examination",
            "abdomen",
            "packed cell volume",
            "total protein",
            "abdominocentesis appearance",
            "abdomcentesis total protein",
            "outcome",
            "surgical lesion?",  # TARGET
            "type of lesion1",
            "type of lesion2",
            "type of lesion3",
            "cp_data",
        ]

        data1 = pd.read_csv(
            "./datasets/Colic/horse-colic.data",
            names=feature_names,
            index_col=False,
            delimiter=" ",
            na_values="?",
        )
        data2 = pd.read_csv(
            "./datasets/Colic/horse-colic.test",
            names=feature_names,
            index_col=False,
            delimiter=" ",
            na_values="?",
        )

        data = pd.concat([data1, data2])
        data = data.fillna(data.mode().iloc[0])
        features_select = [
            "surgery",
            "Age",
            #'Hospital Number',
            "rectal temperature",
            "pulse",
            "respiratory rate",
            "temperature of extremities",
            "peripheral pulse",
            "mucous membranes",
            "capillary refill time",
            "pain",
            "peristalsis",
            "abdominal distension",
            "nasogastric tube",
            "nasogastric reflux",
            "nasogastric reflux PH",
            "rectal examination",
            "abdomen",
            "packed cell volume",
            "total protein",
            "abdominocentesis appearance",
            "abdomcentesis total protein",
            "outcome",
            "surgical lesion?",  # TARGET
            "type of lesion1",
            "type of lesion2",
            "type of lesion3",
            "cp_data",
        ]

        data = data[features_select]

        nominal_features = []
        ordinal_features = [
            "temperature of extremities",
            "peripheral pulse",
            "mucous membranes",
            "capillary refill time",
            "pain",
            "peristalsis",
            "abdominal distension",
            "nasogastric tube",
            "nasogastric reflux",
            "rectal examination",
            "abdomen",
            "abdominocentesis appearance",
            "type of lesion1",
            "type of lesion2",
            "type of lesion3",
        ]

        X_data = data.drop(["surgical lesion?"], axis=1)
        y_data = pd.Series(
            OrdinalEncoder()
            .fit_transform(data["surgical lesion?"].values.reshape(-1, 1))
            .flatten(),
            name="number",
        )
        # Reset the indices of both X_data and y_data to ensure they align
        X_data = X_data.reset_index(drop=True)
        y_data = y_data.reset_index(drop=True)

    elif identifier == "BIN:Echocardiogram":

        feature_names = [
            "survival",  # -- the number of months patient survived (has survived, if patient is still alive). Because all the patients had their heart attacks at different times, it is possible that some patients have survived less than one year but they are still alive. Check the second variable to confirm this. Such patients cannot be used for the prediction task mentioned above.
            "still-alive",  # -- a binary variable. 0=dead at end of survival period, 1 means still alive
            "age-at-heart-attack",  # -- age in years when heart attack occurred
            "pericardial-effusion",  # -- binary. Pericardial effusion is fluid around the heart. 0=no fluid, 1=fluid
            "fractional-shortening",  # -- a measure of contracility around the heart lower numbers are increasingly abnormal
            "epss",  # -- E-point septal separation, another measure of contractility. Larger numbers are increasingly abnormal.
            "lvdd",  # -- left ventricular end-diastolic dimension. This is a measure of the size of the heart at end-diastole. Large hearts tend to be sick hearts.
            "wall-motion-score",  # -- a measure of how the segments of the left ventricle are moving
            "wall-motion-index",  # -- equals wall-motion-score divided by number of segments seen. Usually 12-13 segments are seen in an echocardiogram. Use this variable INSTEAD of the wall motion score.
            "mult",  # -- a derivate var which can be ignored
            "name",  # -- the name of the patient (I have replaced them with "name")
            "group",  # -- meaningless, ignore it
            "alive-at-1",  # -- Boolean-valued. Derived from the first two attributes. 0 means patient was either dead after 1 year or had been followed for less than 1 year. 1 means patient was alive at 1 year.
        ]

        data = pd.read_csv(
            "./datasets/Echocardiogram/echocardiogram.data",
            names=feature_names,
            index_col=False,
            na_values="?",
        )
        data = data.fillna(data.mode().iloc[0])

        features_select = [
            #'survival',# -- the number of months patient survived (has survived, if patient is still alive). Because all the patients had their heart attacks at different times, it is possible that some patients have survived less than one year but they are still alive. Check the second variable to confirm this. Such patients cannot be used for the prediction task mentioned above.
            #'still-alive',# -- a binary variable. 0=dead at end of survival period, 1 means still alive
            "age-at-heart-attack",  # -- age in years when heart attack occurred
            "pericardial-effusion",  # -- binary. Pericardial effusion is fluid around the heart. 0=no fluid, 1=fluid
            "fractional-shortening",  # -- a measure of contracility around the heart lower numbers are increasingly abnormal
            "epss",  # -- E-point septal separation, another measure of contractility. Larger numbers are increasingly abnormal.
            "lvdd",  # -- left ventricular end-diastolic dimension. This is a measure of the size of the heart at end-diastole. Large hearts tend to be sick hearts.
            "wall-motion-score",  # -- a measure of how the segments of the left ventricle are moving
            "wall-motion-index",  # -- equals wall-motion-score divided by number of segments seen. Usually 12-13 segments are seen in an echocardiogram. Use this variable INSTEAD of the wall motion score.
            "mult",  # -- a derivate var which can be ignored
            #'name',# -- the name of the patient (I have replaced them with "name")
            #'group',# -- meaningless, ignore it
            "alive-at-1",  # -- Boolean-valued. Derived from the first two attributes. 0 means patient was either dead after 1 year or had been followed for less than 1 year. 1 means patient was alive at 1 year.
        ]

        data = data[features_select]

        nominal_features = []
        ordinal_features = []

        X_data = data.drop(["alive-at-1"], axis=1)
        y_data = (
            data["alive-at-1"] != 0
        ) * 1  # pd.Series(OrdinalEncoder().fit_transform(data['alive-at-1'].values.reshape(-1, 1)).flatten(), name='number')

    elif identifier == "BIN:Thyroid":
        feature_names = [
            "age",  #:				continuous.
            "sex",  #:				M, F.
            "on thyroxine",  #:			f, t.
            "query on thyroxine",  #:		f, t.
            "on antithyroid medication",  #:	f, t.
            "sick",  #:				f, t.
            "pregnant",  #:			f, t.
            "thyroid surgery",  #:		f, t.
            "I131 treatment",  #:			f, t.
            "query hypothyroid",  #:		f, t.
            "query hyperthyroid",  #:		f, t.
            "lithium",  #:			f, t.
            "goitre",  #:				f, t.
            "tumor",  #:				f, t.
            "hypopituitary",  #:			f, t.
            "psych",  #:				f, t.
            "TSH measured",  #:			f, t.
            "TSH",  #:				continuous.
            "T3 measured",  #:			f, t.
            "T3",  #:				continuous.
            "TT4 measured",  #:			f, t.
            "TT4",  #:				continuous.
            "T4U measured",  #:			f, t.
            "T4U",  #:				continuous.
            "FTI measured",  #:			f, t.
            "FTI",  #:				continuous.
            "TBG measured",  #:			f, t.
            "TBG",  #:				continuous.
            "referral source",  #:		WEST, STMW, SVHC, SVI, SVHD, other.
            "class",
        ]

        data = pd.read_csv(
            "./datasets/Thyroid/thyroid0387.data",
            names=feature_names,
            index_col=False,
            na_values="?",
        )
        data = data.fillna(data.mode().iloc[0])

        features_select = [
            "age",  #:				continuous.
            "sex",  #:				M, F.
            "on thyroxine",  #:			f, t.
            "query on thyroxine",  #:		f, t.
            "on antithyroid medication",  #:	f, t.
            "sick",  #:				f, t.
            "pregnant",  #:			f, t.
            "thyroid surgery",  #:		f, t.
            "I131 treatment",  #:			f, t.
            "query hypothyroid",  #:		f, t.
            "query hyperthyroid",  #:		f, t.
            "lithium",  #:			f, t.
            "goitre",  #:				f, t.
            "tumor",  #:				f, t.
            "hypopituitary",  #:			f, t.
            "psych",  #:				f, t.
            "TSH measured",  #:			f, t.
            "TSH",  #:				continuous.
            "T3 measured",  #:			f, t.
            "T3",  #:				continuous.
            "TT4 measured",  #:			f, t.
            "TT4",  #:				continuous.
            "T4U measured",  #:			f, t.
            "T4U",  #:				continuous.
            "FTI measured",  #:			f, t.
            "FTI",  #:				continuous.
            "TBG measured",  #:			f, t.
            "TBG",  #:				continuous.
            "referral source",  #:		WEST, STMW, SVHC, SVI, SVHD, other.
            "class",
        ]

        data = data[features_select]

        nominal_features = []
        ordinal_features = [
            "sex",  #:				M, F.
            "on thyroxine",  #:			f, t.
            "query on thyroxine",  #:		f, t.
            "on antithyroid medication",  #:	f, t.
            "sick",  #:				f, t.
            "pregnant",  #:			f, t.
            "thyroid surgery",  #:		f, t.
            "I131 treatment",  #:			f, t.
            "query hypothyroid",  #:		f, t.
            "query hyperthyroid",  #:		f, t.
            "lithium",  #:			f, t.
            "goitre",  #:				f, t.
            "tumor",  #:				f, t.
            "hypopituitary",  #:			f, t.
            "psych",  #:				f, t.
            "TSH measured",  #:			f, t.
            "T3 measured",  #:			f, t.
            "TT4 measured",  #:			f, t.
            "T4U measured",  #:			f, t.
            "FTI measured",  #:			f, t.
            "TBG measured",  #:			f, t.
            "referral source",  #:		WEST, STMW, SVHC, SVI, SVHD, other.
        ]

        X_data = data.drop(["class"], axis=1)
        y_data = (
            data["class"].str.contains("-")
        ) * 1  # ((data['class'] != '-') * 1)        #pd.Series(OrdinalEncoder().fit_transform(data['eadible'].values.reshape(-1, 1)).flatten(), name='number')

    elif identifier == "BIN:Congressional Voting":
        feature_names = [
            "Class Name",  #: 2 (democrat, republican)
            "handicapped-infants",  #: 2 (y,n)
            "water-project-cost-sharing",  #: 2 (y,n)
            "adoption-of-the-budget-resolution",  #: 2 (y,n)
            "physician-fee-freeze",  #: 2 (y,n)
            "el-salvador-aid",  #: 2 (y,n)
            "religious-groups-in-schools",  #: 2 (y,n)
            "anti-satellite-test-ban",  #: 2 (y,n)
            "aid-to-nicaraguan-contras",  #: 2 (y,n)
            "mx-missile",  #: 2 (y,n)
            "immigration",  #: 2 (y,n)
            "synfuels-corporation-cutback",  #: 2 (y,n)
            "education-spending",  #: 2 (y,n)
            "superfund-right-to-sue",  #: 2 (y,n)
            "crime",  #: 2 (y,n)
            "duty-free-exports",  #: 2 (y,n)
            "export-administration-act-south-africa",  #: 2 (y,n)
        ]

        data = pd.read_csv(
            "./datasets/Congressional Voting/house-votes-84.data",
            names=feature_names,
            index_col=False,
        )

        features_select = [
            "Class Name",  #: 2 (democrat, republican)
            "handicapped-infants",  #: 2 (y,n)
            "water-project-cost-sharing",  #: 2 (y,n)
            "adoption-of-the-budget-resolution",  #: 2 (y,n)
            "physician-fee-freeze",  #: 2 (y,n)
            "el-salvador-aid",  #: 2 (y,n)
            "religious-groups-in-schools",  #: 2 (y,n)
            "anti-satellite-test-ban",  #: 2 (y,n)
            "aid-to-nicaraguan-contras",  #: 2 (y,n)
            "mx-missile",  #: 2 (y,n)
            "immigration",  #: 2 (y,n)
            "synfuels-corporation-cutback",  #: 2 (y,n)
            "education-spending",  #: 2 (y,n)
            "superfund-right-to-sue",  #: 2 (y,n)
            "crime",  #: 2 (y,n)
            "duty-free-exports",  #: 2 (y,n)
            "export-administration-act-south-africa",  #: 2 (y,n)
        ]

        data = data[features_select]

        nominal_features = []
        ordinal_features = [
            "handicapped-infants",  #: 2 (y,n)
            "water-project-cost-sharing",  #: 2 (y,n)
            "adoption-of-the-budget-resolution",  #: 2 (y,n)
            "physician-fee-freeze",  #: 2 (y,n)
            "el-salvador-aid",  #: 2 (y,n)
            "religious-groups-in-schools",  #: 2 (y,n)
            "anti-satellite-test-ban",  #: 2 (y,n)
            "aid-to-nicaraguan-contras",  #: 2 (y,n)
            "mx-missile",  #: 2 (y,n)
            "immigration",  #: 2 (y,n)
            "synfuels-corporation-cutback",  #: 2 (y,n)
            "education-spending",  #: 2 (y,n)
            "superfund-right-to-sue",  #: 2 (y,n)
            "crime",  #: 2 (y,n)
            "duty-free-exports",  #: 2 (y,n)
            "export-administration-act-south-africa",  #: 2 (y,n)
        ]

        X_data = data.drop(["Class Name"], axis=1)
        y_data = pd.Series(
            OrdinalEncoder()
            .fit_transform(data["Class Name"].values.reshape(-1, 1))
            .flatten(),
            name="number",
        )

    elif identifier == "BIN:Hepatitis":
        feature_names = [
            "Class",  #: DIE, LIVE
            "AGE",  #: 10, 20, 30, 40, 50, 60, 70, 80
            "SEX",  #: male, female
            "STEROID",  #: no, yes
            "ANTIVIRALS",  #: no, yes
            "FATIGUE",  #: no, yes
            "MALAISE",  #: no, yes
            "ANOREXIA",  #: no, yes
            "LIVER BIG",  #: no, yes
            "LIVER FIRM",  #: no, yes
            "SPLEEN PALPABLE",  #: no, yes
            "SPIDERS",  #: no, yes
            "ASCITES",  #: no, yes
            "VARICES",  #: no, yes
            "BILIRUBIN",  #: 0.39, 0.80, 1.20, 2.00, 3.00, 4.00
            # -- see the note below
            "ALK PHOSPHATE",  #: 33, 80, 120, 160, 200, 250
            "SGOT",  #: 13, 100, 200, 300, 400, 500,
            "ALBUMIN",  #: 2.1, 3.0, 3.8, 4.5, 5.0, 6.0
            "PROTIME",  #: 10, 20, 30, 40, 50, 60, 70, 80, 90
            "HISTOLOGY",  #: no, yes
        ]

        data = pd.read_csv(
            "./datasets/Hepatitis/hepatitis.data",
            names=feature_names,
            index_col=False,
            na_values="?",
        )
        data = data.fillna(data.mode().iloc[0])

        features_select = [
            "Class",  #: DIE, LIVE
            "AGE",  #: 10, 20, 30, 40, 50, 60, 70, 80
            "SEX",  #: male, female
            "STEROID",  #: no, yes
            "ANTIVIRALS",  #: no, yes
            "FATIGUE",  #: no, yes
            "MALAISE",  #: no, yes
            "ANOREXIA",  #: no, yes
            "LIVER BIG",  #: no, yes
            "LIVER FIRM",  #: no, yes
            "SPLEEN PALPABLE",  #: no, yes
            "SPIDERS",  #: no, yes
            "ASCITES",  #: no, yes
            "VARICES",  #: no, yes
            "BILIRUBIN",  #: 0.39, 0.80, 1.20, 2.00, 3.00, 4.00
            # -- see the note below
            "ALK PHOSPHATE",  #: 33, 80, 120, 160, 200, 250
            "SGOT",  #: 13, 100, 200, 300, 400, 500,
            "ALBUMIN",  #: 2.1, 3.0, 3.8, 4.5, 5.0, 6.0
            "PROTIME",  #: 10, 20, 30, 40, 50, 60, 70, 80, 90
            "HISTOLOGY",  #: no, yes
        ]

        data = data[features_select]

        nominal_features = []
        ordinal_features = [
            "SEX",  #: male, female
            "STEROID",  #: no, yes
            "ANTIVIRALS",  #: no, yes
            "FATIGUE",  #: no, yes
            "MALAISE",  #: no, yes
            "ANOREXIA",  #: no, yes
            "LIVER BIG",  #: no, yes
            "LIVER FIRM",  #: no, yes
            "SPLEEN PALPABLE",  #: no, yes
            "SPIDERS",  #: no, yes
            "ASCITES",  #: no, yes
            "VARICES",  #: no, yes
            "HISTOLOGY",  #: no, yes
        ]

        X_data = data.drop(["Class"], axis=1)
        y_data = pd.Series(
            OrdinalEncoder()
            .fit_transform(data["Class"].values.reshape(-1, 1))
            .flatten(),
            name="number",
        )

    elif identifier == "BIN:Blood Transfusion":

        feature_names = [
            "R",  # (Recency - months since last donation),
            "F",  # (Frequency - total number of donation),
            "M",  # (Monetary - total blood donated in c.c.),
            "T",  # (Time - months since first donation), and
            "a",  # binary variable representing whether he/she donated blood in March 2007 (1 stand for donating blood; 0 stands for not donating blood).
        ]

        data = pd.read_csv(
            "./datasets/Transfusion/transfusion.data",
            names=feature_names,
            index_col=False,
            header=0,
        )  # , header=0)

        features_select = [
            "R",  # (Recency - months since last donation),
            "F",  # (Frequency - total number of donation),
            "M",  # (Monetary - total blood donated in c.c.),
            "T",  # (Time - months since first donation), and
            "a",  # binary variable representing whether he/she donated blood in March 2007 (1 stand for donating blood; 0 stands for not donating blood).
        ]

        data = data[features_select]

        nominal_features = []
        ordinal_features = []

        X_data = data.drop(["a"], axis=1)
        y_data = pd.Series(
            OrdinalEncoder().fit_transform(data["a"].values.reshape(-1, 1)).flatten(),
            name="a",
        )

    elif identifier == "BIN:German":

        feature_names = [
            "Status of existing checking account",  # nominal
            "Duration in month",  # numeric
            "Credit history",  # nominal
            "Purpose",  # nominal
            "Credit amount",  # numeric
            "Savings account/bonds",  # nominal
            "Present employment since",  # nominal
            "Installment rate in percentage of disposable income",  # numeric
            "Personal status and sex",  # nominal
            "Other debtors / guarantors",  # nominal
            "Present residence since",  # numeric
            "Property",  # nominal
            "Age in years",  # numeric
            "Other installment plans",  # nominal
            "Housing",  # nominal
            "Number of existing credits at this bank",  # numeric
            "Job",  # nominal
            "Number of people being liable to provide maintenance for",  # numeric
            "Telephone",  # binary
            "foreign worker",  # binary
            "label",
        ]

        data = pd.read_csv(
            "./datasets/German/german.data",
            names=feature_names,
            index_col=False,
            delimiter=" ",
        )  # , header=0)#, header=0)

        features_select = [
            "Status of existing checking account",
            "Duration in month",
            "Credit history",
            "Purpose",
            "Credit amount",
            "Savings account/bonds",
            "Present employment since",
            "Installment rate in percentage of disposable income",
            "Personal status and sex",
            "Other debtors / guarantors",
            "Present residence since",
            "Property",
            "Age in years",
            "Other installment plans",
            "Housing",
            "Number of existing credits at this bank",
            "Job",
            "Number of people being liable to provide maintenance for",
            "Telephone",
            "foreign worker",
            "label",
        ]

        data = data[features_select]

        nominal_features = []
        ordinal_features = [
            "Status of existing checking account",  # nominal
            "Credit history",  # nominal
            "Purpose",  # nominal
            "Savings account/bonds",  # nominal
            "Present employment since",  # nominal
            "Personal status and sex",  # nominal
            "Other debtors / guarantors",  # nominal
            "Property",  # nominal
            "Other installment plans",  # nominal
            "Housing",  # nominal
            "Job",  # nominal
            "Telephone",  # binary
            "foreign worker",  # binary
        ]

        X_data = data.drop(["label"], axis=1)
        y_data = pd.Series(
            OrdinalEncoder()
            .fit_transform(data["label"].values.reshape(-1, 1))
            .flatten(),
            name="label",
        )

    elif identifier == "BIN:Banknote Authentication":

        feature_names = [
            "variance",  # of Wavelet Transformed image (continuous)
            "skewness",  # of Wavelet Transformed image (continuous)
            "curtosis",  # of Wavelet Transformed image (continuous)
            "entropy",  # of image (continuous)
            "class",  # (integer)
        ]

        data = pd.read_csv(
            "./datasets/Banknote/data_banknote_authentication.txt",
            names=feature_names,
            index_col=False,
        )  # , delimiter=' ')#, header=0)

        features_select = [
            "variance",  # of Wavelet Transformed image (continuous)
            "skewness",  # of Wavelet Transformed image (continuous)
            "curtosis",  # of Wavelet Transformed image (continuous)
            "entropy",  # of image (continuous)
            "class",  # (integer)
        ]

        data = data[features_select]

        nominal_features = []
        ordinal_features = []

        X_data = data.drop(["class"], axis=1)
        y_data = pd.Series(
            OrdinalEncoder()
            .fit_transform(data["class"].values.reshape(-1, 1))
            .flatten(),
            name="class",
        )

    elif identifier == "Spambase":

        feature_names = flatten_list(
            [
                ["word_freq_WORD_" + str(i) for i in range(48)],
                ["char_freq_CHAR_" + str(i) for i in range(6)],
                "capital_run_length_average",
                "capital_run_length_longest",
                "capital_run_length_total",
                "spam_type",
            ]
        )

        data = pd.read_csv(
            "./datasets/Spambase/spambase.data",
            names=feature_names,
            index_col=False,
        )  # , header=2)#, header=0)#, delimiter=' ')#, header=0)

        features_select = flatten_list(
            [
                ["word_freq_WORD_" + str(i) for i in range(48)],
                ["char_freq_CHAR_" + str(i) for i in range(6)],
                "capital_run_length_average",
                "capital_run_length_longest",
                "capital_run_length_total",
                "spam_type",
            ]
        )

        data = data[features_select]

        nominal_features = []
        ordinal_features = []

        X_data = data.drop(["spam_type"], axis=1)
        y_data = pd.Series(
            OrdinalEncoder()
            .fit_transform(data["spam_type"].values.reshape(-1, 1))
            .flatten(),
            name="spam_type",
        )

    elif identifier == "Iris":

        feature_names = [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
            "class",
        ]
        # Wisconsin Prognostic Breast Cancer
        data = pd.read_csv(
            "./datasets/Iris/iris.data", names=feature_names, index_col=False
        )

        features_select = [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
            "class",
        ]

        data = data[features_select]

        nominal_features = []
        ordinal_features = []
        X_data = data.drop(["class"], axis=1)
        y_data = pd.Series(
            OrdinalEncoder()
            .fit_transform(data["class"].values.reshape(-1, 1))
            .flatten(),
            name="class",
        )

    elif identifier == "MULT:Annealing":

        feature_names = [
            "family",  # --,GB,GK,GS,TN,ZA,ZF,ZH,ZM,ZS
            "product-type",  # C, H, G
            "steel",  # -,R,A,U,K,M,S,W,V
            "carbon",  #: continuous
            "hardness",  #: continuous
            "temper_rolling",  #: -,T
            "condition",  #: -,S,A,X
            "formability",  #: -,1,2,3,4,5
            "strength",  #: continuous
            "non-ageing",  #: -,N
            "surface-finish",  #: P,M,-
            "surface-quality",  #: -,D,E,F,G
            "enamelability",  #: -,1,2,3,4,5
            "bc",  #: Y,-
            "bf",  #: Y,-
            "bt",  #: Y,-
            "bw/me",  #: B,M,-
            "bl",  #: Y,-
            "m",  #: Y,-
            "chrom",  #: C,-
            "phos",  #: P,-
            "cbond",  #: Y,-
            "marvi",  #: Y,-
            "exptl",  #: Y,-
            "ferro",  #: Y,-
            "corr",  #: Y,-
            "blue/bright/varn/clean",  #: B,R,V,C,-
            "lustre",  #: Y,-
            "jurofm",  #: Y,-
            "s",  #: Y,-
            "p",  #: Y,-
            "shape",  #: COIL, SHEET
            "thick",  #: continuous
            "width",  #: continuous
            "len",  #: continuous
            "oil",  #: -,Y,N
            "bore",  #: 0000,0500,0600,0760
            "packing",  #: -,1,2,3
            "classes",  #: 1,2,3,4,5,U
        ]

        data = pd.read_csv(
            "./datasets/Annealing/anneal.data",
            names=feature_names,
            index_col=False,
        )

        features_select = [
            "family",  # --,GB,GK,GS,TN,ZA,ZF,ZH,ZM,ZS
            "product-type",  # C, H, G
            "steel",  # -,R,A,U,K,M,S,W,V
            "carbon",  #: continuous
            "hardness",  #: continuous
            "temper_rolling",  #: -,T
            "condition",  #: -,S,A,X
            "formability",  #: -,1,2,3,4,5
            "strength",  #: continuous
            "non-ageing",  #: -,N
            "surface-finish",  #: P,M,-
            "surface-quality",  #: -,D,E,F,G
            "enamelability",  #: -,1,2,3,4,5
            "bc",  #: Y,-
            "bf",  #: Y,-
            "bt",  #: Y,-
            "bw/me",  #: B,M,-
            "bl",  #: Y,-
            "m",  #: Y,-
            "chrom",  #: C,-
            "phos",  #: P,-
            "cbond",  #: Y,-
            "marvi",  #: Y,-
            "exptl",  #: Y,-
            "ferro",  #: Y,-
            "corr",  #: Y,-
            "blue/bright/varn/clean",  #: B,R,V,C,-
            "lustre",  #: Y,-
            "jurofm",  #: Y,-
            "s",  #: Y,-
            "p",  #: Y,-
            "shape",  #: COIL, SHEET
            "thick",  #: continuous
            "width",  #: continuous
            "len",  #: continuous
            "oil",  #: -,Y,N
            "bore",  #: 0000,0500,0600,0760
            "packing",  #: -,1,2,3
            "classes",  #: 1,2,3,4,5,U
        ]

        data = data[features_select]

        nominal_features = []
        ordinal_features = [
            "family",  # --,GB,GK,GS,TN,ZA,ZF,ZH,ZM,ZS
            "product-type",  # C, H, G
            "steel",  # -,R,A,U,K,M,S,W,V
            "temper_rolling",  #: -,T
            "condition",  #: -,S,A,X
            "bw/me",  #: B,M,-
            "blue/bright/varn/clean",  #: B,R,V,C,-
            "oil",  #: -,Y,N
            "bore",  #: 0000,0500,0600,0760
            "formability",  #: -,1,2,3,4,5
            "non-ageing",  #: -,N
            "surface-finish",  #: P,M,-
            "surface-quality",  #: -,D,E,F,G
            "enamelability",  #: -,1,2,3,4,5
            "bc",  #: Y,-
            "bf",  #: Y,-
            "bt",  #: Y,-
            "bl",  #: Y,-
            "m",  #: Y,-
            "chrom",  #: C,-
            "phos",  #: P,-
            "cbond",  #: Y,-
            "marvi",  #: Y,-
            "exptl",  #: Y,-
            "ferro",  #: Y,-
            "corr",  #: Y,-
            "lustre",  #: Y,-
            "jurofm",  #: Y,-
            "s",  #: Y,-
            "p",  #: Y,-
            "shape",  #: COIL, SHEET
            "packing",  #: -,1,2,3
        ]

        X_data = data.drop(["classes"], axis=1)
        y_data = pd.Series(
            OrdinalEncoder()
            .fit_transform(data["classes"].values.reshape(-1, 1))
            .flatten(),
            name="classes",
        )

    elif identifier == "MULT:Glass":

        feature_names = [
            "Id number",  #: 1 to 214
            "RI",  #: refractive index
            "Na",  #: Sodium (unit measurement: weight percent in corresponding oxide, as are attributes 4-10)
            "Mg",  #: Magnesium
            "Al",  #: Aluminum
            "Si",  #: Silicon
            "K",  #: Potassium
            "Ca",  #: Calcium
            "Ba",  #: Barium
            "Fe",  #: Iron
            "Type of glass",  #: (class attribute)
        ]

        data = pd.read_csv(
            "./datasets/Glass/glass.data",
            names=feature_names,
            index_col=False,
        )

        features_select = [
            #'Id number',#: 1 to 214
            "RI",  #: refractive index
            "Na",  #: Sodium (unit measurement: weight percent in corresponding oxide, as are attributes 4-10)
            "Mg",  #: Magnesium
            "Al",  #: Aluminum
            "Si",  #: Silicon
            "K",  #: Potassium
            "Ca",  #: Calcium
            "Ba",  #: Barium
            "Fe",  #: Iron
            "Type of glass",  #: (class attribute)
        ]

        data = data[features_select]

        nominal_features = []
        ordinal_features = []

        X_data = data.drop(["Type of glass"], axis=1)
        y_data = pd.Series(
            OrdinalEncoder()
            .fit_transform(data["Type of glass"].values.reshape(-1, 1))
            .flatten(),
            name="Type of glass",
        )

    elif identifier == "MULT:Solar Flare":

        feature_names = [
            "Code for class",  # (modified Zurich class) (A,B,C,D,E,F,H)
            "Code for largest spot size",  # (X,R,S,A,H,K)
            "Code for spot distribution",  # (X,O,I,C)
            "Activity",  # (1 = reduced, 2 = unchanged)
            "Evolution",  # (1 = decay, 2 = no growth, 3 = growth)
            "Previous 24 hour flare activity code",  # (1 = nothing as big as an M1, 2 = one M1, 3 = more activity than one M1)
            "Historically-complex",  # (1 = Yes, 2 = No)
            "Did region become historically complex on this pass across the suns disk",  # (1 = yes, 2 = no)
            "Area",  # (1 = small, 2 = large)
            "Area of the largest spot",  # (1 = <=5, 2 = >5)
            "C-class flares production by this region in the following 24 hours (common flares)",  # ; Number
            "M-class flares production by this region in the following 24 hours (moderate flares)",  # ; Number
            "X-class flares production by this region in the following 24 hours (severe flares)",  # ; Number
        ]

        data1 = pd.read_csv(
            "./datasets/Solar Flare/flare.data1",
            names=feature_names,
            index_col=False,
            delimiter=" ",
            header=0,
        )
        data2 = pd.read_csv(
            "./datasets/Solar Flare/flare.data2",
            names=feature_names,
            index_col=False,
            delimiter=" ",
            header=0,
        )

        data = pd.concat([data1, data2])

        features_select = [
            "Code for class",  # (modified Zurich class) (A,B,C,D,E,F,H)
            "Code for largest spot size",  # (X,R,S,A,H,K)
            "Code for spot distribution",  # (X,O,I,C)
            "Activity",  # (1 = reduced, 2 = unchanged)
            "Evolution",  # (1 = decay, 2 = no growth, 3 = growth)
            "Previous 24 hour flare activity code",  # (1 = nothing as big as an M1, 2 = one M1, 3 = more activity than one M1)
            "Historically-complex",  # (1 = Yes, 2 = No)
            "Did region become historically complex on this pass across the suns disk",  # (1 = yes, 2 = no)
            "Area",  # (1 = small, 2 = large)
            "Area of the largest spot",  # (1 = <=5, 2 = >5)
            "C-class flares production by this region in the following 24 hours (common flares)",  # ; Number
            #'M-class flares production by this region in the following 24 hours (moderate flares)',#; Number
            #'X-class flares production by this region in the following 24 hours (severe flares)',#; Number
        ]

        data = data[features_select]

        nominal_features = []
        ordinal_features = [
            "Code for class",  # (modified Zurich class) (A,B,C,D,E,F,H)
            "Code for largest spot size",  # (X,R,S,A,H,K)
            "Code for spot distribution",  # (X,O,I,C)
            "Activity",  # (1 = reduced, 2 = unchanged)
            "Evolution",  # (1 = decay, 2 = no growth, 3 = growth)
            "Previous 24 hour flare activity code",  # (1 = nothing as big as an M1, 2 = one M1, 3 = more activity than one M1)
            "Historically-complex",  # (1 = Yes, 2 = No)
            "Did region become historically complex on this pass across the suns disk",  # (1 = yes, 2 = no)
            "Area",  # (1 = small, 2 = large)
            "Area of the largest spot",  # (1 = <=5, 2 = >5)
        ]

        X_data = data.drop(
            [
                "C-class flares production by this region in the following 24 hours (common flares)"
            ],
            axis=1,
        )
        y_data = pd.Series(
            OrdinalEncoder()
            .fit_transform(
                data[
                    "C-class flares production by this region in the following 24 hours (common flares)"
                ].values.reshape(-1, 1)
            )
            .flatten(),
            name="C-class flares production by this region in the following 24 hours (common flares)",
        )

    elif identifier == "Splice":

        feature_names = flatten_list(
            [
                "One of {n ei ie}",  # , indicating the class.
                "instance name",
                "sequence",
            ]
        )

        data_raw = pd.read_csv(
            "./datasets/Splice/splice.data",
            names=feature_names,
            index_col=False,
        )  # , header=0)
        data_np = np.hstack(
            [
                data_raw["One of {n ei ie}"].values.reshape(-1, 1),
                data_raw["instance name"].values.reshape(-1, 1),
                np.array(
                    [
                        split_string[-61:-1]
                        for split_string in data_raw["sequence"].str.split("")
                    ]
                ),
            ]
        )

        columnnames = flatten_list(
            [
                "One of {n ei ie}",  # , indicating the class.
                "instance name",
                [str(i) for i in range(-30, 30)],
            ]
        )
        data = pd.DataFrame(data=data_np, columns=columnnames)

        features_select = flatten_list(
            [
                "One of {n ei ie}",  # , indicating the class.
                #'instance name',
                [str(i) for i in range(-30, 30)],
            ]
        )

        data = data[features_select]

        nominal_features = flatten_list([])
        ordinal_features = flatten_list(
            [
                [str(i) for i in range(-30, 30)],
            ]
        )

        X_data = data.drop(["One of {n ei ie}"], axis=1)
        y_data = pd.Series(
            OrdinalEncoder()
            .fit_transform(data["One of {n ei ie}"].values.reshape(-1, 1))
            .flatten(),
            name="One of {n ei ie}",
        )

    elif identifier == "MULT:Wine":

        feature_names = [
            "Alcohol",
            "Malic acid",
            "Ash",
            "Alcalinity of ash",
            "Magnesium",
            "Total phenols",
            "Flavanoids",
            "Nonflavanoid phenols",
            "Proanthocyanins",
            "Color intensity",
            "Hue",
            "OD280/OD315 of diluted wines",
            "Proline",
        ]

        data = pd.read_csv(
            "./datasets/Wine/wine.data", names=feature_names, index_col=False
        )  # , header=0)

        features_select = [
            "Alcohol",
            "Malic acid",
            "Ash",
            "Alcalinity of ash",
            "Magnesium",
            "Total phenols",
            "Flavanoids",
            "Nonflavanoid phenols",
            "Proanthocyanins",
            "Color intensity",
            "Hue",
            "OD280/OD315 of diluted wines",
            "Proline",
        ]

        data = data[features_select]

        nominal_features = []
        ordinal_features = []

        X_data = data.drop(["Alcohol"], axis=1)
        y_data = pd.Series(
            OrdinalEncoder()
            .fit_transform(data["Alcohol"].values.reshape(-1, 1))
            .flatten(),
            name="Alcohol",
        )

    elif identifier == "MULT:Dermatology":

        feature_names = [
            "erythema",  #
            "scaling",  #
            "definite borders",  #
            "itching",  #
            "koebner phenomenon",  #
            "polygonal papules",  #
            "follicular papules",  #
            "oral mucosal involvement",  #
            "knee and elbow involvement",  #
            "scalp involvement",  #
            "family history",  # , (0 or 1)
            "melanin incontinence",  #
            "eosinophils in the infiltrate",  #
            "PNL infiltrate",  #
            "fibrosis of the papillary dermis",  #
            "exocytosis",  #
            "acanthosis",  #
            "hyperkeratosis",  #
            "parakeratosis",  #
            "clubbing of the rete ridges",  #
            "elongation of the rete ridges",  #
            "thinning of the suprapapillary epidermis",  #
            "spongiform pustule",  #
            "munro microabcess",  #
            "focal hypergranulosis",  #
            "disappearance of the granular layer",  #
            "vacuolisation and damage of basal layer",  #
            "spongiosis",  #
            "saw-tooth appearance of retes",  #
            "follicular horn plug",  #
            "perifollicular parakeratosis",  #
            "inflammatory monoluclear inflitrate",  #
            "band-like infiltrate",  #
            "Age (linear)",  #
            "diagnosis",
        ]

        data = pd.read_csv(
            "./datasets/Dermatology/dermatology.data",
            names=feature_names,
            index_col=False,
        )
        # data['Age (linear)'].fillna(data['Age (linear)'].mean(), inplace = True)
        data["Age (linear)"].replace(
            ["?"],
            pd.to_numeric(data["Age (linear)"], errors="coerce").mean(),
            inplace=True,
        )
        data["Age (linear)"] = data["Age (linear)"].astype(float)
        features_select = [
            "erythema",  #
            "scaling",  #
            "definite borders",  #
            "itching",  #
            "koebner phenomenon",  #
            "polygonal papules",  #
            "follicular papules",  #
            "oral mucosal involvement",  #
            "knee and elbow involvement",  #
            "scalp involvement",  #
            "family history",  # , (0 or 1)
            "melanin incontinence",  #
            "eosinophils in the infiltrate",  #
            "PNL infiltrate",  #
            "fibrosis of the papillary dermis",  #
            "exocytosis",  #
            "acanthosis",  #
            "hyperkeratosis",  #
            "parakeratosis",  #
            "clubbing of the rete ridges",  #
            "elongation of the rete ridges",  #
            "thinning of the suprapapillary epidermis",  #
            "spongiform pustule",  #
            "munro microabcess",  #
            "focal hypergranulosis",  #
            "disappearance of the granular layer",  #
            "vacuolisation and damage of basal layer",  #
            "spongiosis",  #
            "saw-tooth appearance of retes",  #
            "follicular horn plug",  #
            "perifollicular parakeratosis",  #
            "inflammatory monoluclear inflitrate",  #
            "band-like infiltrate",  #
            "Age (linear)",  #
            "diagnosis",
        ]

        data = data[features_select]

        nominal_features = []
        ordinal_features = []

        X_data = data.drop(["diagnosis"], axis=1)
        y_data = pd.Series(
            OrdinalEncoder()
            .fit_transform(data["diagnosis"].values.reshape(-1, 1))
            .flatten(),
            name="diagnosis",
        )

    elif identifier == "MULT:Balance Scale":

        feature_names = [
            "Class Name",  #: 3 (L, B, R)
            "Left-Weight",  #: 5 (1, 2, 3, 4, 5)
            "Left-Distance",  #: 5 (1, 2, 3, 4, 5)
            "Right-Weight",  #: 5 (1, 2, 3, 4, 5)
            "Right-Distance",  #: 5 (1, 2, 3, 4, 5)
        ]

        data = pd.read_csv(
            "./datasets/Balance Scale/balance-scale.data",
            names=feature_names,
            index_col=False,
        )  # , header=0)

        features_select = [
            "Class Name",  #: 3 (L, B, R)
            "Left-Weight",  #: 5 (1, 2, 3, 4, 5)
            "Left-Distance",  #: 5 (1, 2, 3, 4, 5)
            "Right-Weight",  #: 5 (1, 2, 3, 4, 5)
            "Right-Distance",  #: 5 (1, 2, 3, 4, 5)
        ]

        data = data[features_select]

        nominal_features = []
        ordinal_features = []

        X_data = data.drop(["Class Name"], axis=1)
        y_data = pd.Series(
            OrdinalEncoder()
            .fit_transform(data["Class Name"].values.reshape(-1, 1))
            .flatten(),
            name="Class Name",
        )

    elif identifier == "MULT:Contraceptive":

        feature_names = [
            "Wife's age",  # (numerical)
            "Wife's education",  # (categorical) 1=low, 2, 3, 4=high
            "Husband's education",  # (categorical) 1=low, 2, 3, 4=high
            "Number of children ever born",  # (numerical)
            "Wife's religion",  # (binary) 0=Non-Islam, 1=Islam
            "Wife's now working?",  # (binary) 0=Yes, 1=No
            "Husband's occupation",  # (categorical) 1, 2, 3, 4
            "Standard-of-living index",  # (categorical) 1=low, 2, 3, 4=high
            "Media exposure",  # (binary) 0=Good, 1=Not good
            "Contraceptive method used",  # (class attribute) 1=No-use, 2=Long-term, 3=Short-term
        ]

        data = pd.read_csv(
            "./datasets/Contraceptive/cmc.data",
            names=feature_names,
            index_col=False,
        )  # , delimiter=' ')#, header=0)

        features_select = [
            "Wife's age",  # (numerical)
            "Wife's education",  # (categorical) 1=low, 2, 3, 4=high
            "Husband's education",  # (categorical) 1=low, 2, 3, 4=high
            "Number of children ever born",  # (numerical)
            "Wife's religion",  # (binary) 0=Non-Islam, 1=Islam
            "Wife's now working?",  # (binary) 0=Yes, 1=No
            "Husband's occupation",  # (categorical) 1, 2, 3, 4
            "Standard-of-living index",  # (categorical) 1=low, 2, 3, 4=high
            "Media exposure",  # (binary) 0=Good, 1=Not good
            "Contraceptive method used",  # (class attribute) 1=No-use, 2=Long-term, 3=Short-term
        ]

        data = data[features_select]

        nominal_features = []
        ordinal_features = [
            "Wife's education",  # (categorical) 1=low, 2, 3, 4=high
            "Husband's education",  # (categorical) 1=low, 2, 3, 4=high
            "Husband's occupation",  # (categorical) 1, 2, 3, 4
            "Standard-of-living index",  # (categorical) 1=low, 2, 3, 4=high
        ]

        X_data = data.drop(["Contraceptive method used"], axis=1)
        y_data = pd.Series(
            OrdinalEncoder()
            .fit_transform(data["Contraceptive method used"].values.reshape(-1, 1))
            .flatten(),
            name="Contraceptive method used",
        )

    elif identifier == "Segment":

        feature_names = [
            "target",
            "region-centroid-col",  #: the column of the center pixel of the region.
            "region-centroid-row",  #: the row of the center pixel of the region.
            "region-pixel-count",  #: the number of pixels in a region = 9.
            "short-line-density-5",  #: the results of a line extractoin algorithm that counts how many lines of length 5 (any orientation) with low contrast, less than or equal to 5, go through the region.
            "short-line-density-2",  #: same as short-line-density-5 but counts lines of high contrast, greater than 5.
            "vedge-mean",  #: measure the contrast of horizontally adjacent pixels in the region. There are 6, the mean and standard deviation are given. This attribute is used as a vertical edge detector.
            "vegde-sd",  #: (see 6)
            "hedge-mean",  #: measures the contrast of vertically adjacent pixels. Used for horizontal line detection.
            "hedge-sd",  #: (see 8).
            "intensity-mean",  #: the average over the region of (R + G + B)/3
            "rawred-mean",  #: the average over the region of the R value.
            "rawblue-mean",  #: the average over the region of the B value.
            "rawgreen-mean",  #: the average over the region of the G value.
            "exred-mean",  #: measure the excess red: (2R - (G + B))
            "exblue-mean",  #: measure the excess blue: (2B - (G + R))
            "exgreen-mean",  #: measure the excess green: (2G - (R + B))
            "value-mean",  #: 3-d nonlinear transformation of RGB. (Algorithm can be found in Foley and VanDam, Fundamentals of Interactive Computer Graphics)
            "saturatoin-mean",  #: (see 17)
            "hue-mean",  #: (see 17)
        ]

        data1 = pd.read_csv(
            "./datasets/Segment/segmentation.data",
            names=feature_names,
            index_col=False,
            header=2,
        )  # , header=0)#, delimiter=' ')#, header=0)
        data2 = pd.read_csv(
            "./datasets/Segment/segmentation.test",
            names=feature_names,
            index_col=False,
            header=2,
        )  # , header=0)#, delimiter=' ')#, header=0)
        data = pd.concat([data1, data2])

        features_select = [
            "target",
            "region-centroid-col",  #: the column of the center pixel of the region.
            "region-centroid-row",  #: the row of the center pixel of the region.
            "region-pixel-count",  #: the number of pixels in a region = 9.
            "short-line-density-5",  #: the results of a line extractoin algorithm that counts how many lines of length 5 (any orientation) with low contrast, less than or equal to 5, go through the region.
            "short-line-density-2",  #: same as short-line-density-5 but counts lines of high contrast, greater than 5.
            "vedge-mean",  #: measure the contrast of horizontally adjacent pixels in the region. There are 6, the mean and standard deviation are given. This attribute is used as a vertical edge detector.
            "vegde-sd",  #: (see 6)
            "hedge-mean",  #: measures the contrast of vertically adjacent pixels. Used for horizontal line detection.
            "hedge-sd",  #: (see 8).
            "intensity-mean",  #: the average over the region of (R + G + B)/3
            "rawred-mean",  #: the average over the region of the R value.
            "rawblue-mean",  #: the average over the region of the B value.
            "rawgreen-mean",  #: the average over the region of the G value.
            "exred-mean",  #: measure the excess red: (2R - (G + B))
            "exblue-mean",  #: measure the excess blue: (2B - (G + R))
            "exgreen-mean",  #: measure the excess green: (2G - (R + B))
            "value-mean",  #: 3-d nonlinear transformation of RGB. (Algorithm can be found in Foley and VanDam, Fundamentals of Interactive Computer Graphics)
            "saturatoin-mean",  #: (see 17)
            "hue-mean",  #: (see 17)
        ]

        data = data[features_select]

        nominal_features = []
        ordinal_features = []

        X_data = data.drop(["target"], axis=1)
        y_data = pd.Series(
            OrdinalEncoder()
            .fit_transform(data["target"].values.reshape(-1, 1))
            .flatten(),
            name="target",
        )
        # Reset the indices of both X_data and y_data to ensure they align
        X_data = X_data.reset_index(drop=True)
        y_data = y_data.reset_index(drop=True)

    elif identifier == "MULT:Landsat":

        feature_names = flatten_list(
            [
                [
                    ["spectral_" + str(i) + "_pixel_" + str(j) for i in range(4)]
                    for j in range(9)
                ],
                "target",
            ]
        )

        data1 = pd.read_csv(
            "./datasets/Landsat/sat.trn",
            names=feature_names,
            index_col=False,
            delimiter=" ",
        )  # , header=2)#, header=0)#, delimiter=' ')#, header=0)
        data2 = pd.read_csv(
            "./datasets/Landsat/sat.tst",
            names=feature_names,
            index_col=False,
            delimiter=" ",
        )  # , header=2)#, header=0)#, delimiter=' ')#, header=0)
        data = pd.concat([data1, data2])

        features_select = flatten_list(
            [
                [
                    ["spectral_" + str(i) + "_pixel_" + str(j) for i in range(4)]
                    for j in range(9)
                ],
                "target",
            ]
        )

        data = data[features_select]

        nominal_features = []
        ordinal_features = []

        X_data = data.drop(["target"], axis=1)
        y_data = pd.Series(
            OrdinalEncoder()
            .fit_transform(data["target"].values.reshape(-1, 1))
            .flatten(),
            name="target",
        )
        # Reset the indices of both X_data and y_data to ensure they align
        X_data = X_data.reset_index(drop=True)
        y_data = y_data.reset_index(drop=True)

    elif identifier == "MULT:Lymphography":
        feature_names = [
            "class",  #: normal find, metastases, malign lymph, fibrosis
            "lymphatics",  #: normal, arched, deformed, displaced
            "block of affere",  #: no, yes
            "bl. of lymph. c",  #: no, yes
            "bl. of lymph. s",  #: no, yes
            "by pass",  #: no, yes
            "extravasates",  #: no, yes
            "regeneration of",  #: no, yes
            "early uptake in",  #: no, yes
            "lym.nodes dimin",  #: 0-3
            "lym.nodes enlar",  #: 1-4
            "changes in lym.",  #: bean, oval, round
            "efect in node",  #: no, lacunar, lac. marginal, lac. central
            "changes in node",  #: no, lacunar, lac. margin, lac. central
            "changes in stru",  #: no, grainy, drop-like, coarse, diluted, reticular, stripped, faint,
            "special forms",  #: no, chalices, vesicles
            "dislocation of",  #: no, yes
            "exclusion of no",  #: no, yes
            "no. of nodes in",  #: 0-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, >=70
        ]

        data = pd.read_csv(
            "./datasets/Lymphography/lymphography.data",
            names=feature_names,
            index_col=False,
        )

        features_select = [
            "class",  #: normal find, metastases, malign lymph, fibrosis
            "lymphatics",  #: normal, arched, deformed, displaced
            "block of affere",  #: no, yes
            "bl. of lymph. c",  #: no, yes
            "bl. of lymph. s",  #: no, yes
            "by pass",  #: no, yes
            "extravasates",  #: no, yes
            "regeneration of",  #: no, yes
            "early uptake in",  #: no, yes
            "lym.nodes dimin",  #: 0-3
            "lym.nodes enlar",  #: 1-4
            "changes in lym.",  #: bean, oval, round
            "efect in node",  #: no, lacunar, lac. marginal, lac. central
            "changes in node",  #: no, lacunar, lac. margin, lac. central
            "changes in stru",  #: no, grainy, drop-like, coarse, diluted, reticular, stripped, faint,
            "special forms",  #: no, chalices, vesicles
            "dislocation of",  #: no, yes
            "exclusion of no",  #: no, yes
            "no. of nodes in",  #: 0-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, >=70
        ]

        data = data[features_select]

        nominal_features = []
        ordinal_features = [
            "lymphatics",  #: normal, arched, deformed, displaced
            "lym.nodes dimin",  #: 0-3
            "lym.nodes enlar",  #: 1-4
            "changes in lym.",  #: bean, oval, round
            "efect in node",  #: no, lacunar, lac. marginal, lac. central
            "changes in node",  #: no, lacunar, lac. margin, lac. central
            "changes in stru",  #: no, grainy, drop-like, coarse, diluted, reticular, stripped, faint,
            "special forms",  #: no, chalices, vesicles
        ]

        X_data = data.drop(["class"], axis=1)
        y_data = pd.Series(
            OrdinalEncoder()
            .fit_transform(data["class"].values.reshape(-1, 1))
            .flatten(),
            name="number",
        )

    elif identifier == "MULT:Zoo":
        feature_names = [
            "animal name",  #: Unique for each instance
            "hair",  #: Boolean
            "feathers",  #: Boolean
            "eggs",  #: Boolean
            "milk",  #: Boolean
            "airborne",  #: Boolean
            "aquatic",  #: Boolean
            "predator",  #: Boolean
            "toothed",  #: Boolean
            "backbone",  #: Boolean
            "breathes",  #: Boolean
            "venomous",  #: Boolean
            "fins",  #: Boolean
            "legs",  #: Numeric (set of values: {0,2,4,5,6,8})
            "tail",  #: Boolean
            "domestic",  #: Boolean
            "catsize",  #: Boolean
            "type",  #: Numeric (integer values in range [1,7])
        ]

        data = pd.read_csv(
            "./datasets/Zoo/zoo.data", names=feature_names, index_col=False
        )

        features_select = [
            #'animal name',#: Unique for each instance
            "hair",  #: Boolean
            "feathers",  #: Boolean
            "eggs",  #: Boolean
            "milk",  #: Boolean
            "airborne",  #: Boolean
            "aquatic",  #: Boolean
            "predator",  #: Boolean
            "toothed",  #: Boolean
            "backbone",  #: Boolean
            "breathes",  #: Boolean
            "venomous",  #: Boolean
            "fins",  #: Boolean
            "legs",  #: Numeric (set of values: {0,2,4,5,6,8})
            "tail",  #: Boolean
            "domestic",  #: Boolean
            "catsize",  #: Boolean
            "type",  #: Numeric (integer values in range [1,7])
        ]

        data = data[features_select]

        nominal_features = []
        ordinal_features = []

        X_data = data.drop(["type"], axis=1)
        y_data = pd.Series(
            OrdinalEncoder()
            .fit_transform(data["type"].values.reshape(-1, 1))
            .flatten(),
            name="number",
        )

    elif identifier == "MULT:Car":
        feature_names = [
            "buying",  #       v-high, high, med, low
            "maint",  #        v-high, high, med, low
            "doors",  #        2, 3, 4, 5-more
            "persons",  #      2, 4, more
            "lug_boot",  #     small, med, big
            "safety",  #       low, med, high
            "class",  #        unacc, acc, good, v-good
        ]

        data = pd.read_csv(
            "./datasets/Car/car.data", names=feature_names, index_col=False
        )

        features_select = [
            "buying",  #       v-high, high, med, low
            "maint",  #        v-high, high, med, low
            "doors",  #        2, 3, 4, 5-more
            "persons",  #      2, 4, more
            "lug_boot",  #     small, med, big
            "safety",  #       low, med, high
            "class",  #        unacc, acc, good, v-good
        ]

        data = data[features_select]

        nominal_features = []

        ordinal_features = [
            "buying",  #       v-high, high, med, low
            "maint",  #        v-high, high, med, low
            "doors",  #        2, 3, 4, 5-more
            "persons",  #      2, 4, more
            "lug_boot",  #     small, med, big
            "safety",  #       low, med, high
        ]

        X_data = data.drop(["class"], axis=1)
        y_data = pd.Series(
            OrdinalEncoder()
            .fit_transform(data["class"].values.reshape(-1, 1))
            .flatten(),
            name="class",
        )  # ((data['class'] != 'unacc') * 1)

    else:

        raise SystemExit("Unknown key: " + str(identifier))

    ((X_train, y_train), (X_valid, y_valid), (X_test, y_test), normalizer_list) = (
        preprocess_data(
            X_data,
            y_data,
            nominal_features,
            ordinal_features,
            config,
            random_seed=random_seed,
            verbosity=verbosity,
        )
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_valid": X_valid,
        "y_valid": y_valid,
        "X_test": X_test,
        "y_test": y_test,
        "normalizer_list": normalizer_list,
    }
