from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer, median_absolute_error
import re
import numpy as np
import pandas as pd
import statsmodels.api as sm
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import KFold
from skopt.space import Real, Categorical, Integer
from skopt import BayesSearchCV
from statsmodels.stats.outliers_influence import variance_inflation_factor


def get_num_rooms(num_rooms):
    """
    Function to encode the number of rooms as a float

    :param num_rooms: str
    :return: float
    """
    if num_rooms[-1] == '½':
        return float(num_rooms[:-1] + '.5')
    else:
        return float(num_rooms)


def process_records(property_records):
    """
    Function to fill in missing values with the mean, median or mode of the existing values of a feature.

    :param property_records: pd.DataFrame - the property records DataFrame containing the scraped data
    :return: pd.DataFrame
    """

    property_records['property_postcode'] = property_records['property_address'].apply(
        lambda s: re.findall(r'[0-9]{4}', s)[0])
    property_records.loc[property_records['floor'].isna() == False, 'floor'] = property_records['floor'].apply(
        lambda x: str(x))
    property_records.loc[property_records['property_type'].isna() == False, 'property_type'] = property_records[
        'property_type'].apply(lambda x: str(x))
    property_records.loc[property_records['floor'].isna() == False, 'floor'] = property_records['floor'].apply(
        lambda x: str(x))
    property_records.loc[property_records['property_type'].isna() == False, 'property_type'] = property_records[
        'property_type'].apply(lambda x: str(x))
    property_records.loc[property_records['rooms'].isna() == True, 'rooms'] = property_records[
        'rooms'].dropna().median()
    if 'gross_rent' in property_records.columns:
        property_records.loc[property_records['gross_rent'].isna() == True, 'gross_rent'] = property_records[
        'gross_rent'].dropna().mean()
    else:
        property_records.loc[property_records['property_price'].isna() == True, 'property_price'] = property_records[
            'property_price'].dropna().mean()
    property_records.loc[property_records['living_space'].isna() == True, 'living_space'] = property_records[
        'living_space'].dropna().mean()
    property_records.loc[property_records['property_postcode'].isna() == True, 'property_postcode'] = property_records[
        'property_postcode'].dropna().mode()
    property_records.loc[property_records['floor'].isna() == True, 'floor'] = property_records[
        'floor'].dropna().mode().values.astype(str)[0]
    property_records.loc[property_records['property_type'].isna() == True, 'property_type'] = property_records[
        'property_type'].dropna().mode().values.astype(str)
    property_records.loc[property_records['public_transport'].isna() == True, 'public_transport'] = property_records[
        'public_transport'].dropna().mean()
    property_records.loc[property_records['motorway'].isna() == True, 'motorway'] = property_records[
        'motorway'].dropna().mean()
    property_records.loc[property_records['shop'].isna() == True, 'shop'] = property_records['shop'].dropna().mean()

    encoder = OneHotEncoder().fit(property_records[['property_postcode', 'floor', 'property_type']])

    # save encoder as pickle file
    if 'gross_rent' in property_records.columns:
        with open('data/encoder.pickle', 'wb') as handle:
            pickle.dump(encoder, handle)
    else:
        with open('data/encoder_purchase.pickle', 'wb') as handle:
            pickle.dump(encoder, handle)

    encoding = pd.DataFrame(
        encoder.transform(property_records[['property_postcode', 'floor', 'property_type']]).toarray(),
        columns=[item for sublist in encoder.categories_ for item in sublist])

    property_records = pd.concat([property_records, encoding], axis=1).drop(columns=['Unnamed: 0'])

    return property_records


def get_outliers(x, y):
    """
    Function to find the outliers of the residuals of a model, using Tukey's test.

    :param residuals: pd.DataFrame, pd.DataFrame
    :return: [int] - the list of indices of the outliers in the pandas DataFrames
    """

    # add the intercept to the model
    x2 = sm.add_constant(x)

    # train the model
    estimator = sm.OLS(y, x2)
    model = estimator.fit()

    residuals = model.resid

    # calculate the upper and lower quartiles of the residuals
    q25 = residuals.quantile(.25)
    q75 = residuals.quantile(.75)

    # calculate the interquartile range of the residuals
    iqr = q75 - q25

    # calculate the upper and lower values of the interval
    lower = q25 - (1.5 * iqr)
    upper = q75 + (1.5 * iqr)

    # get the residuals that fall outside the interval (above/below the upper/lower values)
    outliers1 = residuals.loc[(residuals > upper)]
    outliers2 = residuals.loc[(residuals < lower)]
    outliers = pd.concat([outliers1, outliers2])

    return outliers.index.values


def remove_outliers_tukeys_test(x, y):
    """
    Function to remove outliers from the data, using Tukey's test.

    :param x: pd.DataFrame, pd.DataFrame
    :return: pd.DataFrame, pd.DataFrame
    """

    # get the indices of the outlier data points using Tukey's test
    outlier_indices = get_outliers(x, y)

    return x.drop(x.index[outlier_indices]).reset_index(drop=True), y.drop(y.index[outlier_indices]).reset_index(
        drop=True)


def get_vifs(x):
    """
    Function to find the Variance Inflation Factors (VIFs) of a list of independent variables/parameters, for the
    purpose of evaluating multicollinearity in the independent variables.

    :param x: pd.DataFrame - the features to be used in the model
    :return: [(str, float)] - a list of features and their corresponding VIFs
    """

    # calculate the VIFs
    vif = [variance_inflation_factor(x.values, i) for i in range(len(x.columns))]

    # zip the VIFs with their corresponding parameter
    vifs = list(zip(x.columns, vif))

    return sorted(vifs, key=lambda v: v[1], reverse=True)


def get_outliers_isolation_forest(x, y, n_estimators=100, contamination=0.06):
    """
    Function to get the outliers in the data, using an isolation forest.

    :param x: pd.DataFrame, pd.DataFrame, int, float
    :return: [int] - the list of indices of the outliers in the pandas DataFrames
    """

    isolation_forest = IsolationForest(n_estimators=n_estimators, contamination=contamination).fit_predict(
        pd.concat([y, x], axis=1))
    outlier_indices = [i for i in range(0, len(isolation_forest), 1) if isolation_forest[i] == -1]

    return outlier_indices


def remove_outliers_isolation_forest(x, y, n_estimators=100, contamination=0.06):
    """
    Function to remove outliers from the data, using an isolation forest.

    :param x: pd.DataFrame, pd.DataFrame, int, float
    :return: pd.DataFrame, pd.DataFrame
    """

    outlier_indices = get_outliers_isolation_forest(x, y, n_estimators, contamination)

    return x.drop(x.index[outlier_indices]).reset_index(drop=True), y.drop(y.index[outlier_indices]).reset_index(
        drop=True)


def optimize(x, y, model_type, kf):
    """
    Function that implements Bayesian hyperparameter optimization and k-fold cross-validation for a model.

    :param x: pd.DataFrame, pd.DataFrame, str, scikit-learn KFold object
    :return: model object (untrained)
    """

    # select the hyperparameters and their corresponding prior distributions according to the model type.
    if model_type[0] in ['XGBRFRegressor', 'RandomForestRegressor', 'ExtraTreesRegressor', 'GradientBoostingRegressor']:
        hyperparameter_list = {'n_estimators': Integer(1, 50, prior='uniform'),
                               'max_depth': Integer(1, 20, prior='uniform')}
    elif model_type[0] == 'AdaBoostRegressor':
        hyperparameter_list = {'n_estimators': Integer(1, 50, prior='uniform')}
    elif model_type[0] == 'DecisionTreeRegressor':
        hyperparameter_list = {'max_depth': Integer(1, 20, prior='uniform')}
    elif model_type[0] in ['Lasso', 'Ridge', 'LassoLars', 'ElasticNet']:
        hyperparameter_list = {'alpha': Real(0, 1, prior='uniform')}

    base_estimator = {'base_estimator': 'RF'}
    scoring_function = make_scorer(median_absolute_error, greater_is_better=False)

    opt = BayesSearchCV(estimator=model_type[1], search_spaces=hyperparameter_list, optimizer_kwargs=base_estimator,
                        n_iter=40, n_jobs=-1, scoring=scoring_function, cv=kf)

    # executes bayesian optimization
    _ = opt.fit(x, y)

    return opt.best_estimator_


def train_model(x, y, model_types, n_splits=3):
    """
    Function that implements Bayesian hyperparameter optimization and k-fold cross-validation for a list of models,
    and returns one optimal model configuration per model type.

    :param x: pd.DataFrame, pd.DataFrame, str, int
    :return: [[str, model object (untrained), float, float, float]]
    """

    kf = KFold(n_splits=n_splits, shuffle=True)

    model_results = []

    for model_type in model_types:
        rmse_result = []
        mae_result = []
        mme_result = []

        # get the version of the model with optimal hyperparameters
        if model_type[0] not in ['LinearRegression', 'LassoLarsCV', 'Lars']:
            model = optimize(x, y, model_type, kf)
        else:
            model = model_type[1]

        for train_index, test_index in kf.split(x):
            x_train, x_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # train the model on the selected data
            model.fit(x_train, y_train)

            y_pred = model.predict(x_test)

            # RMSE penalizies large prediction errors
            rmse_result.append(mean_squared_error(y_test, y_pred, squared=False))

            # MAE measures the absolute average distance between the real data and the predicted data
            mae_result.append(mean_absolute_error(y_test, y_pred))

            # MME measures the absolute median distance between the real data and the predicted data
            mme_result.append(median_absolute_error(y_test, y_pred))

        model_results.append([model_type[0], model, np.mean(rmse_result), np.mean(mae_result), np.mean(mme_result)])

    return model_results


def encode_input(living_space, rooms, postcode, floor, property_type, public_transport, motorway, shop, scaler,
                 encoder):
    """
    Encode a single instance of input data for gross rent prediction.

    :param x: float, float, str, str, str, float, float, float, scaler model, encoder model
    :return: pd.DataFrame - the processed pandas DataFrame containing the property information
    """

    information_columns = ['living_space', 'rooms', 'public_transport', 'motorway', 'shop', 'property_postcode',
                           'floor', 'property_type']

    actual = pd.DataFrame([[living_space, rooms, public_transport, motorway, shop, postcode, floor, property_type]],
                          columns=information_columns)

    scaled_names = ['scaled_living_space', 'scaled_rooms', 'scaled_public_transport', 'scaled_motorway', 'scaled_shop']

    encoded = pd.DataFrame(encoder.transform([[postcode, floor, property_type]]).toarray(),
                           columns=[item for sublist in encoder.categories_ for item in sublist])

    scaled = pd.DataFrame(scaler.transform([[living_space, rooms, public_transport, motorway, shop]]),
                          columns=scaled_names)

    return pd.concat([actual, encoded, scaled], axis=1)
