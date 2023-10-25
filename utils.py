import numpy as np
import scipy as sp
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize
from sklearn.metrics import mean_squared_error as MSE, mean_absolute_error as MAE
RMSE = lambda y_true, y_pred: MSE(y_true, y_pred, squared=False)
from sklearn.linear_model import LinearRegression, TheilSenRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import re
from tqdm import tqdm
import os
import sys
import shelve
#import plotly.graph_objects as go
#import plotly.express as px
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap

### -------- MISCELLANEOUS -------- ###
def parse_filename(fn):
    splits = re.split('_|\.',fn)
    efc = splits[0].split('efc')[-1]
    dc = splits[1].split('dc')[-1]
    return int(efc), dc



def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)





### -------- VISUALIZATION -------- ###
def plot_3D_mpl(df, clustering=None):

    # axes instance
    fig = plt.figure(figsize=(6,6))
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)

    # get colormap from seaborn
    if clustering is not None:
        n_colors = len(set(clustering))
    else:
        n_colors = 1
    cmap = ListedColormap(sns.color_palette("husl", n_colors).as_hex())

    # plot
    if clustering is not None:
        sc = ax.scatter(df['Voltage'], df['Current'], df['SOC'], s=0.1, marker='o', c=clustering, cmap=cmap, alpha=1)
    else:
        sc = ax.scatter(df['Voltage'], df['Current'], df['SOC'], s=0.1, marker='o', cmap=cmap, alpha=1)
    ax.set_xlabel('V')
    ax.set_ylabel('I')
    ax.set_zlabel('SOC')

    # legend
    plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)

    fig.show()
    
    

def plot_3D_plotly(df, clustering=None):
    return



# def plot_time_window(df, X_idxs, i, X_coeffs=None):
#     idx = np.arange(X_idxs[i,0],X_idxs[i,1])
#     fig = go.Figure(data=[go.Scatter3d(x=df.iloc[idx].loc[:,'Voltage'],
#                                        y=df.iloc[idx].loc[:,'Current'],
#                                        z=df.iloc[idx].loc[:,'SOC'],
#                                        mode='markers',
#                                        marker=dict(size=2))])
#     """
#     # plot interpolating plane
#     # z = ax+by+c
#     a = X_coeffs[i,0]
#     b = X_coeffs[i,1]
#     c = X_coeffs[i,2]
#     print(a)
#     print(b)
#     print(c)
#     x, y = np.mgrid[-1:0.7:0.001, -2:7]
#     z = a*x+b*y+c
#     fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale=[[0, 'rgb(255,0,0)'], [1, 'rgb(170,0,0)']]))
#     """
#     fig.show()

    

def plot_feature(df, features=None, since=None, until=None):
    # keep only selected time window
    if since is None:
        since = df['Timestamp'].iloc[0]
    if until is None:
        until = df['Timestamp'].iloc[-1]
    df = df[(df['Timestamp'] >= pd.Timestamp(since)) & (df['Timestamp'] <= pd.Timestamp(until))]

    if features == None:
        features = df.value_counts('Signal Type').index    # unique features
        for i,f in enumerate(features):
            fig, ax = plt.subplots(figsize=(25,5))
            sns.lineplot(data=df[df['Signal Type'] == f], x='Timestamp', y='Value', markers=True)
            ax.grid(True, axis='both')
            ax.set_xlabel('Timestamp')
            ax.set_ylabel(f)
            if len(ax.get_yticks()) > 20:
                yloc = plt.MaxNLocator(20)
                ax.yaxis.set_major_locator(yloc)

    elif type(features) == str:
        fig, ax = plt.subplots(figsize=(25,5))
        sns.lineplot(data=df[df['Signal Type'] == features], x='Timestamp', y='Value', markers=True)
        ax.set_xlabel('Timestamp')
        ax.set_ylabel(features)
        if len(ax.get_yticks()) > 20:
            yloc = plt.MaxNLocator(20)
            ax.yaxis.set_major_locator(yloc)

    else:
        for i,f in enumerate(features):
            fig, ax = plt.subplots(figsize=(25,5))
            sns.lineplot(data=df[df['Signal Type'] == f], x='Timestamp', y='Value', markers=True)
            ax.grid(True, axis='both')
            ax.set_xlabel('Timestamp')
            ax.set_ylabel(f)
            if len(ax.get_yticks()) > 20:
                yloc = plt.MaxNLocator(20)
                ax.yaxis.set_major_locator(yloc)
    return



def display_scree_plot(pca):
    '''Display a scree plot for the pca'''
    expl_var_percentage = pca.explained_variance_ratio_*100   # y1
    cum_expl_var_percentage = expl_var_percentage.cumsum()    # y2
    x = np.arange(len(expl_var_percentage))+1
    fig, ax = plt.subplots(figsize=(16,6))
    ax.bar(x, expl_var_percentage)
    ax.plot(x, cum_expl_var_percentage, c="red", marker='o')
    ax.set_xticks(x)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlabel("N. of principal components", fontsize=20)
    ax.set_ylabel("% of explained variance", fontsize=20)
    for i, txt in enumerate(cum_expl_var_percentage):
        ax.annotate(f'{txt:.1f}', (x[i], cum_expl_var_percentage[i]+6), horizontalalignment='center')
    ax.set_ylim(0,119)
    print(f'Explained variance with {pca.n_components_} principal components: {cum_expl_var_percentage[-1]:.2f}')





### -------- PREPROCESSING -------- ###
def extract_time_windows(df, length=300, slide=None, freq=5, max_SOC=100, min_SOC=0, random_starting_point=True, verbose=False, overwrite=False, basepath='.', suffix='', random_state=None):
    # produce time windows' indices from the dataset df
    X = []
    y = []
    
    if not (isinstance(length,np.ndarray) or isinstance(length,list)):
        length = [length]
        
    if random_state:
        np.random.seed(random_state)
    
    # reindex the last df index level as increasing integers
    # (so that .loc of the relative temp_df's are the .iloc of the entire df)
    df = df.reset_index()
    df['Point'] = np.arange(df.shape[0])
    df = df.set_index(['SOH','Drive cycle','Point'])

    # cycle through each possible (EFC level, drive cycle) couple
    for (soh, dc), temp_df in tqdm(df.groupby(level=[0, 1]), disable= not verbose):
        temp_df = temp_df[['Voltage','Current','SOC']]

        # compute the minimum and maximum timestamps between which the time window lives in the desired SOC range
        upper_cut = temp_df['SOC'].le(max_SOC).argmax()
        lower_cut = temp_df['SOC'].ge(min_SOC).cumsum().max()
       
        # cycle through each possible length of the time window
        for dt in length:
            
            # set starting point of the first time window
            if random_starting_point:
                # choose at random a starting point for the time window (between 0s and 60s by default)
                starting_point = np.random.randint(0, (len(length)/freq)*freq)
            else:
                # set starting point to 0
                starting_point = 0

            # while the ending point (for the current time window) is within the current dataframe
            while True:

                # compute the ending point (for the current time window)
                ending_point = starting_point + dt*freq

                # NB: we want all the extracted time windows to have the same length dt
                if ending_point > temp_df.shape[0]:
                    break
                else:
                    # the relative indices of the current time window (relative to the subdataframe defined by (soh,dc))
                    # are: [starting_point:ending_point]
                    if starting_point >= upper_cut and ending_point <= lower_cut:   # then, keep the time window
                        idx = (temp_df.iloc[starting_point].name[2], temp_df.iloc[ending_point].name[2])
                        X.append(idx)
                        y.append(soh)
                    
                    if slide:
                        # the starting point of the next time window is <slide> s after the starting point of the current one
                        starting_point = starting_point + slide*freq
                    else:
                        # the ending point of the current time window becomes the starting point of the next one
                        starting_point = ending_point
    
    X, y = np.array(X), np.array(y)
    
    if overwrite:
        pd.DataFrame(X,
                     index=[f'{i}' for i in range(X.shape[0])],
                     columns=[f'{i}' for i in range(X.shape[1])]
                    ).to_parquet(f'{basepath}/X{suffix}_idxs.parquet')
        pd.DataFrame(y,
                     index=[f'{i}' for i in range(len(y))],
                     columns=['0']
                    ).to_parquet(f'{basepath}/y{suffix}.parquet')
    return X, y



def time_window_to_plane(df, idxs, solver='ols', random_state=None, verbose=False, overwrite=False, basepath='.', suffix=''):
    """
    Find the hyperplane interpolating the given time window tw, by means of linear regression with ordinary least squares
    Inputs:
        df (pd dataframe containing the points)
        idxs (tuple of two elements, the starting and ending point of the time window)
        solver (string, the solver for the lin. regr. problem)
    Outputs:
        plane (as an array of 3 coefficients identifying that plane)
    """
    
    
    if len(idxs.shape) == 1:    # in this case, idxs is a tuple of two elements, the starting and the ending point of the time window
        if solver=='theil':
            reg = TheilSenRegressor(n_jobs=-1, random_state=random_state)
        else: #'ols'
            reg = LinearRegression(n_jobs=-1)
        reg.fit(df.iloc[idxs[0]:idxs[1]].loc[:,['Voltage','Current']], df.iloc[np.arange(idxs[0],idxs[1])].loc[:,'SOC'])
        coeffs = np.array([*reg.coef_, reg.intercept_])

    else:   # in this case, idxs is a matrix whose i-th row is a tuple of two elements, the starting and the ending point of i-th time window of the dataset
        coeffs = []
        for start, end in tqdm(idxs, disable=not verbose):
            if solver=='theil':
                reg = TheilSenRegressor(n_jobs=-1, random_state=random_state)
            else: #'ols'
                reg = LinearRegression(n_jobs=-1)
            reg.fit(df.iloc[start:end].loc[:,['Voltage','Current']], df.iloc[start:end].loc[:,'SOC'])
            coeffs.append( [*reg.coef_, reg.intercept_] )
        coeffs = np.array(coeffs)
        
        if overwrite:
            pd.DataFrame(coeffs,
                         index=[f'{i}' for i in range(coeffs.shape[0])],
                         columns=[f'{i}' for i in range(coeffs.shape[1])]
                        ).to_parquet(f'{basepath}/X{suffix}_coeffs_{solver}.parquet')
    
    return coeffs



def ds_to_efc_planes(df, solver='ols', verbose=True):
    """
    The aim of this function is to get an interpolating plane for each efc level (so 21 planes in total)
    But is this function really needed?
    """

    coeffs = []
    # cycle through each possible (EFC level, drive cycle) couple
    for efc, temp_df in tqdm(df.groupby(level=[0]), disable= not verbose):
        temp_df = temp_df[['Voltage','Current','SOC']]
        if solver=='theil':
            reg = TheilSenRegressor(n_jobs=-1, random_state=0)
        else: #'ols'
            reg = LinearRegression(n_jobs=-1)
        reg.fit(temp_df.loc[:,['Voltage','Current']], temp_df.loc[:,'SOC'])
        coeffs.append([*reg.coef_, reg.intercept_])

    return coeffs



def filter_tw(df, X_test, X_idxs_test, max_SOC=100, min_SOC=0, max_length=np.inf, min_length=0, freq=5):
    # find indices of X_test associated to time windows inside the specified SOC range and length range
    idx_selected = []
    for i,(start,end) in tqdm(enumerate(X_idxs_test)):
        temp_df = df['SOC'].iloc[start:end]

        # check SOC range
        upper_cut = temp_df.le(max_SOC).argmax()
        lower_cut = temp_df.ge(min_SOC).cumsum().max()
        if upper_cut != 0 or lower_cut != temp_df.shape[0]:
            continue    # do not keep
        
        # check length
        length = end-start
        if length < min_length*freq or length > max_length*freq:
            continue    # do not keep
        
        # if the two checks are passed, finally record the relative index of the current tw
        idx_selected.append(i)

    return idx_selected



###-------- AVILOO PREPROCESSING --------###
def aviloo_preprocessing(df, features=None):
    # string to datetime (=timestamps)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H:%M:%S')

    # drop 'Timestamp (Unix Microseconds)' column
    df = df.drop(columns='Timestamp (Unix Microseconds)')
    
    # if features is not None, keep only the selected features
    if features:
        df = df.loc[df['Signal Type'].isin(features)]
    
    # exclude non-numeric features
    for s in df['Signal Type'].unique():
        try:
            df[df['Signal Type'] == s]['Value'].astype('float')
        except ValueError:
            df = df.drop(df[df['Signal Type'] == s].index)
    
    # cast 'Value' column to float (in case non-numeric features were present)
    df['Value'] = df['Value'].astype('float')

    # add 2 rows for each signal type:
    # - one for df['Timestamp'].min()
    # - one for df['Timestamp'].max()
    # this is because sometimes T_OUTSIDE and SPEED are sampled much later than
    # df['Timestamp'].min() and much earlier than df['Timestamp'].max()

    min_ts = df['Timestamp'].min()
    max_ts = df['Timestamp'].max()

    for s in df['Signal Type'].unique():
        # compute the first and last value by replicating the first or last value in the dataset
        prepend_value = df[df['Signal Type'] == s]['Value'].iloc[0]
        prepend_row = pd.DataFrame([[min_ts, s, prepend_value]], columns=df.columns)
        append_value = df[df['Signal Type'] == s]['Value'].iloc[-1]
        append_row = pd.DataFrame([[max_ts, s, append_value]], columns=df.columns)
        df = pd.concat([prepend_row, df, append_row], ignore_index=True)
        
    return df



def extract_feature(df, feature, sampling_time=None, to_excel=True, file_name='extracted.xlsx'):
    # filter only the signal associated to the specified feature
    feature_df = df[df['Signal Type'] == feature]
    # for each measurement (timestamp, feature), compute elapsed total seconds from its timestamp
    feature_df['Elapsed'] = feature_df['Timestamp'] - feature_df['Timestamp'].iloc[0]    # subtract the initial timestamp to all rows
    feature_df['Elapsed'] = feature_df['Elapsed'].dt.total_seconds()

    if sampling_time:
        # if a sampling_time is specified, perform linear interpolation and resample the measurements every sampling_time seconds

        # for each couple (timestamp, measurement), compute elapsed total seconds from its timestamp
        total_elapsed = feature_df.iloc[-1].loc['Elapsed']     # total elapsed time
        elapsed_resampled = np.arange(0, total_elapsed, sampling_time)
        value_resampled = np.interp(elapsed_resampled, feature_df['Elapsed'], feature_df['Value'])
        feature_df = pd.DataFrame({'Elapsed':elapsed_resampled,'Value':value_resampled})

    if to_excel:
        feature_df[['Elapsed','Value']].to_excel(file_name, header=False, index=False)
    else:
        return feature_df[['Elapsed','Value']]



def extract_feature(df, feature, sampling_time=None, to_excel=True, file_name='extracted.xlsx'):
    # filter only the signal associated to the specified feature
    feature_df = df[df['Signal Type'] == feature]
    # for each measurement (timestamp, feature), compute elapsed total seconds from its timestamp
    feature_df['Elapsed'] = feature_df['Timestamp'] - feature_df['Timestamp'].iloc[0]    # subtract the initial timestamp to all rows
    feature_df['Elapsed'] = feature_df['Elapsed'].dt.total_seconds()

    if sampling_time:
        # if a sampling_time is specified, perform linear interpolation and resample the measurements every sampling_time seconds

        # for each couple (timestamp, measurement), compute elapsed total seconds from its timestamp
        total_elapsed = feature_df.iloc[-1].loc['Elapsed']     # total elapsed time
        elapsed_resampled = np.arange(0, total_elapsed, sampling_time)
        value_resampled = np.interp(elapsed_resampled, feature_df['Elapsed'], feature_df['Value'])
        feature_df = pd.DataFrame({'Elapsed':elapsed_resampled,'Value':value_resampled})

    if to_excel:
        feature_df[['Elapsed','Value']].to_excel(file_name, header=False, index=False)
    else:
        return feature_df[['Elapsed','Value']]

    
    
def filter_features_and_put_together(df, features, sampling_time=0.2):
    assert(isinstance(features, list))
    # if there is no measurements for one of the specified features, then exclude it
    features = [f for f in features if not df[df['Signal Type'] == f].empty]
    feature_dfs = []
    new_col_names = {'CURRENT':'Current',
                     'VOLTAGE':'Voltage',
                     'SOC_REAL':'SOC',
                     'SPEED':'Speed',
                     'T_CELL_AVG': 'Temperature'}
    for f in features:
        feature_df = extract_feature(df, f, sampling_time=sampling_time, to_excel=False)
        feature_df = feature_df.set_index('Elapsed')
        feature_df = feature_df.rename(columns={'Value':new_col_names.get(f,f)})
        feature_dfs.append(feature_df)
    return pd.concat(feature_dfs, axis=1).reset_index(drop=True)