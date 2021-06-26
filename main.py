'''
Time:   Created on June 24th, 2021
Author: Wang Haoran
Environment Requirement:
        |---------------------------|
        |pandas         1.2.4       |
        |---------------------------|
        |numpy          1.20.1      |
        |---------------------------|
        |scikit-learn    0.24.1     |
        |---------------------------|
Last-Modified-On:

'''

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.cluster import SpectralClustering
from sklearn.svm import SVR

# Change the PATH to the absolute path of the project in your PC.
PROJECT_PATH = ''

categories_columns_name_list = [
    'class_number_skewness_level'
    'bandwidth_skewness_level'
    'duration_time_skewness_level'
]

# class number skewness classification
# !!! Note: 0.5, 1.0 and 1.5 are NOT FINAL parameters.
def class_number_skewness_map(x):
    if x < 0.5:
        class_number_skewness_index = 0
    elif x < 1.0:
        class_number_skewness_index = 1
    elif x < 1.5:
        class_number_skewness_index = 2
    else:
        class_number_skewness_index = 3
    return class_number_skewness_index

# bandwidth skewness classification
# !!! Note: 0.5, 1.0 and 1.5 are NOT FINAL parameters.
def bandwidth_skewness_map(x):
    if x < 0.5:
        bandwidth_skewness_index = 0
    elif x < 1.0:
        bandwidth_skewness_index = 1
    elif x < 1.5:
        bandwidth_skewness_index = 2
    else:
        bandwidth_skewness_index = 3
    return bandwidth_skewness_index

# Duration time skewness classification
# !!! Note: 0.5, 1.0 and 1.5 are NOT FINAL parameters.
def duration_time_skewness_map(x):
    if x < 0.5:
        duration_time_skewness_index = 0
    elif x < 1.0:
        duration_time_skewness_index = 1
    elif x < 1.5:
        duration_time_skewness_index = 2
    else:
        duration_time_skewness_index = 3
    return duration_time_skewness_index

def build_categories_columns(df):
    # map to metadata category level
    df["class_number_skewness_level"] = df["class_number"].map(class_number_skewness_map)
    df["bandwidth_skewness_level"] = df["bandwidth"].map(bandwidth_skewness_map)
    df["duration_time_skewness_level"] = df["duration_time"].map(duration_time_skewness_map)
    return df

def build_matrix_by_metadata(metadata_task_list):
    tasks_num = metadata_task_list.__len__()
    d_meta = np.zeros((tasks_num, tasks_num))

    # stage II: Jaccard Distance
    for i, metadata_1 in enumerate(metadata_task_list):
        for j, metadata_2 in enumerate(metadata_task_list):
            diff_count = 0
            for ii in range(len(categories_columns_name_list)):
                diff_count += (metadata_1[ii] != metadata_2[ii])
            j_d = 1 - (len(categories_columns_name_list) - diff_count) / (len(categories_columns_name_list) + diff_count)
            d_meta[i, j] = j_d

    return d_meta

def train_clp_model(df_train):
    y_label = 'class_predict'
    # clp stands for 'class predict'.
    clp_feature_columns = [
        'bandwidth',
        'package_size',
    ]
    # print('class predict feature used: ', clp_feature_columns)
    x_i = df_train.loc[:, clp_feature_columns].values
    y_i = df_train.loc[:, y_label].values
    # print("Actual training shape：x_i, y_i", x_i.shape, y_i.shape)

    model_clp = SVR()
    model_clp.fit(x_i, y_i)

    return model_clp

def train_and_eval_clp_model(df_train, df_test):
    y_label = 'class_predict'
    # clp stands for 'class predict'.
    clp_feature_columns = [
        'bandwidth',
        'package_size',
    ]
    # print('class predict feature used: ', clp_feature_columns)
    x_i = df_train.loc[:, clp_feature_columns].values
    y_i = df_train.loc[:, y_label].values
    # print("Actual training shape：x_i, y_i", x_i.shape, y_i.shape)

    # 1.2 train
    model_clp = SVR()
    model_clp.fit(x_i, y_i)

    # 1.3 validation
    x_i_val = df_test.loc[:, clp_feature_columns].values
    y_i_val_infer = model_clp.predict(x_i_val)  # infer
    y_i_val_true = df_test.loc[:, y_label].values  # GT

    clp_err_vec = np.abs(y_i_val_infer - y_i_val_true)
    # print('cop loss:', np.mean((y_i_val_infer - y_i_val_true) ** 2))
    # print('cop loss:', cop_err)

    return clp_err_vec

if __name__ == '__main__':
    # 1. load csv data
    data = pd.read_csv('data/sample.csv')
    print(data.shape)
    #print('chillerName:', data['chillerName'].unique())

    data = data.loc[(data['cop'] >= 1) & (data['cop'] <= 5)].reset_index(drop=True)
    print('filter data out of range(1,5) , shape:', data.shape)


    #with open(PROJECT_PATH + '/retrain/save_data/estimate_capacity.pkl', 'rb') as r:
    #    estimate_capacity = pickle.load(r)
    #    data['plr'] = data.apply(cal_sample_plr_for_df, axis=1)

    # 1.2 create category column
    data = build_categories_columns(df=data)
    print('Done 1. load csv data')

    # 2. build data into tasks
    task_dict = {}
    metadata_task_list = []
    for metadata, df_small in data.groupby(categories_columns_name_list):
        task_dict.update({metadata: df_small})
        metadata_task_list.append(metadata)
    print('Done 2. build data into %s tasks' % metadata_task_list.__len__())

    # 3. build matrix
    d_mate_matrix = build_matrix_by_metadata(metadata_task_list=metadata_task_list)
    print('Done 3. build matrix')

    # 4. meta-clustering modeling
    metadata_clustering_dict = {}
    n_clusters = 20
    clustering = SpectralClustering(n_clusters=n_clusters,
                                    assign_labels='discretize',
                                    random_state=0).fit(d_mate_matrix)
    labels = clustering.labels_
    for c in range(n_clusters):
        metadata_clustering_dict.update({c: np.where(labels == c)[0]})

    # 4.1 save model
    with open('save/metadata_task_list.pkl', 'wb') as w:
        pickle.dump(metadata_task_list, w)
    with open('save/metadata_clustering_dict.pkl', 'wb') as w:
        pickle.dump(metadata_clustering_dict, w)

    # 4.2 train model for all clustering
    metadata_clustering_models_dict = {}
    for cluster_id, metadata_id_list in metadata_clustering_dict.items():
        df_this_cluster_list = []
        for metadata_id in metadata_id_list:
            the_metadata = metadata_task_list[metadata_id]
            df_this_cluster_list.append(task_dict[the_metadata])
        df_this_cluster = pd.concat(df_this_cluster_list, axis=0).reset_index(drop=True)
        cop_model_this_cluster = train_clp_model(df_train=df_this_cluster)
        metadata_clustering_models_dict.update({cluster_id: cop_model_this_cluster})
    # 4.3 save clustering models results
    with open('save/metadata_clustering_models_dict.pkl', 'wb') as w:
        pickle.dump(metadata_clustering_models_dict, w)
    print('Done 4. meta-clustering modeling')

    # 5. inference and evaluation
    total_err = 0
    sample_num = 0
    check_clustering_index = 0
    cluster = metadata_clustering_dict[check_clustering_index]
    print('this cluster:', cluster)
    for choose_i in range(cluster.shape[0]):
        df_to_infer = None
        df_train_from_this_cluster = []

        for i, metadata_index in enumerate(cluster):
            the_metadata_tuple = metadata_task_list[metadata_index]
            df_by_metadata = task_dict[the_metadata_tuple]

            if i == choose_i:
                df_to_infer = df_by_metadata.reset_index(drop=True)
            else:
                df_train_from_this_cluster.append(df_by_metadata)

        df_train = pd.concat(df_train_from_this_cluster, axis=0).reset_index(drop=True)

        cop_err_vector = train_and_eval_clp_model(df_train=df_train, df_test=df_to_infer)
        total_err += cop_err_vector.sum()
        sample_num += df_to_infer.shape[0]

    mean_err_this_cluster = total_err / sample_num
    print('error for cluster %s:' % check_clustering_index, mean_err_this_cluster)
    print('sample num:', sample_num)

