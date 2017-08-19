import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing

matplotlib.style.use('ggplot')

def drop_labels(data, list):
    data = data.drop(labels = list, axis = 1)
    data = data.reset_index(drop=True)
    return(data)

def transform_labels_type(data):
    for y in data.columns:
        if (data[y].dtype == "object"):
            try:
                data[y] = pd.to_numeric(data[y])
            except ValueError:
                data[y] = pd.get_dummies(data[y])    #Convert categorical variable into dummy/indicator variables
    return(data)

def scale_features_df(data):
    scaled = preprocessing.StandardScaler().fit_transform(data)
    scaled = pd.DataFrame(scaled, columns=data.columns)
    print("New Variances:\n", scaled.var())
    print("New Describe:\n", scaled.describe())
    return(scaled)

def do_pca(data, svd_solver):
    pca = PCA(n_components=2, svd_solver=svd_solver)
    pca.fit(data)
    T = pca.transform(data)
    return(T)

def transform_into_DF(data):
    data = pd.DataFrame(data)
    data.columns = ['component1', 'component2']
    return(data)

def plot_xy(data, x_axis, y_axis, title):
    data.plot.scatter(x=x_axis, y=y_axis, marker='o', c=labels, alpha=0.75)
    plt.title(title)
    plt.show()

#upload and preparation of data
df = pd.read_csv("dataset\kidney_disease.csv")
df = df.dropna(axis=0)
labels = ['red' if i == 'ckd' else 'green' for i in df.classification]
#data frame with only numeric features
df_num = drop_labels(df, ['id', 'classification', 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'])
#data frame with numeric and  nominal features
df_nom = drop_labels(df, ['id', 'classification'])
df_num = transform_labels_type(df_num)
df_nom = transform_labels_type(df_nom)


for scaleFeatures in [False, True]:
    title = "Data after PCA (was only numeric features)"
    if scaleFeatures:
        df_num = scale_features_df(df_num)
        title = "\n".join([title, " With scaling features"])

    df_num_pca = do_pca(df_num,svd_solver="full")
    df_num_pca = transform_into_DF(df_num_pca)
    plot_xy(df_num_pca, "component1", "component2", title)

for scaleFeatures in [False, True]:
    title = "Data after PCA (was numeric and nominal features)"
    if scaleFeatures:
        df_nom = scale_features_df(df_nom)
        title = "\n".join([title, " With scaling features"])

    df_nom_pca = do_pca(df_nom, svd_solver="full")
    df_nom_pca = transform_into_DF(df_nom_pca)
    plot_xy(df_nom_pca, "component1", "component2", title)

