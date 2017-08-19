import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn import preprocessing

matplotlib.style.use('ggplot')

def transform_labels_to_numeric(data):
    for y in data.columns:
        if (data[y].dtype == "object"):
            data[y] = pd.to_numeric(data[y])
    return(data)

def do_pca(data, svd_solver):
    pca = PCA(n_components=2, svd_solver=svd_solver)
    pca.fit(data)
    T = pca.transform(data)
    return(T)

def scale_features_df(x):
    scaled = preprocessing.StandardScaler().fit_transform(x)
    scaled = pd.DataFrame(scaled, columns=x.columns)
    print("New Variances:\n", scaled.var())
    print("Ne)w Describe:\n", scaled.describe())
    return(scaled)

def plot_xy(data, x_axis, y_axis, title):
    data.plot.scatter(x=x_axis, y=y_axis, marker='o', c=labels, alpha=0.75)
    plt.title(title)
    plt.show()

def plot_xyz(x_axis, y_axis, z_axis):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.scatter(x_axis, y_axis, z_axis, c=labels, marker='.', alpha=0.75)
    plt.title("3D visualization")
    plt.show()

def transform_into_DF(data):
    data = pd.DataFrame(data)
    data.columns = ['component1', 'component2']
    return(data)

#upload and preparation of data
df = pd.read_csv("dataset\kidney_disease.csv")
df = df.dropna(axis =0)
labels = ['red' if i == 'ckd' else 'green' for i in df.classification]
df_blood = df.loc[:, ["bgr", "wc", "rc"]]
df_blood = df_blood.reset_index(drop=True)
df_blood = transform_labels_to_numeric(df_blood)
print(df_blood.describe())

#3D plot
plot_xyz(df_blood.bgr, df_blood.wc, df_blood.rc)

#PCA(full)
df_blood_pca = do_pca(df_blood, "full")
df_blood_pca = transform_into_DF(df_blood_pca)
plot_xy(df_blood_pca, "component1", "component2", "Data after PCA (without scaling features)")

#PCA(full) when  features of data frame are scale
df_blood_scale = scale_features_df(df_blood)
df_blood_pca_scale = do_pca(df_blood_scale, "full")
df_blood_pca_scale = transform_into_DF(df_blood_pca_scale)
plot_xy(df_blood_pca_scale, "component1", "component2", "Data after PCA (with scaling features)")