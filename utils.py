# classification

#libraries

import os
import pandas as pd
import geopandas as gpd
import numpy as np

import rasterio as rio
from itertools import product
from rasterio import windows
import matplotlib.pyplot as plt
from sklearn import preprocessing

from osgeo import gdal
from osgeo.gdalconst import GDT_Int16


#train-test
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GroupShuffleSplit


#ML
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, classification_report, cohen_kappa_score 
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN

#stats
import scipy.stats

#visuals
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots

plt.style.use(['science', 'no-latex']) 
sns.set_style("whitegrid", {'axes.grid' : False})
plt.rcParams.update({
    "font.family": "Palatino",   
    "font.serif": ["Times"],  
    "font.size":9,
    "axes.labelsize":9,
    "xtick.labelsize" : 9,
    "ytick.labelsize" : 9})   

colors =[
    '#117733', '#9e2a90', '#88ccee', '#ca7878', '#dbd73e', '#c7c7c9', '#60d52a','#418c84' #, '#92462d'
]

dict_normal_names = {1: 'trees', 
2: 'flooded vegetation',
3: 'open water',
4: 'settlements', 
5: 'bare soil',
6: 'agriculture and grass', 
7: 'shrubs',
#8: 'open rocks',
9:'sparse vegetation'
}

#model saving
from joblib import dump, load

#for getting model predictions
def get_predictions(data_scaled, 
                    model,
                    param_grid, 
                    target_column,
                    stratify_column,
                    split_rate: float=0.3,
                    smote_balance: bool=True,
                    cv: int=5, 
                    n_iter_search: int=3):

    labels = data_scaled[target_column] #get label data
    indices=np.arange(data_scaled.shape[0]) #get indices numpy

    while True:
        train_inds, test_inds = next(GroupShuffleSplit(test_size=split_rate, n_splits=2#,random_state = 40
                                                      ).split(data_scaled, groups=data_scaled[stratify_column]))
        train = data_scaled.iloc[train_inds]
        test = data_scaled.iloc[test_inds]
        if len(train[target_column].unique(
        )) == len(test[target_column].unique(
        )) == len(data_scaled[target_column].unique()): #because we need target feature to be represented in train and test
            break
    #class balansing with smote
    if smote_balance is True:
        smote = SMOTE(random_state = 42)
        X, y = smote.fit_resample(train.loc[:,  ~train.columns.isin([target_column, stratify_column])],
                                  train[target_column]) #drops 3 columns: key, class, and forest
        df_smote = pd.DataFrame(X, columns = train.loc[:,  ~train.columns.isin([target_column, stratify_column])].columns.tolist()) #drops 3 columns: key, class, and forest

        #we set train/test from SMOTE results
        X_train = df_smote
        y_train = y
        X_test = test.loc[:, ~test.columns.isin([target_column, stratify_column])]  
        y_test = test[target_column]
        #we set train/test as it is
    else:
        X_train = train.loc[:, ~train.columns.isin([target_column, stratify_column])]
        y_train = train[target_column]
        X_test = test.loc[:, ~test.columns.isin([target_column, stratify_column])]
        y_test = test[target_column]
    
    #parameters optimisation
    
    gs = RandomizedSearchCV(model, 
                            param_distributions = param_grid,
                            n_iter = n_iter_search, 
                            cv = cv, 
                            scoring= 'f1_weighted',
                            n_jobs = -1)
    gs.fit(X_train, y_train)  
    y_pred = gs.best_estimator_.predict(X_test)
    model_fit = gs.best_estimator_
    
    results = {'model': model_fit,
               'X_train data': X_train,
               'y train data':  y_train,
               'X test data': X_test,
               'y test data': y_test,
               'y predicted': y_pred
        
    }

    return results

def metrics_description(y_true, y_pred, 
                        metrics_by_class: bool=True, 
                        confusion_matrix_multiclass_on: bool=True,
                        binary_matrix_on: bool=False):

    
    print('Accuracy score: %.2f%%' %(accuracy_score(y_true, y_pred)*100))  
    print('Precision score: %.2f%%' % (precision_score(y_true, y_pred, average= 'weighted')*100))
    print('Recall score: %.2f%%' % (recall_score(y_true, y_pred, average= 'weighted')*100))
    print('F1-Score: %.2f%%'%(f1_score(y_true, y_pred, average = 'macro')*100))
    print('Kappa score: %.2f%%'%(cohen_kappa_score(y_true, y_pred)*100))
    
    
    #dataframe with metrics by class
    if metrics_by_class is True:
        metrics_by_class = pd.DataFrame(
                {
                    'names': list(map(dict_normal_names.get, list(np.unique(y_true)))),
                    'f1_scores': f1_score(y_true, y_pred,
                               average=None).round(2).tolist(),
                    'precision_list': precision_score(y_true, y_pred, 
                                       average=None).round(2).tolist(),
                    'recall':recall_score(y_true, y_pred,
                                       average=None).round(2).tolist()
                }
            )
        display(metrics_by_class)

    #confusion matrix multiclass
    if confusion_matrix_multiclass_on is True:
        data = confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(data, columns=list(map(dict_normal_names.get, list(np.unique(y_true)))), 
                             index = list(map(dict_normal_names.get, list(np.unique(y_true)))))
        df_cm.index.name = 'Actual'
        df_cm.columns.name = 'Predicted'

        #confusion matrix plot
        f, ax = plt.subplots(figsize=(6, 10))
        cmap = sns.cubehelix_palette(light=1, as_cmap=True)

        sns.heatmap(df_cm, cbar=False, annot=True, cmap=cmap, square=True, fmt='.0f',
                    annot_kws={'size': 10})
        plt.title('Actuals vs Predicted')
        plt.show()
        
    #confusion matrix binary    
    if binary_matrix_on is True:
        cm = confusion_matrix(y_true, y_pred)
        print('Confusion matrix\n\n', cm)
        ConfusionMatrixDisplay(confusion_matrix=cm).plot()

#getting dataset with metrics by class for each random prediction
def get_classes_metrics(models_vector): #vector with model variations, y predicted and y true from the dataset
    class_metrics_dataframe = pd.DataFrame()
    count = 0 #counter of iteration

    for i in models_vector:

        count += 1 #counting
        pred = i['y predicted'] #predicted values 
        true = i['y test data'] #corresponding labels from random test set
        names_list = list(np.unique(true))

        temp = pd.DataFrame(
            {
                'iteration':[count]*len(names_list), 
                'names': list(map(dict_normal_names.get, names_list)),
                'f1_scores': f1_score(true, pred,
                           average=None).round(2).tolist(),
                'precision_list': precision_score(true, 
                                   pred, 
                                   average=None).round(2).tolist(),
                'recall':recall_score(true, 
                                   pred, 
                                   average=None).round(2).tolist()
            }
        ) #dataset for each model 

        class_metrics_dataframe = pd.concat([class_metrics_dataframe, temp], ignore_index=True)
    return class_metrics_dataframe 


#getting dataset with average metrics for each random prediction
def get_metrics_average(models_vector): #vector with model variations, y predicted and y true from the dataset
    average_metrics_dataframe = pd.DataFrame()
    count = 0 #counter of iteration

    for i in models_vector:

        count += 1 #counting
        pred = i['y predicted'] #predicted values 
        true = i['y test data'] #corresponding labels from random test set

        temp = pd.DataFrame(
            {
                'iteration':[count],#*len(names_list), 
                #'names': list(map(dict_normal_names.get, names_list)),
                'f1_scores': f1_score(true, pred,
                           average='macro').round(2).tolist(),
                'precision_list': precision_score(true, 
                                   pred, 
                                   average='weighted').round(2).tolist(),
                'recall':recall_score(true, 
                                   pred, 
                                   average='weighted').round(2).tolist()
            }
        ) #dataset for each model 

        average_metrics_dataframe = pd.concat([average_metrics_dataframe, temp], 
                                              ignore_index=True)
    return average_metrics_dataframe 


def get_best_model(datavector_models):
    number = get_metrics_average(datavector_models).sort_values(by='f1_scores', 
                                                                ascending=False).head(1).reset_index()['index'].values[0]
    best_model = datavector_models[number]['model']
    return best_model


def get_worst_model(datavector_models):
    number = get_metrics_average(datavector_models).sort_values(by='f1_scores', 
                                                                ascending=False).tail(1).reset_index()['index'].values[0]
    best_model = datavector_models[number]['model']
    return best_model


# Final classified maps utils


col_names = ['B1',
 'B2',
 'B3',
 'B4',
 'B5',
 'B6',
 'B7',
 'B8',
 'B8A',
 'B9',
 'B11',
 'B12']


col_names_full = ['B1',
 'B2',
 'B3',
 'B4',
 'B5',
 'B6',
 'B7',
 'B8',
 'B8A',
 'B9',
 'B11',
 'B12',
 'ndvi',
 'evi',
 'savi',
 'msi',
 'bsi',
 'ndbi',
 'nbi',
 'bal',
 'mbi',
 'ndsoil',
 'blfei']

def NDVI(red: pd.Series, nir: pd.Series):
    ndvi = (nir - red) / ((nir + red).apply(lambda x: 0.000001 if x == 0 else x))
    return ndvi

def EVI(red: pd.Series, nir: pd.Series, blue: pd.Series):
    evi = (2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1)).apply(lambda x: 0.000001 if x == 0 else x))
    return evi
    

def SAVI(red: pd.Series, nir: pd.Series):  
    savi = ((nir - red) / 1.428*(nir + red + 0.428).apply(lambda x: 0.000001 if x == 0 else x))
    return savi
    

def MSI(nir: pd.Series, swir2: pd.Series): 
    msi = ((swir2/nir).apply(lambda x: 0.000001 if x == 0 else x))
    return msi
    

def BSI(red: pd.Series, nir: pd.Series, swir2: pd.Series, blue: pd.Series):  
    bsi = (((swir2+red)-(nir+blue))/((swir2+red)+(nir+blue)).apply(lambda x: 0.000001 if x == 0 else x))
    return bsi

  #https://doi.org/10.1016/j.envc.2022.100568
   
def NDBI(swir2: pd.Series, nir: pd.Series):
    ndbi = ((swir2-nir)/(swir2+nir).apply(lambda x: 0.000001 if x == 0 else x)) 
    return ndbi

def NBI(red: pd.Series, swir2: pd.Series, nir: pd.Series):
    nbi = (((red*swir2)/nir).apply(lambda x: 0.000001 if x == 0 else x))  
    return nbi

#https://awesome-ee-spectral-indices.readthedocs.io/en/latest/list.html#soil

def BAL(red: pd.Series, swir2: pd.Series, nir: pd.Series):
    bal = ((red+swir2-nir).apply(lambda x: 0.000001 if x == 0 else x))
    return bal

def MBI(swir2: pd.Series, swir22: pd.Series, nir: pd.Series):
    mbi = ((((swir2 - swir22 - nir)/(swir2+swir22+nir)) +0.5).apply(lambda x: 0.000001 if x == 0 else x))
    return mbi

def NDSOIL(swir22: pd.Series, green: pd.Series):
    ndsoil = ((swir22-green)/(swir22+green).apply(lambda x: 0.000001 if x == 0 else x))
    return ndsoil

def BLFEI(red: pd.Series, swir22: pd.Series, swir2: pd.Series, green: pd.Series):
    blfei = ((((green+red+swir22)/3)-swir2)/(((green+red+swir22)/3)+swir2)).apply(lambda x: 0.000001 if x == 0 else x)
    return blfei 


def get_spectral_indices(df: pd.DataFrame) -> pd.DataFrame:
    blue = df['B2']
    green = df['B3']
    red = df['B4']
    nir = df['B8']
    swir2 = df['B11']
    swir22 = df['B12']
    
    df.loc[:, "NDVI"] = NDVI(red=red, nir=nir)
    df.loc[:, "EVI"] = EVI(red=red, nir=nir, blue=blue)
    df.loc[:, "SAVI"] = SAVI(red=red, nir=nir)
    df.loc[:, "MSI"] = MSI(nir=nir, swir2=swir2)
    df.loc[:, "BSI"] = BSI(swir2=swir2, red=red, nir=nir, blue=blue)
    df.loc[:, "NDBI"] = NDBI(swir2=swir2, nir=nir)
    df.loc[:, "NBI"] = NBI(swir2=swir2, red=red, nir=nir)

    df.loc[:, "BAL"] = BAL(red=red, swir2=swir2, nir=nir)
    df.loc[:, "MBI"] = MBI(swir2=swir2, swir22=swir22, nir=nir)
    df.loc[:, "NDSOIL"] = NDSOIL(swir22=swir22, green=green)
    df.loc[:, "BLFEI"] = BLFEI(green=green, red=red, swir2=swir2, swir22=swir22)

    return df

def to_2d_array(x: np.ndarray)->np.ndarray: 
    return x.reshape(x.shape[0], x.shape[1] * x.shape[2])

def save_tif(raster_input:str, raster_output:str, values:np.array):
    in_data, out_data = None, None
    in_data = gdal.Open(raster_input)
    if in_data is None:
        print ('Unable to open %s' % raster_input)
    band1 = in_data.GetRasterBand(1)
    rows = in_data.RasterYSize
    cols = in_data.RasterXSize
    driver = in_data.GetDriver()
    out_data = driver.Create(raster_output, cols, rows, 1, GDT_Int16)
    dem_data = np.array(values)
    out_band = out_data.GetRasterBand(1)
    out_band.WriteArray(dem_data)
    out_band.FlushCache()
    out_band.SetNoDataValue(-1)

    out_data.SetGeoTransform(in_data.GetGeoTransform())
    out_data.SetProjection(in_data.GetProjection())
    del out_data
    return 'Done'

def get_dataset(x: np.ndarray, to_drop = [])->pd.DataFrame:
    bands = x[:12, ...]
    bands = to_2d_array(x[:12, ...]) 
    raw_data = pd.DataFrame(bands.T, columns=col_names)
    df_ = get_spectral_indices(raw_data)
    df_.replace([np.inf, -np.inf], np.nan, inplace=True)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(df_.values)
    df = pd.DataFrame(x_scaled, columns = col_names_full)
    return df

#function for pixels classification
def simple_classifier(df: pd.DataFrame, model)->np.ndarray:
    null_sample = df[df.isnull().any(axis=1)]
    predict_sample = df[~df.isnull().any(axis=1)]
    predict_sample['class'] = model.predict(predict_sample)
    null_sample['class'] = 0
    fin_sample = pd.concat([predict_sample, null_sample], sort=False).sort_index()
    mask = fin_sample['class']
    return mask.values #return np array

def get_raster(path, model, output_name, to_drop = []):
   with rio.open(path, 'r+') as src:
    x = src.read() #raster read
    df = get_dataset(x) #raster to dataframe
    df = df.drop(to_drop, axis=1) #in case some variable need to be droped
    predictions = simple_classifier(df,model) #dataframe classification
    cover_tile = predictions.reshape((x.shape[1], x.shape[2])) #reshaping array to the shape of the raster
    output_mask=cover_tile

    raster_output = output_name + '.tif'#output file name
    status = save_tif(raster_input=path, raster_output=raster_output, values=output_mask)
    print(status)

 
