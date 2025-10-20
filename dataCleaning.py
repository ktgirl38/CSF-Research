import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder


############################################
#   Read the data from an excel file and combine with the patient data to match the CSF data with the Cohort
#
#   Parameters: clinical_group - Whether we want to look at the baseline or 6 month data
#               path - the path to the excel data file
#
#   Returns: df - a pandas dataframe containing CSF data and the patient data
############################################
def getData(clinical_group="BL", path="FORD-0101-21ML+ DATA TABLES_CSF (METADATA UPDATE).XLSX"):
    df = pd.read_excel(io=path, sheet_name="Batch-normalized Data", header=0, index_col="PARENT_SAMPLE_NAME")

    patientData=pd.read_excel(
        io=path, 
        sheet_name="Sample Meta Data", 
        header = 0, index_col="PARENT_SAMPLE_NAME", 
        usecols=["PARENT_SAMPLE_NAME", "COHORT", "PPMI_CLINICAL_EVENT", "PPMI_COHORT"]
        )

    patientData.drop(patientData[patientData.PPMI_CLINICAL_EVENT != clinical_group].index, inplace=True, axis=0)
    patientData.drop(patientData[patientData.COHORT != "PPMI"].index, inplace=True, axis=0)
    patientData.drop("COHORT", axis=1, inplace=True)
    patientData.drop("PPMI_CLINICAL_EVENT", axis=1, inplace=True)

    #Drop diet related metabolites
    dropMetabolites=[849, 100000445, 100006361, 100004634, 100001605]
    df.drop(columns=dropMetabolites, axis=1, inplace=True)

    
    df=patientData.join(df, on="PARENT_SAMPLE_NAME", how="inner")
    df.columns=df.columns.astype(str)
    
    return df

#Store the data to be used in the functions below
dataset=getData()


############################################
#   Display the number of missing values in each row
#
#   Parameters: data (dataframe) - the pandas dataset to look at
#               thresh (int) - the threshold of missing data to show (ex. When thresh=50, all columns with over 50 missing values will be displayed)
#               display (bool) - a boolean indicating whether to print the amount of missing data in each column
#
#   Returns: the columns that are over the threshold for missing data
############################################
def displayMissing(data, thresh, display):
    
    if(display):
        print(data.columns[data.isnull().sum() > thresh])
    return data.columns[data.isnull().sum() > thresh]


############################################
#   Remove columns with more than a certain amount of missing values. Use KNN Imputation to calculate the remaining missing values
#
#   Parameters: data (pandas dataframe) - the dataframe to clean
#               thresh (float) - the percentage of missing values deemed acceptable
#               n (int) - the number of neighbors to use in the imputation
#           
#   Returns: newX (ndarray) - the cleaned information
############################################
def cleanData(data, thresh=0.5, n=6):
    data.dropna(axis=1, thresh=(len(data)-(thresh*(len(data)))), inplace=True)
    X=data.drop(data.columns[0:1], axis=1)
    Y=data.PPMI_COHORT

    imputer=KNNImputer(n_neighbors=n)
    imputer.set_output(transform="pandas")
    newX=imputer.fit_transform(X,Y)
    return newX

############################################
#   Normalize all information in the dataset using z-normalization
#
#   Parameters: data (pandas dataframe) - the dataframe to normalize
#           
#   Returns: z_scaled (pandas datafram) - the dataframe with normalized data
############################################
def normalizeData(data=dataset):
    z_scaled=data.copy()

    for column in z_scaled.columns:
        if (column != "PARENT_SAMPLE_NAME" and column != "PPMI_COHORT"):
            z_scaled[column] = (z_scaled[column] - z_scaled[column].mean())/z_scaled[column].std()
        
    return z_scaled

############################################
#   Get the X and Y information suitable to use in classification. Data is normalized and clean before broken into the data and the results
#
#   Parameters: data (pandas dataframe) - the dataframe to normalize
#           
#   Returns: X (ndarray) - The CSF information to be used for prediction
#            Y (ndarray) - The cohort data to use for checking and training
############################################
def getXY(data=dataset):
    le=LabelEncoder()

    data=normalizeData()
    X=data.drop(data.columns[0:1], axis=1)
    Y= data.PPMI_COHORT

    Y=le.fit_transform(Y)
    X = cleanData(data)
    X=pd.DataFrame(X)
    return X,Y

#"""
X,Y = getXY()
X['PPMI_COHORT']=Y
X.to_csv("cleanData.csv")
#"""
