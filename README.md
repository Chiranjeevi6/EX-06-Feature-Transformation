# EX-06-Feature-Transformation

## AIM
To Perform the various feature transformation techniques on a dataset and save the data to a file. 

# Explanation
Feature Transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

 
# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature Transformation techniques to all the feature of the data set
### STEP 4
Save the data to the file


# CODE

# importing packages

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm

import scipy.stats as stats

from sklearn.preprocessing import QuantileTransformer 

qt=QuantileTransformer(output_distribution='normal')

df=pd.read_csv("Data_to_Transform.csv")

df

#checking and analysing the data

df.isnull().sum()

#checking for skewness of data

df.skew()

#applying data transformations

dfmp=pd.DataFrame()

#for Moderate Positive Skew

#function transformation

dfmp["Moderate Positive Skew"]=df["Moderate Positive Skew"]


dfmp["MPS_log"]=np.log(df["Moderate Positive Skew"]) 

dfmp["MPS_rp"]=np.reciprocal(df["Moderate Positive Skew"])

dfmp["MPS_sqr"]=np.sqrt(df["Moderate Positive Skew"])

#power transformation

dfmp["MPS_yj"], parameters=stats.yeojohnson(df["Moderate Positive Skew"])

dfmp["MPS_bc"], parameters=stats.boxcox(df["Highly Positive Skew"]) 

#quantile transformation

dfmp["MPS_qt"]=qt.fit_transform(df[["Moderate Positive Skew"]])

dfmp.skew()

dfmp.drop('MPS_rp',axis=1,inplace=True)

dfmp.skew()

dfmp

#for Highly Positive Skew

#function transformation

dfhp=pd.DataFrame()

dfhp["Highly Positive Skew"]=df["Highly Positive Skew"]

dfhp["HPS_log"]=np.log(df["Highly Positive Skew"]) 

dfhp["HPS_rp"]=np.reciprocal(df["Highly Positive Skew"])

dfhp["HPS_sqr"]=np.sqrt(df["Highly Positive Skew"])

#power transformation

dfhp["HPS_yj"], parameters=stats.yeojohnson(df["Highly Positive Skew"])

dfhp["HPS_bc"], parameters=stats.boxcox(df["Highly Positive Skew"]) 

#quantile transformation

dfhp["HPS_qt"]=qt.fit_transform(df[["Highly Positive Skew"]])


dfhp.skew()

dfhp.drop('HPS_sqr',axis=1,inplace=True)

dfhp.skew()

dfhp

#for Moderate Negative Skew

dfmn=pd.DataFrame()

#function transformation

dfmn["Moderate Negative Skew"]=df["Moderate Negative Skew"]

dfmn["MNS_rp"]=np.reciprocal(df["Moderate Negative Skew"])

dfmn["MNS_sq"]=np.square(df["Moderate Negative Skew"])

#power transformation

dfmn["MNS_yj"], parameters=stats.yeojohnson(df["Moderate Negative Skew"]) 
#quantile transformation

dfmn["MNS_qt"]=qt.fit_transform(df[["Moderate Negative Skew"]])

dfmn.skew()

dfmn.drop('MNS_rp',axis=1,inplace=True)

dfmn.skew()

dfmn

#for Highly Negative Skew

dfhn=pd.DataFrame()

#function transformation

dfhn["Highly Negative Skew"]=df["Highly Negative Skew"]

dfhn["HNS_rp"]=np.reciprocal(df["Highly Negative Skew"])

dfhn["HNS_sq"]=np.square(df["Highly Negative Skew"])

#phwer transformation

dfhn["HNS_yj"], parameters=stats.yeojohnson(df["Highly Negative Skew"]) 

#quantile transformation

dfhn["HNS_qt"]=qt.fit_transform(df[["Highly Negative Skew"]])

dfhn.skew()

dfhn.drop('HNS_rp',axis=1,inplace=True)

dfhn.skew()

dfhn

#graphical representation

#for Moderate Positive Skew

df["Moderate Positive Skew"].hist()

dfmp["MPS_log"].hist()

dfmp["MPS_sqr"].hist()


dfmp["MPS_bc"].hist()

dfmp["MPS_yj"].hist()

sm.qqplot(df['Moderate Positive Skew'],line='45')


plt.show()

sm.qqplot(dfmp['MPS_qt'],line='45')

plt.show()

#for Highly Positive Skew

df["Highly Positive Skew"].hist()

dfhp["HPS_log"].hist()

dfhp["HPS_rp"].hist()

dfhp["HPS_bc"].hist()

dfhp["HPS_yj"].hist()

sm.qqplot(df['Highly Positive Skew'],line='45')

plt.show()

sm.qqplot(dfhp['HPS_qt'],line='45')

plt.show()

#for Moderate Negative Skew


df["Moderate Negative Skew"].hist()

dfmn["MNS_sq"].hist()

dfmn["MNS_yj"].hist()

sm.qqplot(df['Moderate Negative Skew'],line='45')

plt.show()

sm.qqplot(dfmn['MNS_qt'],line='45')

plt.show()

#for Highly Negative Skew

df["Highly Negative Skew"].hist()

dfhn["HNS_sq"].hist()

dfhn["HNS_yj"].hist()


sm.qqplot(df['Highly Negative Skew'],line='45')

plt.show()

sm.qqplot(dfhn['HNS_qt'],line='45')

plt.show()

# OUPUT

## Reading the data set:
![s1](https://user-images.githubusercontent.com/94980741/169945732-a3c5e53c-af67-433f-a832-53c594d419a6.png)
![s2](https://user-images.githubusercontent.com/94980741/169945793-5fe88a2c-34c7-493c-95cc-189fdd136b55.png)


## FUNCTION TRANSFORMATION:
![s3](https://user-images.githubusercontent.com/94980741/169945811-b6ce7e69-715e-4260-a111-574afe30c3dc.png)
 ![s4](https://user-images.githubusercontent.com/94980741/169945862-3edd294e-3a4d-4ce4-826d-0baa887ce665.png)
![s5](https://user-images.githubusercontent.com/94980741/169945876-d8c4e20e-7c10-45c0-85e7-62cf90bd3e36.png)
![s6](https://user-images.githubusercontent.com/94980741/169945908-ea310faf-7bb6-44ed-95e5-7a2e14d286da.png)


## POWER TRANSFORMATION:
![s7](https://user-images.githubusercontent.com/94980741/169945976-98bfb49c-2eb8-4b01-9276-1188cbd9236d.png)
![s8](https://user-images.githubusercontent.com/94980741/169945989-03ca665f-9901-4383-b2e5-854ecddadbe3.png)
![s9](https://user-images.githubusercontent.com/94980741/169946000-427861de-ff8c-4ac7-95ed-5e886f031087.png)
![s10](https://user-images.githubusercontent.com/94980741/169946021-560ab6e0-1219-4cdd-9474-191653a26b69.png)

## QUANTILE TRANSFORAMATION:
![s12](https://user-images.githubusercontent.com/94980741/169946056-2ba19c74-9377-4f6a-9f3e-abefb44b39d0.png)
![s13](https://user-images.githubusercontent.com/94980741/169946065-b4c84953-73a0-46bf-8bcf-2e98c636c6e4.png)
![s14](https://user-images.githubusercontent.com/94980741/169946073-a6c4d48e-cf08-48e1-b571-3d026f802fe3.png)
![s15](https://user-images.githubusercontent.com/94980741/169946087-1794afbc-77bd-4ee7-a5b6-ee7cab392dc0.png)
![s17](https://user-images.githubusercontent.com/94980741/169946103-a7bf2c78-6100-4a0c-a610-fa60fb77bea0.png)
![s18](https://user-images.githubusercontent.com/94980741/169946130-65c274f8-427a-41b8-946a-6a29d9d21076.png)
![s19](https://user-images.githubusercontent.com/94980741/169946140-c33f75c1-a344-42c5-abb3-d9c019081904.png)

## Final Result:
![s20](https://user-images.githubusercontent.com/94980741/169946158-7c7f50dd-a0e5-4f39-8b90-c32299a54dba.png)
![s21](https://user-images.githubusercontent.com/94980741/169946166-7afb3420-38fb-4e6d-8c87-c3fdf070877f.png)



# Result:
Hence, Feature transformation techniques is been performed on given dataset and saved into a file successfully.
