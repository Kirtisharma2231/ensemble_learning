import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import statistics

df= pd.read_csv("Data_Science/data/AI Job Market Dataset.csv")
print(df.info())
print(df.head(5))
print(df.job_title.value_counts(normalize=True)*100)
print("\n")
df.drop(columns=['job_id'],axis=1,inplace=True)
print(df.head(5))
print(df.job_title.mode())

def jobtitles(titles):
    titles = titles.lower() 
    if 'engineer' in titles and ('ai'in titles or 'machine learning' in titles or 'ml' in titles):
        return 'AI/ML Engineer'
    
    elif 'data scientist' in titles or 'data analyst' in titles:
        return 'Data Scientist/Analyst'
    
    elif 'business analyst' in titles:
        return 'Business Analyst'
    elif 'data engineer' in titles:
        return 'Data Engineer'
    else:
        return 'Others'

df['job_group'] = df['job_title'].apply(jobtitles)
print(df['job_group'])
print(df.head(5))

mapping={
    'IT': "Techonlogy",
    'Info tech': "Techonlogy",
    'Tech': "Techonlogy"
    }
df['company_industry'] = df['company_industry'].str.lower().str.strip()
df['company_industry']=df['company_industry'].replace(mapping)
print(df.head(5))
print(df['company_industry'].unique())
print("\n")

###### combining two columns ie the job_posting_month and job_posting_year #######

df['job_posting_date']=pd.to_datetime(df['job_posting_year'].astype(str)+ "-"+ df['job_posting_month'].astype(str),format='%Y-%m')
df.drop(columns=['job_posting_month','job_posting_year'],inplace=True)
print(df.head(5))

###### Handling the missing values #######
print(df.isnull().sum()) # there are no missing vlaues i this data 

#### feature scaling #####
print(f"Max years of experience : {df.years_experience.max()}")
print(f"Min years of experience : {df.years_experience.min()}")
print(f"Max salary : {df.salary.max()}")
print(f"Min Salary : {df.salary.min()}")

print("\n")
###### this will not work becoz, we have to do the featrue scaling aprt which means converting the categorical data to numerical data but here years_experience is 
###### already a numerical data so it will reaturn the same as years_ experience in years_experience_label #########

labels=preprocessing.LabelEncoder()
df['years_experience_label']=labels.fit_transform(df.years_experience.values)
print(df.head(5))
print(df.years_experience)

###### this is encoding  ie converting the categorical data to numberical data ##########
labels=preprocessing.LabelEncoder()
df['experience_level_label']=labels.fit_transform(df.experience_level.values)
print(df.head(5))
print(df.experience_level.value_counts())
print(df.experience_level_label.value_counts())

label1=preprocessing.LabelEncoder()
df['remote_labels']=label1.fit_transform(df.remote_type.values)
print(df.head(5))
print(df.remote_type.value_counts())
print(df.remote_labels.value_counts())

####### company industry has many categories so how will we encode them ########
frequency=df['company_industry'].value_counts()
df['industry_frequency']=df['company_industry'].map(frequency)
print(df.head(10))
print(df.company_industry.value_counts())


####### this is scaling ie converting the numerical data to the range in numerical form itself #######
scaler=StandardScaler()
df['Salary_scaled']=scaler.fit_transform(df[['salary']])
print(df.head(5))

















    


    


