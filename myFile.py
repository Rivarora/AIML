import pandas as pd
combined_df=pd.read_csv('combined_encoded.csv')
print(combined_df.isnull().sum().sum())
print(combined_df.duplicated().sum())
print(combined_df.dtypes.apply(lambda x: x.name).value_counts())


