import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from narwhals import col
import matplotlib.pyplot as plt


# 1. Load the dataset:
# car price row data set
np.random.seed(42)
n_samples = 650

data = {
    'Car_Brand': np.random.choice(['Maruti', 'Hyundai', 'Honda', 'Toyota', 'Tata', 'BMW'], size=n_samples, p=[0.25, 0.25, 0.15, 0.15, 0.15, 0.05]),
    'Year': np.random.randint(2010, 2023, size=n_samples),
    'Kilometers_Driven': np.random.exponential(scale=50000, size=n_samples) + 10000, # Skewed numeric
    'Fuel_Type': np.random.choice(['Petrol', 'Diesel', 'CNG'], size=n_samples, p=[0.50, 0.45, 0.05]),
    'Transmission': np.random.choice(['Manual', 'Automatic'], size=n_samples, p=[0.75, 0.25]),
    'Engine_CC': np.random.randint(800, 3000, size=n_samples),
    'Seats': np.random.choice([4, 5, 7], size=n_samples, p=[0.05, 0.85, 0.10]),
    'Price': np.random.gamma(shape=3, scale=200000, size=n_samples) + 150000 # Continuous Target
}
df_raw = pd.DataFrame(data)

# data type ko Engine_CC ko number se text banana
df_raw['Engine_CC'] = df_raw['Engine_CC'].astype(str) + "CC"

# null values ko dalna  taki unhe median se bhara ja sake
df_raw.iloc[20:60, 2] = np.nan 
df_raw.iloc[100:150, 6] = np.nan

# duplicate rows ko add karna taki unhe remove kiya ja sake
df_raw = pd.concat([df_raw, df_raw.iloc[0:15]], ignore_index=True)

# Data ko raw_car_data.csv naam se save karna
df_raw.to_csv('raw_car_data.csv', index=False)

print("raw_car_data.csv' successfull ban gaya")

# data fist five rows, data types, and shape 
print("--- first 5 Rows---")
print(df_raw.head(5))

print("---Data Types---")
print(df_raw.dtypes)

print("---Shape DataFrame---")
print(df_raw.shape)

# 2.Null value analysis:
# sabse pahle file ko 'df' me load karna  

df = pd.read_csv('raw_car_data.csv')
print("--- Task 2: Null Value Analysis ---")
print("---Har column me null value ki sankhya---")
print(df.isnull().sum())

print("---Har column me null value ki percentage(%)---")
print((df.isnull().sum() / df.shape[0]) * 100)

# jin column me null value hai, unhe uske median se bharna
df['Kilometers_Driven'] = df['Kilometers_Driven'].fillna(df['Kilometers_Driven'].median())
df['Seats'] = df['Seats'].fillna(df['Seats'].median())

print("--- After Filling Null Values (Confirmation) ---")
print(df.isnull().sum())

# 3.Duplicate detection and removal:
print("---3. Total Duplicate Rows ---")
print(df.duplicated().sum())

# Duplicate rows ko remove karna
df = df.drop_duplicates()
print("---New DataFrame shape after dropping duplicates---")
print(df.shape)

# 4. Data type correction: 
print("--- 4. Data Type Correction & Memory Optimization ---")

# Engine_CC column me se 'CC' ko hatakar numeric me convert karna   

df['Engine_CC'] = df['Engine_CC'].str.replace('CC', '', regex=False)
df['Engine_CC'] = pd.to_numeric(df['Engine_CC'], errors='coerce')

# string ko category me badalna
df['Car_Brand'] = df['Car_Brand'].astype('category')
df['Fuel_Type'] = df['Fuel_Type'].astype('category')

print("--- Corrected Data Types ---")
print(df.dtypes)

# 5.Descriptive statistics and skewness: 
print("--- 5. Descriptive Statistics and Skewness ---")
# Sabhi numerical columns ki data summary (.describe()) print karna
numeric_cols = df.select_dtypes(include=['int32', 'int64', 'float64']).columns
print(df[numeric_cols].describe())

# Har numerical column ka skewness  check karna
print("\n--- Skewness Values ---")
print(df[numeric_cols].skew())

# 6. Outlier detection with IQR:
print("--- 6. Outlier Detection with IQR ---")
columns_to_check = ['Kilometers_Driven', 'Price']
for col in columns_to_check:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers_count = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
    print(f"column '{col}' me outliers ki sankhya: {outliers_count}")

# 7.Visualizations (all five types required):
print("--- 7. Visualizations ---")

# 1. line plot Price sorted
plt.figure(figsize=(8,4))
plt.plot(df['Price'].sort_index())
plt.title('Line Plot of Car Prices')
plt.xlabel('Row Index') 
plt.ylabel('Price')
plt.show()

# 2.bar chart Car_Brand vs Mean Price
plt.figure(figsize=(8, 4))
df.groupby('Car_Brand')['Price'].mean().plot(kind='bar', color='skyblue')
plt.title('mean price by Car Brand')
plt.xlabel('Car Brand')
plt.ylabel('mean price')
plt.show()

# 3. histogram  kilometers driven ka skewness 
plt.figure(figsize=(8, 4))
sns.histplot(df['Kilometers_Driven'],bins=20,kde=True,color='green')
plt.title('histogram of Kilometers Driven')
plt.xlabel('Kilometers Driven')
plt.show()

# 4.scatter plot (Engine_CC vs Price - correlation check)
plt.figure(figsize=(8, 4))
sns.scatterplot(data=df, x='Engine_CC', y='Price', alpha=0.6)
plt.title('scatter plot of Engine CC vs Price')
plt.xlabel('Engine CC')
plt.ylabel('Price')
plt.show()

# 5.Box Plot (Price split by Fuel_Type)
plt.figure(figsize=(8, 4))
sns.boxplot(data=df, x='Fuel_Type', y='Price')
plt.title('box plot ka price by fuel type')
plt.xlabel('Fuel Type')
plt.ylabel('Price')
plt.show()

# 8. Correlation heat map: 
print("--- 8. Correlation Heatmap ---")
pearson_matrix = df[numeric_cols].corr(method='pearson')
plt.figure(figsize=(8, 6))
sns.heatmap(pearson_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('pearson correlation')
plt.show()

# 9a.  Imputation strategy comparison.
print("--- 9. Imputation Strategy Comparison ---")
# numberic columns ki  skewess nikalna
skew_values = df[numeric_cols].skew().abs()
top_skewed = skew_values.nlargest(2).index.tolist()
for col in top_skewed:
    mean_value = df[col].mean()
    median_value = df[col].median()
    print(f"Column: {col} | Mean: {mean_value:.2f} | Median: {median_value:.2f}")

# Skewed data ke liye median achha hota hai, isliye median se fill kar rahe hai
df[col] = df[col].fillna(median_value)
print("null value 0 hona chahiye")
print(df.isnull().sum()) 

#9b. Spearman rank correlation. 
print("--- 9b. Spearman Rank Correlation ---")
spearman_matrix = df[numeric_cols].corr(method='spearman')
diff_matrix = (spearman_matrix - pearson_matrix).abs()
print("---pearson_matrix---")
print(pearson_matrix)
print("--- Spearman Matrix ---")
print(spearman_matrix)
print("--- Absolute Difference (|Spearman - Pearson|) ---")
print(diff_matrix)

top_diffs = diff_matrix.unstack().sort_values(ascending=False)
top_diffs = top_diffs[top_diffs.index.get_level_values(0) < top_diffs.index.get_level_values(1)]
print("-- top 3 pairs with large differences---")
print(top_diffs.head(3))

# 9c.Grouped Aggregation:
print("--- 9c. Grouped Aggregation ---")
agg_df = df.groupby('Car_Brand')['Price'].agg(['mean', 'std', 'count'])
print('---grouped aggregation (brand vs price)---')
print(agg_df)

# calculation for readme.
highest_mean = agg_df['mean'].idxmax()
highest_std = agg_df['std'].idxmax()
radio = agg_df['mean'].max() / agg_df['std'].max()  
print("highest mean: {highest_mean}")
print(f"highest std: {highest_std}")
print(f"ratio (max/min): {radio:.2f}")

# 10. Save the clean dataset.
df.to_csv('cleaned_car_data.csv', index=False)
print("cleaned_car_data.csv' successfully saved")












