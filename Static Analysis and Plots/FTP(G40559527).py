#%%

########    FINAL TERM PROJECT(FTP) Begins  ########

import numpy as np
import pandas as pd
import re
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata
import scipy.stats as stats
from scipy.stats import boxcox
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pandas.plotting import scatter_matrix
from scipy.stats import shapiro
from scipy.stats import kstest, norm
from itertools import combinations


print("\n\nYoutube Pulse : Tracking Trends and Engagement\n")
print("Designed by :- Surya vamsi Patiballa(G40559527)\n\n")
odata = pd.read_csv('/Users/surya/Documents/Class files/Data visualization/FTP/FTP(Surya)/finaldata.csv')
numeric_columns = odata.select_dtypes(include=['float64', 'int64']).columns.dropna('category_id')

#%%

#%%

###   DESCRIPTION OF DATASET   ###

print("Description of my Youtube.csv Dataset\n\n")


mydata = pd.read_csv('/Users/surya/Documents/Class files/Data visualization/FTP/FTP(Surya)/youtube.csv')
nfeat = mydata[['views', 'likes', 'dislikes', 'comment_count']].describe()
cfeat = ['video_id', 'trending_date', 'title', 'channel_title', 'category_id',
                        'publish_date', 'time_frame', 'published_day_of_week', 'publish_country', 'tags',
                        'comments_disabled', 'ratings_disabled', 'video_error_or_removed']
csumm = {}
categorical_features = mydata.select_dtypes(include=['object']).columns.tolist()
print("\n\nCategorical Features :- \n", categorical_features)
for feature in cfeat:
    csumm[feature] = mydata[feature].value_counts()

print("\nNumerical Features :-\n", nfeat)
print(mydata.head(10))

#%%

#%%

###   DATA PREPROCESSING & DATA CLEANING   ###

print("\nData Preprocessing & Cleaning\n\n")

missing_values = numeric_columns.isnull().sum()
print("\nCount of missing values for each feature :-\n")
print(missing_values)

# dropping NaN/NULL values
mydata = mydata.dropna()

# Dropping 'video_id' and 'index' columns
mydata = mydata.drop('index', axis=1,inplace=False)
mydata = mydata.drop('video_id', axis=1,inplace=False)
mydata = mydata.drop('comments_disabled', axis=1,inplace=False)
mydata = mydata.drop('ratings_disabled', axis=1,inplace=False)
mydata = mydata.drop('video_error_or_removed', axis=1,inplace=False)


# Converting format of 'trending_date' column
mydata['trending_date'] = pd.to_datetime(mydata['trending_date'], format='%y.%d.%m').dt.strftime('%m-%d-%Y')

# Converting format of 'publish_date' column
mydata['publish_date'] = pd.to_datetime(mydata['publish_date'], format='%d/%m/%Y').dt.strftime('%m-%d-%Y')

# Modifying the 'time_frame' column to bring out WEEKDAYS and naming it as 'part_of_day'
def categorize_time_of_day(time_range):
    start_time = pd.to_datetime(time_range.split(' to ')[0], format='%H:%M').time()
    if start_time.hour < 6:
        return 'Night'
    elif start_time.hour < 12:
        return 'Morning'
    elif start_time.hour < 18:
        return 'Afternoon'
    else:
        return 'Evening'

mydata['part_of_day'] = mydata['time_frame'].apply(categorize_time_of_day)


# Eliminating Structural errors like Misspellings,Improper capitalization,Naming convention errors,etc..,

def clean_text(text):
    # Fixing common misspellings,capitalization and removing extra spaces
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r"[^\w\s]", '', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'#\S+', '', text)
    return text.strip()


    # Applying the cleaning function to text columns
textcols = ['title', 'channel_title', 'tags']
for col in textcols:
    mydata[col] = mydata[col].apply(clean_text)
print("\nDataset after dropping useless columns :-\n")
print(mydata.head(10))

#%%

#%%

# Outliers Detection (using the IQR method) & Data Transformation(Box-Cox Transform)

print("\nOutlier Detection\n")
print("\nBox plot before Outlier Removal\n")

numcols = ['views', 'likes', 'dislikes', 'comment_count']
plt.figure(figsize=(12, 8))
for index, column in enumerate(numcols):
    plt.subplot(2, 2, index + 1)
    sns.boxplot(y=mydata[column])
    plt.title(f'Box Plot (before Outliers) of {column}', fontdict={'fontname': 'serif', 'color': 'blue', 'fontsize': 'large'})
    plt.ylabel('Values', fontdict={'fontname': 'serif', 'color': 'darkred', 'fontsize': 'large'})
    plt.xlabel(column, fontdict={'fontname': 'serif', 'color': 'darkred', 'fontsize': 'large'})
plt.tight_layout()
plt.show()

print("\nOops,the data is Non-Gaussian Distributed!!\n")
print("So,I am going to Transform data into Gaussian Distribution using Box-Cox Transformation\n\n")

# Applying Box-Cox transformation

myviewsdata = mydata[mydata['views'] > 0]
viewstrans, fitted_lambda1 = boxcox(myviewsdata['views'])
mydata.loc[myviewsdata.index, 'transformed_views'] = viewstrans
plt.figure(figsize=(10, 6))
plt.hist(viewstrans, bins=50, color='green', edgecolor='black')
plt.title('Histogram of Transformed YouTube Video Views', fontdict={'fontname': 'serif', 'color': 'blue', 'fontsize': 'large'})
plt.xlabel('Transformed Views', fontdict={'fontname': 'serif', 'color': 'darkred', 'fontsize': 'large'})
plt.ylabel('Frequency', fontdict={'fontname': 'serif', 'color': 'darkred', 'fontsize': 'large'})
plt.grid(True)
plt.show()
print(f"Fitted Lambda for Box-Cox Transformation[Views]: {fitted_lambda1}")


mylikesdata = mydata[mydata['likes'] > 0]
likestrans, fitted_lambda2 = boxcox(mylikesdata['likes'])
mydata.loc[mylikesdata.index, 'transformed_likes'] = likestrans
plt.figure(figsize=(10, 6))
plt.hist(likestrans, bins=50, color='green', edgecolor='black')
plt.title('Histogram of Transformed YouTube Video Likes', fontdict={'fontname': 'serif', 'color': 'blue', 'fontsize': 'large'})
plt.xlabel('Transformed Likes', fontdict={'fontname': 'serif', 'color': 'darkred', 'fontsize': 'large'})
plt.ylabel('Frequency', fontdict={'fontname': 'serif', 'color': 'darkred', 'fontsize': 'large'})
plt.grid(True)
plt.show()
print(f"Fitted Lambda for Box-Cox Transformation[Likes]: {fitted_lambda2}")


mydislikesdata = mydata[mydata['dislikes'] > 0]
dislikestrans, fitted_lambda3 = boxcox(mydislikesdata['dislikes'])
mydata.loc[mydislikesdata.index, 'transformed_dislikes'] = dislikestrans
plt.figure(figsize=(10, 6))
plt.hist(dislikestrans, bins=50, color='green', edgecolor='black')
plt.title('Histogram of Transformed YouTube Video Dislikes', fontdict={'fontname': 'serif', 'color': 'blue', 'fontsize': 'large'})
plt.xlabel('Transformed Dislikes', fontdict={'fontname': 'serif', 'color': 'darkred', 'fontsize': 'large'})
plt.ylabel('Frequency', fontdict={'fontname': 'serif', 'color': 'darkred', 'fontsize': 'large'})
plt.grid(True)
plt.show()
print(f"Fitted Lambda for Box-Cox Transformation[Dislikes]: {fitted_lambda3}")


mycommdata = mydata[mydata['comment_count'] > 0]
commtrans, fitted_lambda4 = boxcox(mycommdata['comment_count'])
mydata.loc[mycommdata.index, 'transformed_comment_count'] = commtrans
plt.figure(figsize=(10, 6))
plt.hist(commtrans, bins=50, color='green', edgecolor='black')
plt.title('Histogram of Transformed YouTube Video Comment_count', fontdict={'fontname': 'serif', 'color': 'blue', 'fontsize': 'large'})
plt.xlabel('Transformed Comment_count', fontdict={'fontname': 'serif', 'color': 'darkred', 'fontsize': 'large'})
plt.ylabel('Frequency', fontdict={'fontname': 'serif', 'color': 'darkred', 'fontsize': 'large'})
plt.grid(True)
plt.show()
print(f"Fitted Lambda for Box-Cox Transformation[Comment_count]: {fitted_lambda4}")

#%%

#%%

# Outlier Removal and END OF DATA CLEANING

print("\nOutlier Removal Begins...\n")

print("\nShape od data before Outlier Removal :-",mydata.shape)
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

for column in ['views', 'likes', 'dislikes', 'comment_count']:
    mydata = remove_outliers(mydata, column)

print("\nShape of data after removing outliers :-", mydata.shape)


print("\nBox plot after Outliers removal\n")

plt.figure(figsize=(12, 8))
for index, column in enumerate(numcols):
    plt.subplot(2, 2, index + 1)
    sns.boxplot(y=mydata[column])
    plt.title(f'Box Plot (After Outliers) of {column}',fontdict={'fontname': 'serif', 'color': 'blue', 'fontsize': 'large'})
    plt.ylabel('Values', fontdict={'fontname': 'serif', 'color': 'darkred', 'fontsize': 'large'})
    plt.xlabel(column, fontdict={'fontname': 'serif', 'color': 'darkred', 'fontsize': 'large'})
plt.tight_layout()
plt.show()

mydata = mydata.drop('transformed_views', axis=1,inplace=False)
mydata = mydata.drop('transformed_likes', axis=1,inplace=False)
mydata = mydata.drop('transformed_dislikes', axis=1,inplace=False)
mydata = mydata.drop('transformed_comment_count', axis=1,inplace=False)
mydata = mydata.drop('time_frame', axis=1,inplace=False)
print("\nFinally,This 'youtube.csv' dataset is cleaned!!\n")
'''mydata.to_csv('finaldata.csv', index=False)'''
print("\nThe cleaned dataset is saved as 'finaldata.csv'.\n\n")

#%%

#%%

###   PRINCIPAL COMPONENT ANALYSIS (PCA)   ###

print("\nPrincipal Component Analysis(PCA)\n\n")
odata = pd.read_csv('/Users/surya/Documents/Class files/Data visualization/FTP/FTP(Surya)/finaldata.csv')
ncols = odata.select_dtypes(include=[np.number]).dropna()
sc = StandardScaler()
scaleddata = sc.fit_transform(ncols)
pca = PCA(n_components=min(scaleddata.shape))
pca.fit(scaleddata)
sval = pca.singular_values_
cdnum = sval.max() / sval.min()
print("\nSingular values :-\n\n", sval)
print("\nCondition number :-\n\n", cdnum)
datatrans = pca.transform(scaleddata)
print("\nTransformed Data Shape :-", datatrans.shape)

#%%

#%%

###   CUMULATIVE EXPLAINED VARIANCE GRAPH   ###

X_std = StandardScaler().fit_transform(numeric_columns)
pca = PCA().fit(X_std)
cumulative_var = pca.explained_variance_ratio_.cumsum()
plt.figure(figsize=(8, 4))
plt.plot(range(1, len(cumulative_var) + 1), cumulative_var, marker='o', linestyle='--', label='Cumulative Explained Variance')
plt.title('Cumulative Explained Variance by PCA Components')
plt.xlabel('Number of PCA Components')
plt.ylabel('Cumulative Explained Variance')
plt.legend()
plt.show()

#%%

#%%

###   Normality Test (Kolmogorov-Smirnov Test)  ###

print("\nKolmogorov-Smirnov Test\n")

cols = ['views', 'likes', 'dislikes', 'comment_count']
impdata = odata[cols].dropna()

ks_results = {}
for column in impdata.columns:
    data = impdata[column]
    stat, p_value = kstest(data, 'norm', args=(data.mean(), data.std()))
    ks_results[column] = (stat, p_value)

for column, results in ks_results.items():
    stat, p_value = results
    print(f'Column :- {column}')
    print(f'K-S Statistic :- {stat}, P-value :- {p_value}')
    if p_value > 0.05:
        print("Normally distributed\n")
    else:
        print("Not Normally distributed\n")

#%%

#%%

###   HEATMAP & Pearson Correlation Coefficient(PCC)   ###

odata = pd.read_csv('/Users/surya/Documents/Class files/Data visualization/FTP/FTP(Surya)/finaldata.csv')
print("\nHeat Map,Scatter plot and Table\n")
columns = ['views', 'likes', 'dislikes', 'comment_count']
correlation_matrix = odata[columns].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm',square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title('Heatmap of Pearson Correlation Coefficients for Selected Variables',fontdict={'fontname': 'serif', 'color': 'blue', 'size': 'large'})
plt.show()

sns.pairplot(odata[columns])
plt.suptitle('Pair Plot for Selected Variables', fontdict={'fontname': 'serif', 'color': 'blue', 'size': 'large'}, y=1.02)
plt.show()

# Tabular format of all values

descriptive_stats = odata[columns].describe()
tab = pd.concat([correlation_matrix, descriptive_stats])
table = PrettyTable()
table.field_names = [""] + list(tab.columns)
for index, row in tab.iterrows():
    table.add_row([index] + row.tolist())
print(table)

#%%

#%%

###   Estimated Multivariate Kernel Density Estimation (KDE)   ###

columns = ['views', 'likes', 'dislikes', 'comment_count']
g = sns.pairplot(odata[columns], kind='kde')
plt.suptitle('KDE Pair Plot for Selected Variables', fontdict={'fontname': 'serif', 'color': 'blue', 'fontsize': 'large'}, y=1.05)
for ax in g.axes.flatten():
    ax.set_xlabel(ax.get_xlabel(), fontdict={'fontname': 'serif', 'color': 'darkred', 'fontsize': 'large'})
    ax.set_ylabel(ax.get_ylabel(), fontdict={'fontname': 'serif', 'color': 'darkred', 'fontsize': 'large'})
plt.tight_layout()
plt.show()

#%%


#%%

###   DATA VISUALIZATION(Static plots)   ###

# BAR PLOT for most common tags(Cleaning and splitting tags)
def process_tags(tags):
    # Replacing quotation marks and splitting the tags
    tags = tags.replace('"', '').split('|')
    return [tag.strip().lower() for tag in tags if tag.strip().lower() != '[none]']

# Applying the function and expanding all tags
all_tags = mydata['tags'].apply(process_tags).explode()

# Counting the occurrences of each tag
tag_counts = Counter(all_tags)

# Choosing the top 20 most common tags to plot
most_common_tags = tag_counts.most_common(20)
tags, counts = zip(*most_common_tags)
plt.figure(figsize=(12, 8))
plt.bar(tags, counts, color='blue')
plt.xlabel('Tags', fontname='serif', color='darkred', fontsize='large')
plt.ylabel('Frequency', fontname='serif', color='darkred', fontsize='large')
plt.title('Top 20 Most Common YouTube Tags', fontname='serif', color='blue', fontsize='large')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

#%%

#%%

###   LINE PLOTS   ###

odata = pd.read_csv('/Users/surya/Documents/Class files/Data visualization/FTP/FTP(Surya)/finaldata.csv')
odata['publish_date'] = pd.to_datetime(odata['publish_date'])
monthly_data = odata.groupby(odata['publish_date'].dt.to_period('M')).agg({
    'views': 'sum',
    'likes': 'sum',
    'dislikes': 'sum',
    'comment_count': 'sum'}).reset_index()

monthly_data['publish_date'] = monthly_data['publish_date'].dt.to_timestamp()
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
fig.suptitle('Monthly Trends of YouTube Video Metrics', fontdict={'fontname': 'serif', 'color': 'blue', 'size': 'large'}, y=0.95)

axes[0, 0].plot(monthly_data['publish_date'], monthly_data['views'], label='Views', color='blue')
axes[0, 0].set_title('Monthly Views', fontdict={'fontname': 'serif', 'color': 'blue', 'size': 'large'})
axes[0, 0].set_xlabel('Month', fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 'large'})
axes[0, 0].set_ylabel('Views', fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 'large'})
axes[0, 0].grid(True)

axes[0, 1].plot(monthly_data['publish_date'], monthly_data['likes'], label='Likes', color='green')
axes[0, 1].set_title('Monthly Likes', fontdict={'fontname': 'serif', 'color': 'blue', 'size': 'large'})
axes[0, 1].set_xlabel('Month', fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 'large'})
axes[0, 1].set_ylabel('Likes', fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 'large'})
axes[0, 1].grid(True)

axes[1, 0].plot(monthly_data['publish_date'], monthly_data['dislikes'], label='Dislikes', color='red')
axes[1, 0].set_title('Monthly Dislikes', fontdict={'fontname': 'serif', 'color': 'blue', 'size': 'large'})
axes[1, 0].set_xlabel('Month', fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 'large'})
axes[1, 0].set_ylabel('Dislikes', fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 'large'})
axes[1, 0].grid(True)

axes[1, 1].plot(monthly_data['publish_date'], monthly_data['comment_count'], label='Comment Count', color='purple')
axes[1, 1].set_title('Monthly Comment Count', fontdict={'fontname': 'serif', 'color': 'blue', 'size': 'large'})
axes[1, 1].set_xlabel('Month', fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 'large'})
axes[1, 1].set_ylabel('Comment Count', fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 'large'})
axes[1, 1].grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

#%%

#%%

###   BAR PLOTS   ###

odata = pd.read_csv('/Users/surya/Documents/Class files/Data visualization/FTP/FTP(Surya)/finaldata.csv')
plot_data = odata.head(10)
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(plot_data['title'], plot_data['likes'], label='Likes')
ax.bar(plot_data['title'], plot_data['dislikes'], bottom=plot_data['likes'], label='Dislikes')
ax.bar(plot_data['title'], plot_data['comment_count'], bottom=plot_data['likes'] + plot_data['dislikes'], label='Comment Count')

ax.set_ylabel('Counts', fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 'large'})
ax.set_title('Stacked Metrics per Video', fontdict={'fontname': 'serif', 'color': 'blue', 'size': 'large'})
ax.legend()

plt.xticks(rotation=90)
plt.tight_layout()

# Grouped bar chart
fig, ax = plt.subplots(figsize=(10, 6))
width = 0.25
r1 = range(len(plot_data['title']))
r2 = [x + width for x in r1]
r3 = [x + width for x in r2]

ax.bar(r1, plot_data['likes'], color='b', width=width, edgecolor='grey', label='Likes')
ax.bar(r2, plot_data['dislikes'], color='r', width=width, edgecolor='grey', label='Dislikes')
ax.bar(r3, plot_data['comment_count'], color='g', width=width, edgecolor='grey', label='Comment Count')

ax.set_xlabel('Video Title', fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 'large'})
ax.set_ylabel('Counts', fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 'large'})
ax.set_title('Grouped Metrics per Video', fontdict={'fontname': 'serif', 'color': 'blue', 'size': 'large'})
ax.set_xticks([r + width/2 for r in r1])
ax.set_xticklabels(plot_data['title'], fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 'large'})
ax.legend()

plt.xticks(rotation=90)
plt.tight_layout()

plt.show()

#%%

#%%

###   COUNTOUR PLOTS   ###

odata = pd.read_csv('/Users/surya/Documents/Class files/Data visualization/FTP/FTP(Surya)/finaldata.csv')
data = odata[['views', 'likes', 'dislikes']].dropna()
x = np.linspace(data['views'].min(), data['views'].max(), 100)
y = np.linspace(data['likes'].min(), data['likes'].max(), 100)
X, Y = np.meshgrid(x, y)
Z = griddata((data['views'], data['likes']), data['dislikes'], (X, Y), method='cubic')

# Create contour plot
plt.figure(figsize=(10, 7))
cp = plt.contourf(X, Y, Z, 20, cmap='viridis')
plt.colorbar(cp)
plt.title('Contour Plot of Dislikes against Views and Likes', fontdict={'fontname': 'serif', 'color': 'blue', 'size': 'large'})
plt.xlabel('Views', fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 'large'})
plt.ylabel('Likes', fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 'large'})
plt.grid(True)
plt.show()

#%%

#%%

### PIE CHARTS   ###

# Chart - 1
odata = pd.read_csv('/Users/surya/Documents/Class files/Data visualization/FTP/FTP(Surya)/finaldata.csv')
feature_to_plot = 'part_of_day'
category_counts = odata[feature_to_plot].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=90)
plt.title(f'Distribution of Videos by {feature_to_plot}', fontdict={'fontname': 'serif', 'color': 'blue', 'fontsize': 'large'})
plt.axis('equal')
plt.show()

# Chart - 2
feature_toplot = 'category_id'
category_count = odata[feature_toplot].value_counts().head(10)
plt.figure(figsize=(8, 8))
plt.pie(category_count, labels=category_count.index, autopct='%1.1f%%', startangle=90)
plt.title('Top 10 Video Categories', fontdict={'fontname': 'serif', 'color': 'blue', 'fontsize': 'large'})
plt.axis('equal')
plt.show()

# Chart - 3
featuretoplot = 'published_day_of_week'
category_counts = odata[featuretoplot].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=90)
plt.title(f'Distribution of Videos by {featuretoplot}', fontdict={'fontname': 'serif', 'color': 'blue', 'fontsize': 'large'})
plt.axis('equal')
plt.show()

#%%

#%%

###   HIST,BOX & KDE PLOTS   ###

odata = pd.read_csv('/Users/surya/Documents/Class files/Data visualization/FTP/FTP(Surya)/finaldata.csv')

# Histogram Plot (4 plots)

plt.figure(figsize=(15, 10))
for i, column in enumerate(numeric_columns, 1):
    plt.subplot(2, 2, i)
    sns.histplot(odata[column], kde=True, element='step')
    plt.title(f'Distribution of {column}', fontdict={'fontname': 'serif', 'color': 'blue', 'fontsize': 'large'})
    plt.xlabel(column, fontdict={'fontname': 'serif', 'color': 'darkred', 'fontsize': 'large'})
    plt.ylabel('Frequency', fontdict={'fontname': 'serif', 'color': 'darkred', 'fontsize': 'large'})
plt.tight_layout()
plt.show()

# Box plot (4 plots)
plt.figure(figsize=(15, 10))
for i, column in enumerate(numeric_columns, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(y=odata[column])
    plt.title(f'Boxplot of {column}', fontdict={'fontname': 'serif', 'color': 'blue', 'fontsize': 'large'})
    plt.xlabel(column, fontdict={'fontname': 'serif', 'color': 'darkred', 'fontsize': 'large'})
    plt.ylabel('Values', fontdict={'fontname': 'serif', 'color': 'darkred', 'fontsize': 'large'})
plt.tight_layout()
plt.show()

# KDE Plot (4 plots)
plt.figure(figsize=(15, 10))
for i, column in enumerate(numeric_columns, 1):
    plt.subplot(2, 2, i)
    sns.kdeplot(odata[column], fill=True)
    plt.title(f'Kernel Density Estimation of {column}', fontdict={'fontname': 'serif', 'color': 'blue', 'fontsize': 'large'})
    plt.xlabel(column, fontdict={'fontname': 'serif', 'color': 'darkred', 'fontsize': 'large'})
    plt.ylabel('Density', fontdict={'fontname': 'serif', 'color': 'darkred', 'fontsize': 'large'})
plt.tight_layout()
plt.show()

#%%

#%%

###   MULTIVARIATE BOX / BOXEN PLOTS   ###

odata = pd.read_csv('/Users/surya/Documents/Class files/Data visualization/FTP/FTP(Surya)/finaldata.csv')
plt.figure(figsize=(15, 10))
colors = sns.color_palette("husl", len(numeric_columns))
for i, column in enumerate(numeric_columns, 1):
    plt.subplot(2, 2, i)
    sns.boxenplot(x=odata[column], color=colors[i-1])
    plt.title(f'Boxen plot for {column}', fontdict={'fontname': 'serif', 'color': 'blue', 'fontsize': 'large'})
    plt.xlabel(column, fontdict={'fontname': 'serif', 'color': 'darkred', 'fontsize': 'large'})
    plt.ylabel('Values', fontdict={'fontname': 'serif', 'color': 'darkred', 'fontsize': 'large'})
plt.tight_layout()
plt.show()

#%%

#%%

###   COUNT PLOTS   ###

odata = pd.read_csv('/Users/surya/Documents/Class files/Data visualization/FTP/FTP(Surya)/finaldata.csv')
categcolumns = odata.select_dtypes(include=['object', 'category']).columns
categcolumns = [col for col in categcolumns if odata[col].nunique() < 20]
plt.figure(figsize=(15, 5 * len(categcolumns)))
palette = sns.color_palette("husl", len(categcolumns))
for i, column in enumerate(categcolumns, 1):
    plt.subplot(len(categcolumns), 1, i)
    sns.countplot(x=odata[column], order=odata[column].value_counts().index, palette=[palette[i-1]])
    plt.title(f'Count Plot for {column}', fontdict={'fontname': 'serif', 'color': 'blue', 'fontsize': 'large'})
    plt.xlabel(column, fontdict={'fontname': 'serif', 'color': 'darkred', 'fontsize': 'large'})
    plt.ylabel('Count', fontdict={'fontname': 'serif', 'color': 'darkred', 'fontsize': 'large'})
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%%

#%%

###   Q-Q PLOTS   ###

odata = pd.read_csv('/Users/surya/Documents/Class files/Data visualization/FTP/FTP(Surya)/finaldata.csv')
plt.figure(figsize=(10, 5 * len(numeric_columns)))
for i, column in enumerate(numeric_columns, 1):
    plt.subplot(len(numeric_columns), 1, i)
    stats.probplot(odata[column], dist="norm", plot=plt)
    plt.title(f'Q-Q Plot for {column}', fontdict={'fontname': 'serif', 'color': 'blue', 'fontsize': 'large'})
    plt.xlabel('Theoretical Quantiles', fontdict={'fontname': 'serif', 'color': 'darkred', 'fontsize': 'large'})
    plt.ylabel('Ordered Values', fontdict={'fontname': 'serif', 'color': 'darkred', 'fontsize': 'large'})
plt.tight_layout()
plt.show()

#%%

#%%

###   REGRESSION PLOT   ###

odata = pd.read_csv('/Users/surya/Documents/Class files/Data visualization/FTP/FTP(Surya)/finaldata.csv')
feature_combinations = combinations(numeric_columns, 2)
for combo in feature_combinations:
    sns.regplot(x=combo[0], y=combo[1], data=odata)
    plt.xlabel(combo[0], fontdict={'fontname': 'serif', 'color': 'darkred', 'fontsize': 'large'})
    plt.ylabel(combo[1], fontdict={'fontname': 'serif', 'color': 'darkred', 'fontsize': 'large'})
    plt.title(f'Regression Plot of {combo[0]} vs {combo[1]}',fontdict={'fontname': 'serif', 'color': 'blue', 'fontsize': 'large'})
    plt.show()

#%%

#%%

###   BAR GRAPHS   ###

odata = pd.read_csv('/Users/surya/Documents/Class files/Data visualization/FTP/FTP(Surya)/finaldata.csv')
categorical_columns = [col for col in odata.columns if odata[col].nunique() < 20]
num_plots = len(categorical_columns)
cols = 2
rows = num_plots // cols + (num_plots % cols > 0)
plt.figure(figsize=(20, 5 * rows))
palette = sns.color_palette("hsv", len(categorical_columns))
for i, column in enumerate(categorical_columns, 1):
    plt.subplot(rows, cols, i)
    sns.countplot(x=odata[column], order=odata[column].value_counts().index, color=palette[i-1])
    plt.title(f'Bar Graph for {column}', fontdict={'fontname': 'serif', 'color': 'blue', 'fontsize': 'large'})
    plt.xlabel(column, fontdict={'fontname': 'serif', 'color': 'darkred', 'fontsize': 'large'})
    plt.ylabel('Count', fontdict={'fontname': 'serif', 'color': 'darkred', 'fontsize': 'large'})
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%%

#%%

###   AREA PLOTS   ###

odata = pd.read_csv('/Users/surya/Documents/Class files/Data visualization/FTP/FTP(Surya)/finaldata.csv')
numeric_columns = odata.select_dtypes(include=['float64', 'int64']).columns.dropna()
for column in numeric_columns:
    plt.figure(figsize=(10, 6))
    odata[column].plot(kind='area', color='skyblue', alpha=0.5)
    plt.title(f'Area Plot of {column}', fontdict={'fontname': 'serif', 'color': 'blue', 'fontsize': 'large'})
    plt.xlabel('Index', fontdict={'fontname': 'serif', 'color': 'darkred', 'fontsize': 'large'})
    plt.ylabel(column, fontdict={'fontname': 'serif', 'color': 'darkred', 'fontsize': 'large'})
    plt.grid(True)
    plt.show()

#%%

#%%

###   VIOLIN PLOTS   ###

odata = pd.read_csv('/Users/surya/Documents/Class files/Data visualization/FTP/FTP(Surya)/finaldata.csv')
numeric_columns = odata.select_dtypes(include=['float64', 'int64']).columns.dropna()
sns.set(style="whitegrid")
fig, axes = plt.subplots(len(numeric_columns), 1, figsize=(10, 5 * len(numeric_columns)))
if len(numeric_columns) == 1:
    axes = [axes]
for ax, column in zip(axes, numeric_columns):
    sns.violinplot(data=odata, x=column, ax=ax, inner='quartile', palette='husl')
    ax.set_title(f'Violin Plot for {column}', fontdict={'fontname': 'serif', 'color': 'blue', 'fontsize': 'large'})
    ax.set_xlabel(column, fontdict={'fontname': 'serif', 'color': 'darkred', 'fontsize': 'large'})
    ax.set_ylabel('Density', fontdict={'fontname': 'serif', 'color': 'darkred', 'fontsize': 'large'})
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%%

#%%

###   hexbin plots   ###

odata = pd.read_csv('/Users/surya/Documents/Class files/Data visualization/FTP/FTP(Surya)/finaldata.csv')
numeric_columns = odata.select_dtypes(include=['float64', 'int64']).columns.dropna()
for (column1, column2) in combinations(numeric_columns, 2):
    plt.figure(figsize=(10, 6))
    plt.hexbin(odata[column1], odata[column2], gridsize=30, cmap='Blues', bins='log')
    plt.colorbar(label='log10(N)')
    plt.title(f'Hexbin Plot of {column1} vs {column2}', fontdict={'fontname': 'serif', 'color': 'blue', 'fontsize': 'large'})
    plt.xlabel(column1, fontdict={'fontname': 'serif', 'color': 'darkred', 'fontsize': 'large'})
    plt.ylabel(column2, fontdict={'fontname': 'serif', 'color': 'darkred', 'fontsize': 'large'})
    plt.grid(True)
    plt.show()

#%%

#%%

###   STRIP PLOT   ###

odata = pd.read_csv('/Users/surya/Documents/Class files/Data visualization/FTP/FTP(Surya)/finaldata.csv')
numeric_columns = odata.select_dtypes(include=['float64', 'int64']).columns.dropna()
for column in numeric_columns:
    plt.figure(figsize=(10, 6))
    sns.stripplot(x=column, y=column, data=odata, jitter=0.25, palette='Set2', size=5)
    plt.title(f'Strip Plot of {column}', fontdict={'fontname': 'serif', 'color': 'blue', 'fontsize': 'large'})
    plt.xlabel('Index', fontdict={'fontname': 'serif', 'color': 'darkred', 'fontsize': 'large'})
    plt.ylabel('Value', fontdict={'fontname': 'serif', 'color': 'darkred', 'fontsize': 'large'})
    plt.grid(True)
    plt.show()

#%%
