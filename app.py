import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import re
import chardet


def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']


# Define file paths
csv_file1_path = 'Movies.csv'
csv_file2_path = 'Ratings.csv'
csv_file3_path = 'Users.csv'

# Detect file encodings
encoding1 = detect_encoding(csv_file1_path)
encoding2 = detect_encoding(csv_file2_path)
encoding3 = detect_encoding(csv_file3_path)

# Load data with detected encodings
df1 = pd.read_csv(csv_file1_path, encoding=encoding1)
df2 = pd.read_csv(csv_file2_path, encoding=encoding2)
df3 = pd.read_csv(csv_file3_path, encoding=encoding3)

# Clean data
df1.drop_duplicates(inplace=True)
df2.drop_duplicates(inplace=True)
df3.drop_duplicates(inplace=True)

df1.fillna('', inplace=True)
df2.fillna(0, inplace=True)
df3.fillna('', inplace=True)

# Extract Year of Release from Title
df1['YearOfRelease'] = df1['Title'].apply(lambda x: re.search(
    r'\((\d{4})\)', x).group(1) if re.search(r'\((\d{4})\)', x) else None)
movies_per_year = df1.groupby('YearOfRelease').size().reset_index(name='Count')

# Define age groups
bins = [0, 18, 25, 35, 45, 50, 56, 100]
labels = ['0-18', '19-25', '26-35', '36-45', '46-50', '51-56', '57+']
df3['AgeGroup'] = pd.cut(df3['Age'], bins, labels=labels)

# Merge datasets
merged_df = df2.merge(df1, on='MovieID').merge(df3, on='UserID')

# Find highest rated category per year
highest_rated_per_year = merged_df.groupby(['YearOfRelease', 'Category'])[
    'Rating'].mean().reset_index()
highest_rated_per_year = highest_rated_per_year.sort_values(['YearOfRelease', 'Rating'], ascending=[
                                                            True, False]).groupby('YearOfRelease').first().reset_index()

# Find the number of users from each age group who rated each category
age_group_likings = merged_df.groupby(
    ['Category', 'AgeGroup']).size().reset_index(name='Count')

# Cluster models to segregate movie category and age group wise likings
category_agegroup_counts = merged_df.groupby(
    ['Category', 'AgeGroup']).size().unstack(fill_value=0)
X_agegroup = category_agegroup_counts.values

# Apply KMeans clustering
kmeans_agegroup = KMeans(n_clusters=5, random_state=0).fit(X_agegroup)
category_agegroup_counts['Cluster'] = kmeans_agegroup.labels_

# Clustering methods to segregate movie category and occupation
category_occupation_counts = merged_df.groupby(
    ['Category', 'Occupation']).size().unstack(fill_value=0)
X_occupation = category_occupation_counts.values

# Apply KMeans clustering
kmeans_occupation = KMeans(n_clusters=5, random_state=0).fit(X_occupation)
category_occupation_counts['Cluster'] = kmeans_occupation.labels_

# Refine the model by including age group
category_agegroup_occupation_counts = merged_df.groupby(
    ['Category', 'AgeGroup', 'Occupation']).size().unstack(fill_value=0).fillna(0)
X_combined = category_agegroup_occupation_counts.values

# Apply KMeans clustering
kmeans_combined = KMeans(n_clusters=5, random_state=0).fit(X_combined)
category_agegroup_occupation_counts['Cluster'] = kmeans_combined.labels_

# Predictive model based on category, age group, and occupation
category_agegroup_occupation_counts = merged_df.groupby(
    ['Category', 'AgeGroup', 'Occupation']).size().reset_index(name='Count')
X_pred = category_agegroup_occupation_counts[[
    'Category', 'AgeGroup', 'Occupation']]
y_pred = category_agegroup_occupation_counts['Count']

X_pred = pd.get_dummies(X_pred)

# Use a smaller subset of the data for training
X_train, X_test, y_train, y_test = train_test_split(
    X_pred, y_pred, test_size=0.9, random_state=0)

model = RandomForestClassifier(n_estimators=50, random_state=0)
model.fit(X_train, y_train)

y_pred_model = model.predict(X_test)

# Streamlit UI
st.title("Movie Analysis and Prediction System")

# Query 1: Total number of movies released in each year
st.header("Total number of movies released in each year")
st.dataframe(movies_per_year)

# Query 2: Movie category having highest ratings in each year
st.header("Movie category having highest ratings in each year")
st.dataframe(highest_rated_per_year)

# Query 3: Movie category and age group wise likings
st.header("Movie category and age group wise likings")
st.dataframe(age_group_likings)

# Query 4: Cluster models to segregate movie category and age group wise likings
st.header("Cluster models to segregate movie category and age group wise likings")
st.dataframe(category_agegroup_counts)

# Query 5: Year wise count of movies released
st.header("Year wise count of movies released")
st.dataframe(movies_per_year)

# Query 6: Year wise, category wise count of movies released
st.header("Year wise, category wise count of movies released")
movies_per_year_category = df1.groupby(
    ['YearOfRelease', 'Category']).size().reset_index(name='Count')
st.dataframe(movies_per_year_category)

# Query 7: Clustering methods to segregate movie category and occupation
st.header("Clustering methods to segregate movie category and occupation")
st.dataframe(category_occupation_counts)

# Query 8: Refine the model by including age group
st.header("Refined model by including age group and occupation")
st.dataframe(category_agegroup_occupation_counts)

# Query 9: Predictive model based on category, age group, and occupation
st.header("Predictive model based on category, age group, and occupation")
st.write('Model Accuracy:', accuracy_score(y_test, y_pred_model))

# User inputs for prediction
st.header("Predict movie likings based on user inputs")
category_input = st.selectbox("Select Category", df1['Category'].unique())
agegroup_input = st.selectbox("Select Age Group", labels)
occupation_input = st.selectbox("Select Occupation", range(21))

if st.button("Predict"):
    user_input = pd.DataFrame({
        'Category': [category_input],
        'AgeGroup': [agegroup_input],
        'Occupation': [occupation_input]
    })
    user_input = pd.get_dummies(user_input)

    # Ensure all columns in training data are in user input
    for col in X_train.columns:
        if col not in user_input.columns:
            user_input[col] = 0

    # Reorder columns to match the training set
    user_input = user_input[X_train.columns]

    prediction = model.predict(user_input)
    st.write(f'Predicted number of users: {prediction[0]}')
