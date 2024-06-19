import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
covid_19_data = pd.read_csv('covid_19_data (1).csv')
covid19_line_list_data = pd.read_csv('COVID19_line_list_data_modified (1).csv')

# Data Cleaning and Preprocessing


def preprocess_data():
    # Drop unnecessary columns
    covid_19_data.drop(columns=['SNo', 'Last Update'], inplace=True)
    covid19_line_list_data.drop(columns=['Unnamed: 3'], inplace=True)

    # Fill missing values
    covid_19_data['Province/State'].fillna('Unknown', inplace=True)
    covid19_line_list_data.fillna({'case_in_country': 0, 'gender': 'unknown', 'age': 0,
                                   'visiting Wuhan': 0, 'from Wuhan': 0, 'death': 0,
                                   'recovered': 0, 'symptom': 'unknown'}, inplace=True)

    # Convert data types
    covid_19_data['ObservationDate'] = pd.to_datetime(
        covid_19_data['ObservationDate'])
    covid_19_data[['Confirmed', 'Deaths', 'Recovered']] = covid_19_data[[
        'Confirmed', 'Deaths', 'Recovered']].astype(int)

    date_columns = ['reporting date', 'symptom_onset',
                    'hosp_visit_date', 'exposure_start', 'exposure_end']
    for col in date_columns:
        covid19_line_list_data[col] = pd.to_datetime(
            covid19_line_list_data[col], errors='coerce')

    numeric_columns = ['case_in_country', 'age',
                       'visiting Wuhan', 'from Wuhan', 'death', 'recovered']
    covid19_line_list_data[numeric_columns] = covid19_line_list_data[numeric_columns].apply(
        pd.to_numeric, errors='coerce')

    # Define age groups
    age_bins = [0, 18, 30, 50, 70, 100]
    age_labels = ['0-18', '19-30', '31-50', '51-70', '71+']
    covid19_line_list_data['age_group'] = pd.cut(
        covid19_line_list_data['age'], bins=age_bins, labels=age_labels, right=False)


preprocess_data()

# Highest and Second Highest Affected Areas


def get_highest_affected_areas():
    country_data = covid_19_data.groupby(
        'Country/Region').agg({'Confirmed': 'sum'}).reset_index()
    highest_affected_area = country_data.sort_values(
        by='Confirmed', ascending=False).iloc[0]
    second_highest_affected_area = country_data.sort_values(
        by='Confirmed', ascending=False).iloc[1]
    return highest_affected_area, second_highest_affected_area


highest_affected_area, second_highest_affected_area = get_highest_affected_areas()

# Mortality vs Recovery Ratio


def get_mortality_recovery_ratio():
    total_deaths = covid_19_data['Deaths'].sum()
    total_recovered = covid_19_data['Recovered'].sum()
    mortality_recovery_ratio = total_deaths / \
        total_recovered if total_recovered != 0 else np.nan
    return mortality_recovery_ratio


mortality_recovery_ratio = get_mortality_recovery_ratio()

# Age and Gender Distribution


def plot_age_gender_distribution():
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    sns.histplot(covid19_line_list_data['age'], bins=30, kde=True, ax=ax[0])
    ax[0].set_title('Age Distribution')
    ax[0].set_xlabel('Age')
    ax[0].set_ylabel('Frequency')

    gender_counts = covid19_line_list_data['gender'].value_counts()
    sns.barplot(x=gender_counts.index, y=gender_counts.values, ax=ax[1])
    ax[1].set_title('Gender Distribution')
    ax[1].set_xlabel('Gender')
    ax[1].set_ylabel('Frequency')

    return fig

# Mortality Rate Among Different Age Groups


def get_age_group_mortality():
    age_group_mortality = covid19_line_list_data.groupby('age_group').agg(
        {'death': 'sum', 'case_in_country': 'count'}).reset_index()
    age_group_mortality['age_group'] = age_group_mortality['age_group'].astype(
        str)
    age_group_mortality['mortality_rate'] = age_group_mortality['death'] / \
        age_group_mortality['case_in_country']
    return age_group_mortality


age_group_mortality = get_age_group_mortality()

# Streamlit Interface
st.title("COVID-19 Data Analysis")

# Highest and Second Highest Affected Areas
st.header("Highest and Second Highest Affected Areas")
st.write(
    f"Highest affected area: {highest_affected_area['Country/Region']} with {highest_affected_area['Confirmed']} confirmed cases")
st.write(
    f"Second highest affected area: {second_highest_affected_area['Country/Region']} with {second_highest_affected_area['Confirmed']} confirmed cases")

# Mortality vs Recovery Ratio
st.header("Mortality vs Recovery Ratio")
st.write(f"Mortality vs. Recovery Ratio: {mortality_recovery_ratio}")

# Age and Gender Distribution
st.header("Age and Gender Distribution")
fig = plot_age_gender_distribution()
st.pyplot(fig)

# Mortality rate among different age groups
st.header("Mortality Rate Among Different Age Groups")
st.write(age_group_mortality[['age_group', 'mortality_rate']])
