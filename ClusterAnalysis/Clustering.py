import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Title of the app
st.title("Cluster Analysis App")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file
    data = pd.read_csv(uploaded_file)
    
    # Display the data
    st.write("Data Preview:")
    st.write(data.head())
    
    # Select columns for clustering
    numerical_columns = st.multiselect("Select numerical columns for clustering", data.select_dtypes(include=['float64', 'int64']).columns)
    categorical_columns = st.multiselect("Select categorical columns for clustering", data.select_dtypes(include=['object']).columns)
    
    if len(numerical_columns) > 0 or len(categorical_columns) > 0:
        # Handle categorical data
        if len(categorical_columns) > 0:
            le_dict = {}
            for col in categorical_columns:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
                le_dict[col] = le
        
        # Combine selected columns
        selected_columns = numerical_columns + categorical_columns
        
        # Standardize numerical data
        if len(numerical_columns) > 0:
            scaler = StandardScaler()
            data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
        
        # Determine clustering algorithm to use
        if len(numerical_columns) > 0 and len(categorical_columns) == 0:
            # Use KMeans for numerical data only
            st.write("Using KMeans for clustering")
            num_clusters = st.slider("Select number of clusters", 2, 10, 3)
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            clusters = kmeans.fit_predict(data[selected_columns])
        
        elif len(numerical_columns) == 0  and len(categorical_columns) > 1:
            # Use KModes for categorical data only
            st.write("Using KModes for clustering")
            num_clusters = st.slider("Select number of clusters", 2, 10, 3)
            kmodes = KModes(n_clusters=num_clusters, random_state=42)
            clusters = kmodes.fit_predict(data[selected_columns])
        
        elif len(numerical_columns) > 0 and len(categorical_columns) > 0:
            # Use KPrototypes for mixed data types
            st.write("Using KPrototypes for clustering")
            num_clusters = st.slider("Select number of clusters", 2, 10, 3)
            kprototypes = KPrototypes(n_clusters=num_clusters, random_state=42)
            clusters = kprototypes.fit_predict(data[selected_columns], categorical=[selected_columns.index(col) for col in categorical_columns])
        
        else:
            st.error("Please select at least one numerical or categorical column for clustering.")
        
        # Add cluster labels to the data
        data['Cluster'] = clusters + 1
        
        # Display the clustered data
        st.write("Clustered Data:")
        st.write(data)
        
        # Plot the clusters (if there are at least two numerical columns to plot)
        if len(numerical_columns) >= 2:
            st.write("Cluster Plot:")
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=data[numerical_columns[0]], y=data[numerical_columns[1]], hue=data['Cluster'], palette='viridis')
            plt.title("Clusters")
            plt.xlabel(numerical_columns[0])
            plt.ylabel(numerical_columns[1])
            st.pyplot(plt)

# Instructions for the user
st.write("""
## Instructions:
1. Upload a CSV file containing your data.
2. Select the numerical and/or categorical columns you want to use for clustering.
3. Choose the number of clusters.
4. View the clustered data and cluster plot.
""")
