# %% [markdown]
# # Fast Food Nutrition Analytics

# %% [markdown]
# ## Problem Statement
# Fast food chains lack data-driven understanding of how menu items cluster nutritionally across the industry, leading to inefficient product development and missed market opportunities. This analysis segments items by nutritional profiles to identify underserved markets, optimize menu strategy, and target health-conscious consumers effectively—ultimately improving R&D ROI and competitive positioning in the growing wellness-focused market.
# 

# %% [markdown]
# ## 1. Import libraries

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %% [markdown]
# ## 2. Load dataset

# %%
df = pd.read_csv("FOOD-DATA-GROUP5.csv")
df.head()

# %% [markdown]
# ## 3. Exploratory Data Analysis (EDA)

# %%
# Basic info

print("Shape", df.shape)
print("Columns", df.columns)
print("Missing Values", df.isnull().sum())
print("Duplicates", df.duplicated().sum())

# %%
df.drop(columns=["Unnamed: 0", "Unnamed: 0.1"], inplace=True)

# %%
print("Dataset Info:")
print(df.info())

# %%
print("\nDescriptive statistics:")
print(df.describe())

# %%
print("\nMissing values:")
print(df.isnull().sum())

# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_food_data(df):
    # Remove unnecessary columns
    df_clean = df.drop(['Unnamed: 0.1', 'Unnamed: 0'], axis=1)
    
    # Select nutritional features (excluding food name and potentially nutrition density)
    nutritional_features = [
        'Caloric Value', 'Fat', 'Saturated Fats', 'Monounsaturated Fats', 
        'Polyunsaturated Fats', 'Carbohydrates', 'Sugars', 'Protein', 
        'Dietary Fiber', 'Cholesterol', 'Sodium', 'Water', 'Vitamin A', 
        'Vitamin B1', 'Vitamin B11', 'Vitamin B12', 'Vitamin B2', 'Vitamin B3',
        'Vitamin B5', 'Vitamin B6', 'Vitamin C', 'Vitamin D', 'Vitamin E', 
        'Vitamin K', 'Calcium', 'Copper', 'Iron', 'Magnesium', 'Manganese',
        'Phosphorus', 'Potassium', 'Selenium', 'Zinc'
    ]
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_clean[nutritional_features])
    scaled_df = pd.DataFrame(scaled_features, columns=nutritional_features)
    scaled_df['food'] = df_clean['food']
    scaled_df['Nutrition Density'] = df_clean['Nutrition Density']
    
    return scaled_df, nutritional_features, df_clean

# Load and preprocess
df = pd.read_csv('FOOD-DATA-GROUP5.csv')  
scaled_df, nutritional_features, original_df = preprocess_food_data(df)

# %% [markdown]
# ## 4. Enhanced EDA for Comprehensive Nutrition Data

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Correlation matrix for key nutrients
key_nutrients = ['Caloric Value', 'Fat', 'Carbohydrates', 'Protein', 'Sugars', 
                 'Dietary Fiber', 'Sodium', 'Nutrition Density']
plt.figure(figsize=(12, 10))
sns.heatmap(original_df[key_nutrients].corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Key Nutritional Correlations')
plt.show()

# Nutrition density distribution
plt.figure(figsize=(10, 6))
plt.hist(original_df['Nutrition Density'], bins=30, alpha=0.7, edgecolor='black')
plt.xlabel('Nutrition Density')
plt.ylabel('Frequency')
plt.title('Distribution of Nutrition Density Across Foods')
plt.show()

# %% [markdown]
# ## 5. Clustering with Comprehensive Features

# %%
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

def perform_comprehensive_clustering(scaled_data, max_k=10):
    # Find optimal k
    wcss = []
    silhouette_scores = []
    
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(scaled_data)
        wcss.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(scaled_data, labels))
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(range(2, max_k + 1), wcss, marker='o')
    ax1.set_title('Elbow Method')
    ax1.set_xlabel('Number of Clusters')
    
    ax2.plot(range(2, max_k + 1), silhouette_scores, marker='o')
    ax2.set_title('Silhouette Scores')
    ax2.set_xlabel('Number of Clusters')
    
    plt.show()
    
    # Choose optimal k (you'll adjust based on plots)
    optimal_k = 5  # Start with 5, adjust based on plots
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    final_labels = kmeans_final.fit_predict(scaled_data)
    
    return final_labels, kmeans_final

# Perform clustering
cluster_labels, kmeans_model = perform_comprehensive_clustering(scaled_df[nutritional_features].values)

# %%
# Create and fit PCA model

pca = PCA(n_components=2)  # or n_components=3 for 3D visualization
principal_components = pca.fit_transform(scaled_df[nutritional_features].values)

print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.2%}")

# %% [markdown]
# ## 6. Saving All Models 

# %%
# Create models directory
import os
import joblib
import json
os.makedirs('models', exist_ok=True)

# 1. Save KMeans model
joblib.dump(kmeans_model, 'models/kmeans_clustering_model.pkl')
print("✅ Saved: kmeans_clustering_model.pkl")

# 2. Save PCA model
joblib.dump(pca, 'models/pca_model.pkl')
print("✅ Saved: pca_model.pkl")

# 3. Save StandardScaler (assuming you have 'scaler' from preprocessing)
# Note: If you haven't saved scaler yet, add this line after your preprocessing step
joblib.dump(StandardScaler, 'models/standard_scaler.pkl')
print("✅ Saved: standard_scaler.pkl")

# 4. Save feature names
with open('models/feature_names.json', 'w') as f:
    json.dump(nutritional_features, f)
print("✅ Saved: feature_names.json")

# 5. Save cluster centers
np.save('models/cluster_centers.npy', kmeans_model.cluster_centers_)
print("✅ Saved: cluster_centers.npy")

# 6. Save cluster labels for reference
cluster_labels_df = pd.DataFrame({
    'food': original_df['food'],
    'cluster': cluster_labels
})
cluster_labels_df.to_csv('models/cluster_labels.csv', index=False)
print("✅ Saved: cluster_labels.csv")

print("\n🎉 All models saved successfully in 'models/' directory!")

# %% [markdown]
# ## 7. Advanced Visualization

# %%
# First, make sure Cluster column is added to original_df
# Add this line BEFORE the violin plot code:

original_df['Cluster'] = cluster_labels  # Add clusters to dataframe

# Now create violin plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

nutrients_to_plot = ['Caloric Value', 'Protein', 'Fat', 'Carbohydrates', 'Sugars', 'Nutrition Density']

for i, nutrient in enumerate(nutrients_to_plot):
    data = [original_df[original_df['Cluster'] == cluster][nutrient] 
            for cluster in sorted(original_df['Cluster'].unique())]
    
    axes[i].violinplot(data, showmeans=True)
    axes[i].set_title(f'{nutrient} by Cluster')
    axes[i].set_xlabel('Cluster')
    axes[i].set_ylabel(nutrient)
    axes[i].set_xticks(range(1, len(data) + 1))
    axes[i].set_xticklabels([f'Cluster {c}' for c in sorted(original_df['Cluster'].unique())])

plt.tight_layout()
plt.show()

# %%
# Normalize for better visualization
cluster_means = original_df.groupby('Cluster')[key_nutrients].mean()
cluster_means_normalized = (cluster_means - cluster_means.mean()) / cluster_means.std()

plt.figure(figsize=(12, 8))
sns.heatmap(cluster_means_normalized.T, annot=True, cmap='RdBu_r', center=0,
            cbar_kws={'label': 'Standard Deviations from Mean'})
plt.title('Cluster Nutritional Profiles (Standardized)')
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(12, 8))

for cluster in sorted(original_df['Cluster'].unique()):
    cluster_data = original_df[original_df['Cluster'] == cluster]
    
    plt.scatter(cluster_data['Caloric Value'], 
                cluster_data['Nutrition Density'],
                s=cluster_data['Protein']*10,  # Bubble size represents protein
                alpha=0.6, label=f'Cluster {cluster}')

plt.xlabel('Caloric Value')
plt.ylabel('Nutrition Density')
plt.title('Nutritional Value vs Calories (Bubble Size = Protein Content)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# %%
# Calculate macronutrient percentages
macros = ['Protein', 'Fat', 'Carbohydrates']
cluster_macros = original_df.groupby('Cluster')[macros].mean()

# Convert to percentages
cluster_macros_pct = cluster_macros.div(cluster_macros.sum(axis=1), axis=0) * 100

cluster_macros_pct.plot(kind='bar', stacked=True, figsize=(12, 8),
                       color=['#2E86AB', '#A23B72', '#F18F01'])
plt.title('Macronutrient Composition by Cluster (%)')
plt.xlabel('Cluster')
plt.ylabel('Percentage')
plt.legend(title='Macronutrients')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# %%
# PCA Visualization with enhanced features
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_df[nutritional_features].values)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(principal_components[:, 0], principal_components[:, 1], 
                     c=cluster_labels, cmap='viridis', alpha=0.6)
plt.colorbar(scatter)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.title('Food Clustering by Comprehensive Nutritional Profiles')
plt.show()

# Add clusters to original data
original_df['Cluster'] = cluster_labels
scaled_df['Cluster'] = cluster_labels

# %% [markdown]
# ## 8. Cluster Profiling and Business Insights

# %%
# Analyze cluster characteristics
cluster_profiles = original_df.groupby('Cluster')[key_nutrients].mean()
print("Cluster Nutritional Profiles:")
print(cluster_profiles)

# Identify representative foods from each cluster
print("\nRepresentative Foods from Each Cluster:")
for cluster in sorted(original_df['Cluster'].unique()):
    cluster_foods = original_df[original_df['Cluster'] == cluster]['food'].head(5)
    print(f"\nCluster {cluster}:")
    for food in cluster_foods:
        print(f"  - {food}")

# Nutrition density by cluster
plt.figure(figsize=(10, 6))
original_df.boxplot(column='Nutrition Density', by='Cluster')
plt.title('Nutrition Density Distribution by Cluster')
plt.suptitle('')  # Remove automatic title
plt.show()

# %% [markdown]
# ## 9. Business Interpretation Framework

# %%
def interpret_clusters(cluster_data):
    """
    Provide business interpretation for each cluster
    """
    interpretations = {
        0: "High-Protein, Low-Carb Foods",
        1: "High-Carb, Moderate Protein", 
        2: "Balanced Nutrition Profile",
        3: "High-Nutrient Density Foods",
        4: "High-Calorie, High-Fat Items"
    }
    
    for cluster, interpretation in interpretations.items():
        if cluster in cluster_data['Cluster'].values:
            size = len(cluster_data[cluster_data['Cluster'] == cluster])
            avg_calories = cluster_data[cluster_data['Cluster'] == cluster]['Caloric Value'].mean()
            avg_density = cluster_data[cluster_data['Cluster'] == cluster]['Nutrition Density'].mean()
            
            print(f"\n📊 Cluster {cluster}: {interpretation}")
            print(f"   Size: {size} foods | Avg Calories: {avg_calories:.0f} | Avg Density: {avg_density:.2f}")
            
            # Show top 3 representative foods
            top_foods = cluster_data[cluster_data['Cluster'] == cluster]['food'].head(3).tolist()
            print(f"   Sample foods: {', '.join(top_foods)}")

interpret_clusters(original_df)

# %%
# Silhouette analysis
from sklearn.metrics import silhouette_samples

silhouette_vals = silhouette_samples(scaled_df[nutritional_features], cluster_labels)
original_df['Silhouette_Score'] = silhouette_vals

print(f"Overall Silhouette Score: {silhouette_vals.mean():.3f}")

# Check cluster quality
cluster_quality = original_df.groupby('Cluster')['Silhouette_Score'].mean()
print("\nCluster Quality (Avg Silhouette Score):")
print(cluster_quality)

# %%
# Save clustered dataset
original_df.to_csv('food_nutrition_clusters.csv', index=False)

# Save cluster profiles
cluster_summary = original_df.groupby('Cluster')[nutritional_features].agg(['mean', 'std'])
cluster_summary.to_csv('cluster_profiles_detailed.csv')

print("✅ Project completed! Files saved:")
print("   - food_nutrition_clusters.csv")
print("   - cluster_profiles_detailed.csv")
print("   - Business insights generated")

# %% [markdown]
# ---


