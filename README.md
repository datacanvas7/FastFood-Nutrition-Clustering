# 🍔 Fast Food Nutrition Clustering

## 📌 Overview
Unsupervised machine learning project that segments 722 food items into distinct nutritional clusters using 37 nutritional features (macros, vitamins, minerals). Discovers hidden patterns in food composition to optimize product portfolio and target health-conscious consumers.

## 📊 Clustering Performance Results

| Metric | Score |
|--------|-------|
| **Optimal Clusters** | 5 segments |
| **Silhouette Score** | 0.52 |
| **Inertia (WCSS)** | 1850.23 |
| **Total Variance Explained (PCA)** | 67.8% |

## 🏆 Cluster Profiles Discovered

| Cluster | Size | Key Characteristics | Avg Calories | Avg Protein | Avg Nutrition Density |
|---------|------|---------------------|--------------|-------------|----------------------|
| **Cluster 0** | 156 foods | High-Protein, Low-Carb | 245 | 24g | 0.72 |
| **Cluster 1** | 189 foods | High-Carb, Moderate Protein | 310 | 12g | 0.58 |
| **Cluster 2** | 142 foods | Balanced Nutrition | 278 | 18g | 0.68 |
| **Cluster 3** | 98 foods | Nutrient-Dense (High Vitamins) | 198 | 15g | 0.85 |
| **Cluster 4** | 137 foods | High-Calorie, High-Fat (Indulgence) | 520 | 20g | 0.42 |

## 🔍 Key Insights

### Most Important Differentiating Features:
- **Caloric Value** - Primary cluster differentiator (35% variance explained)
- **Fat & Saturated Fats** - Separates indulgence from healthy segments
- **Protein Content** - Distinguishes fitness-focused foods
- **Vitamin Density** - Clusters 3 uniquely high in micronutrients
- **Sugar Content** - Key differentiator for breakfast vs dessert items

### Surprising Discovery:
> 23% of "salad" items clustered with high-calorie foods due to heavy dressings, revealing a marketing vs. reality gap.

## 🎯 Business Impact

| Area | Impact |
|------|--------|
| **Product Development** | Identifies 2 underserved nutritional segments (high-protein breakfast, nutrient-dense snacks) |
| **Marketing Optimization** | Enables segment-specific targeting (fitness, wellness, indulgence) |
| **Menu Rationalization** | 47 redundant items identified across clusters, potential 15% SKU reduction |
| **Competitive Positioning** | Gap analysis reveals opportunity in high-protein, low-carb segment (+32% growth potential) |

---
## 📁 Project Structure
