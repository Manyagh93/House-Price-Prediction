
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score
from sklearn.linear_model import LinearRegression

# Set page layout to wide
st.set_page_config(layout="wide", page_title="Mumbai Real Estate Analytics & Prediction")

# --- STEP 1: DATA PIPELINE ---
@st.cache_data
def load_and_clean_data():
    # Replace this with your actual file path
    # df = pd.read_csv('/content/mumbai_real_estate_dataset1.csv')
    
    # Mocking data structure based on your notebook for a self-contained execution
    np.random.seed(42)
    locations = ['Worli', 'Dadar', 'Juhu', 'Andheri', 'Bandra', 'Powai', 'Thane']
    mock_data = {
        'location': np.random.choice(locations, 1000),
        'size': np.random.choice(['2 BHK', '3 BHK', '4 BHK', '1 BHK'], 1000),
        'total_sqft': np.random.randint(400, 3500, 1000).astype(str),
        'bath': np.random.randint(1, 5, 1000),
        'balcony': np.random.randint(0, 4, 1000),
        'price': np.random.randint(40, 800, 1000),
        'area_type': 'Super Built-up Area', 'availability': 'Ready To Move', 'society': 'None'
    }
    df = pd.DataFrame(mock_data)
    
    # 1. Drop unnecessary columns & missing rows
    df = df.drop(['area_type', 'availability', 'society'], axis='columns', errors='ignore')
    df = df.dropna()
    
    # 2. Convert size to BHK (oda)
    df['bhk'] = df['size'].apply(lambda x: int(str(x).split(' ')[0]))
    df = df.drop(['size'], axis='columns', errors='ignore')
    
    # 3. Clean total_sqft ranges
    def convert_range_to_num(x):
        tokens = str(x).split('-')
        if len(tokens) == 2:
            return (float(tokens[0]) + float(tokens[1])) / 2
        try:
            return float(x)
        except:
            return None
            
    df['total_sqft'] = df['total_sqft'].apply(convert_range_to_num)
    df = df.dropna()
    
    # 4. Standardize locations
    df.location = df.location.apply(lambda x: str(x).strip().lower())
    loc_stats = df.groupby('location')['location'].size().sort_values(ascending=False)
    less_than_10 = loc_stats[loc_stats <= 10]
    df.location = df.location.apply(lambda x: 'other' if x in less_than_10 else x)
    
    # 5. Business logic outlier removal (Min 300 sqft per BHK)
    df = df[~(df['total_sqft'] / df['bhk'] < 300)]
    
    # 6. Price per m2 outlier removal
    df['metrekare'] = df['total_sqft'] * 0.092903
    df['price_per_m2'] = df['price'] * 1_000 / df['metrekare']
    
    def remove_ppm_outliers(dataframe):
        df_out = pd.DataFrame()
        for key, subdf in dataframe.groupby('location'):
            m = np.mean(subdf.price_per_m2)
            st_dev = np.std(subdf.price_per_m2)
            reduced_df = subdf[(subdf.price_per_m2 > (m - st_dev)) & (subdf.price_per_m2 <= (m + st_dev))]
            df_out = pd.concat([df_out, reduced_df], ignore_index=True)
        return df_out
        
    df = remove_ppm_outliers(df)
    
    # 7. Remove BHK outliers (where smaller BHK costs more per m2 than larger BHK in same area)
    def remove_bhk_outliers(dataframe):
        exclude_indices = np.array([])
        for location, location_df in dataframe.groupby('location'):
            bhk_stats = {}
            for bhk, bhk_df in location_df.groupby('bhk'):
                bhk_stats[bhk] = {
                    'mean': np.mean(bhk_df.price_per_m2),
                    'std': np.std(bhk_df.price_per_m2),
                    'count': bhk_df.shape[0]
                }
            for bhk, bhk_df in location_df.groupby('bhk'):
                stats = bhk_stats.get(bhk - 1)
                if stats and stats['count'] > 5:
                    exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_m2 < (stats['mean'])].index.values)
        return dataframe.drop(exclude_indices, axis='index')
        
    df = remove_bhk_outliers(df)
    
    # 8. Bathroom outlier removal
    df = df[df.bath < df.bhk + 2]
    
    # Final cleanup before feature engineering
    cleaned_df = df.drop(['balcony', 'price_per_m2', 'metrekare'], axis='columns', errors='ignore')
    return cleaned_df

df_clean = load_and_clean_data()

# --- STEP 2: MODEL TRAINING ---
# Creating dummy variables cleanly
df_dummies = pd.get_dummies(df_clean, columns=['location'], drop_first=True, prefix='location')

X = df_dummies.drop(['price'], axis='columns')
y = df_dummies['price']

# Extract all unique clean locations available for selection
unique_locations = sorted(df_clean['location'].unique())
# Identify the first location name that pandas dropped from dummy columns
dropped_location = "location_" + unique_locations[0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
model = LinearRegression()
model.fit(X_train.values, y_train)
accuracy = model.score(X_test.values, y_test)

# --- STEP 3: STREAMLIT UI ---
st.title("🏙️ Mumbai Real Estate Dashboard & Predictor")
st.markdown("Analyze property insights and accurately predict prices using Linear Regression modeling.")
st.markdown("---")

# Layout Split: Sidebar for inputs, Main panel for Tabs
sidebar = st.sidebar
sidebar.header("🔮 Price Predictor Inputs")

selected_loc = sidebar.selectbox("Select Location", [loc.title() for loc in unique_locations])
input_sqft = sidebar.number_input("Total Area (Square Feet)", min_value=300, max_value=10000, value=1200, step=50)
input_bhk = sidebar.slider("Number of BHK", min_value=1, max_value=8, value=2)
input_bath = sidebar.slider("Number of Bathrooms", min_value=1, max_value=8, value=2)

# Calculation Trigger
if sidebar.button("⚡ Predict Price"):
    # Target dummy column string pattern
    target_col = f"location_{selected_loc.lower().strip()}"
    
    # Initialize input vector array match X format
    x_input = np.zeros(len(X.columns))
    x_input[0] = input_sqft
    x_input[1] = input_bath
    x_input[2] = input_bhk
    
    # Check if target column exists in feature matrix space
    if target_col in X.columns:
        loc_index = X.columns.get_loc(target_col)
        x_input[loc_index] = 1
    # If it doesn't exist, it is the dropped category column: keeping features at 0 handles this natively.

    prediction = model.predict([x_input])[0]
    
    sidebar.success(f"### Predicted Price: ₹ {prediction:.2f} Lakhs")
    sidebar.caption(f"Model trained with R² accuracy score: **{accuracy*100:.2f}%**")

# --- STEP 4: VISUALIZATION TABS ---
tab1, tab2 = st.tabs(["📊 Data Visualizations", "📋 Cleaned Dataset Viewer"])

with tab1:
    st.subheader("Statistical Data Distribution Charts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # GRAPH 1: Price distribution Histogram
        st.markdown("#### 1. Property Valuation Spread")
...         fig1, ax1 = plt.subplots(figsize=(8, 5))
...         sns.histplot(df_clean['price'], kde=True, color='skyblue', bins=20, ax=ax1)
...         ax1.set_xlabel("Price (Lakh INR)")
...         ax1.set_ylabel("Count Density")
...         st.pyplot(fig1)
...         
...         # GRAPH 2: BHK vs Price Boxplot
...         st.markdown("#### 2. Price Distribution by Configuration Size (BHK)")
...         fig2, ax2 = plt.subplots(figsize=(8, 5))
...         sns.boxplot(x='bhk', y='price', data=df_clean, palette='Set2', ax=ax2)
...         ax2.set_xlabel("BHK Count")
...         ax2.set_ylabel("Price (Lakh INR)")
...         st.pyplot(fig2)
... 
...     with col2:
...         # GRAPH 3: Scatter Plot Sqft vs Price
...         st.markdown("#### 3. Correlation: Area (Sqft) vs Valuation")
...         fig3, ax3 = plt.subplots(figsize=(8, 5))
...         sns.scatterplot(x='total_sqft', y='price', hue='bhk', data=df_clean, palette='viridis', alpha=0.7, ax=ax3)
...         ax3.set_xlabel("Total Square Feet")
...         ax3.set_ylabel("Price (Lakh INR)")
...         st.pyplot(fig3)
...         
...         # GRAPH 4: Top Locations Chart
...         st.markdown("#### 4. Top Real Estate Sample Counts by Location")
...         fig4, ax4 = plt.subplots(figsize=(8, 5))
...         top_locs = df_clean['location'].value_counts().head(10)
...         sns.barplot(x=top_locs.values, y=[l.title() for l in top_locs.index], palette='plasma', ax=ax4)
...         ax4.set_xlabel("Number of Properties Listed")
...         ax4.set_ylabel("Location Hub")
...         st.pyplot(fig4)
... 
... with tab2:
...     st.subheader("Data Explorer")
...     st.markdown("This is the processed dataframe after handling null inputs, parsing categories, and stripping standard distribution outliers.")
