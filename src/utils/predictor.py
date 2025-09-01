import pandas as pd
class GhanaRainfallPredictor:
    """
    A machine learning pipeline for predicting rainfall intensity (heavy/moderate/small)
    based on indigenous ecological indicators from Ghana's Pra River Basin.
    """

    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.label_encoder = None
        self.feature_names = None
        self.performance_metrics = {}
    
    # Loading and data explorinf function
    def load_and_explore_dta(self, file_path):
        try:
            self.df = pd.read_csv(file_path)
            print("Dataset loaded Successfully")
            print(f"Shape: {self.df.shape}")
            print("\nColumns:", self.df.columns.tolist())
            print(self.df.head())

            # Basic Statistics
            print("\nDataset Info:")
            print(self.df.info())

            # checking for missing values
            print("\nMissing Values")
            missing_counts = self.df.isnull().sum().sort_values(ascending = False)
            missing_pct = (missing_counts / len(self.df) * 100).round(2)
            missing_df = pd.DataFrame({
                'Missing Count': missing_counts,
                'Missing %': missing_pct
            })
            print(missing_df)

            # analyzing target distribution
            if 'Target' in self.df.columns:
                print(f"\nTarget distribution:")
                target_dist = self.df['Target'].value_counts()
                print(target_dist)
                print(f"Target percentges: ")
                print((target_dist / len(self.df) *100).round(2))

            # Analyzing the key categorical columns
            key_cols = ['indicator', 'predicted_intensity', 'confidence', 'community', 'district']
            for col in key_cols:
                if col in self.df.columns:
                    unique_vals = self.df[col].nunique()
                    print(f"\n{col}: {unique_vals} unique values")
                    if  unique_vals <= 20:
                        print(self.df[col].value_counts().head(10))
            return self.df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    # Preprocessing data
    def preprocess_data(self, df):
        processed_df = df.copy()
        print("=== Preprocessing Ghana Rainfall Data ===")

        # Drop columns with excessive missing values (>90%)
        columns_to_drop = []
        missing_threshold = 0.9  # 90% threshold
        
        for col in df.columns:
            missing_pct = df[col].isnull().sum() / len(df)
            if missing_pct > missing_threshold:
                columns_to_drop.append(col)
                print(f"Dropping {col}: {missing_pct:.1%} missing values")
        
        processed_df = processed_df.drop(columns = columns_to_drop)

        # Handling missing indicators
        if 'indicator' in processed_df.columns:
            processed_df['indicator'] = processed_df['indicator'].fillna('no_indicator')

        # convert prediction_time to datetime features
        if 'prediction_time' in processed_df.columns:
            processed_df['predicted_time'] = pd.to_datetime(processed_df['prediction_time'])
            processed_df['hour'] = processed_df['prediction_time'].dt.hour
            processed_df['day_of_week'] = processed_df['prediction_time'].dt.dayofweek
            processed_df['month'] = processed_df['prediction_time'].dt.month
            processed_df['is_weekend'] = processed_df['day_of_week'].isin([5, 6]).astype(int)

            # creating a time-based feature for agricultural context
            processed_df['is_morning'] = (processed_df['hour'] >= 6) & (processed_df['hour'] < 12)
            processed_df['is_afternoon'] = (processed_df['hour'] >= 12)  & (processed_df['hour'] < 18)
            processed_df['is_evening'] = (processed_df['hour'] >= 18) & (processed_df['hour'] < 22)
            processed_df['is_night'] = (processed_df['hour'] >= 22) & (processed_df['hour'] < 6)
        
        # creating farmer experience proxy based 
        user_submission_counts = processed_df['user_id'].value_counts()
        processed_df['farmer_experience_level'] = processed_df['user_id'].map(user_submission_counts)
        processed_df['is_experienced_farmer'] = (processed_df['farmer_experience_level'] >= processed_df['farmer_experience_level'].quantile(0.75)).astype(int)

        # creating a community-based feature (local knowledge patterns)
        community_avg_confidence = processed_df.groupby('community')['confidence'].mean()
        processed_df['community_avg_confidence'] = processed_df['community'].map(community_avg_confidence)

        # creating a district based feature
        district_avg_confidence = processed_df.groupby('district')['confidence'].mean()
        processed_df['district_avg_confidence'] = processed_df['district'].map(district_avg_confidence)

        # creating an indicator availability flag
        if 'indicator' in processed_df.columns:
            processed_df['has_indicator'] = (processed_df['indicator'] != 'no_indicator').astype(int)
        
        # Defining categorical and numerical features
        categorical_features = ['indicator', 'community', 'district', 'target']

        # Adding time-based categorical features
        if 'day_of_week' in processed_df.columns:
            categorical_features.extend(['day_of_week', 'month'])
        
        # Defining numerical features
        numerical_features = [
            'farmer_experience_level', 'community_avg_confidence', 'district_avg_confidence',
            'prediction_intensity', 'confidence', 'is_weekend', 'is_experienced_farmer', 'forecast_length'
        ]

        # adding indicator flag if it exists
        if 'has_indicator' in processed_df.columns:
            numerical_features.append('has_indicator')
        
        # Adding time features
        time_features = ['hour', 'is_morning', 'is_afternoon', 'is_evening', 'is_night']
        for feat in time_features:
            if feat in processed_df.columns:
                numerical_features.append(feat)

        # filter features that exist in the dataset
        categorical_features = [f for f in categorical_features if f in processed_df.columns]
        numerical_features = [f for f in numerical_features if f in processed_df.columns]

        print(f"After dropping sparse columns: ")
        print(f"Categorical feature ({len(categorical_features)}): {categorical_features}")
        print(f"Numerical feature ({len(numerical_features)}): {numerical_features}")

        # storing features in a list
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features

        # 

