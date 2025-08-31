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

        # Handling missing indicators
        processed_df['indicator'] = processed_df['indicator'].fillna('no_indicator')
        processed_df['indicator_desc'] = processed_df['indicator_desc'].fillna('no_description')

        # Handling time_observed with high missing values
        processed_df['time_observed'] = processed_df['time_observed'].fillna('not_specified')

        # convert prediction_time to datetime features
        if 'prediction_time' in processed_df.columns:
            processed_df['predicted_time'] = pd.to_datetime(processed_df['prediction_time'])
            processed_df['hour'] = processed_df['prediction_time'].dt.hour
            processed_df['day_of_week'] = processed_df['prediction_time'].dt.dayofweek
            processed_df['month'] = processed_df['prediction_time'].dt.month
            processed_df['is_weekend'] = processed_df['day_of_week'].isin([5, 6]).astype(int)

            # creating a time-based feature for agricultural context
            processed_df['is_morning'] = (processed_df['hour'] >= 6) & (processed_df['hour'] < 12)
            processed_df

