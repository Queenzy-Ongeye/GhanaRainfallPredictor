import pandas as pd
import matplotlib.pyplot as plt
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

            # Data types
            print(f"\nDATA TYPES")
            print("=" * 30)
            dtype_counts = self.df.dtypes.value_counts()
            for dtype, count in dtype_counts.items():
                print(f"  {dtype}: {count} columns")

            # checking for missing values
            print("\nMissing Values")
            missing_counts = self.df.isnull().sum().sort_values(ascending = False)
            missing_pct = (missing_counts / len(self.df) * 100).round(2)
            missing_df = pd.DataFrame({
                'Missing Count': missing_counts,
                'Missing %': missing_pct
            })
            print(missing_df)

            # Target variable analysis
            if 'Target' in self.df.columns:
                print(f"\nTARGET VARIABLE ANALYSIS")
                print("=" * 40)
                target_dist = self.df['Target'].value_counts().sort_index()
                print("Target distribution:")
                for target, count in target_dist.items():
                    pct = (count / len(self.df) * 100)
                    print(f"  {target}: {count:,} ({pct:.1f}%)")
                
                # Plot target distribution
                plt.figure(figsize=(10, 4))
                
                plt.subplot(1, 2, 1)
                target_dist.plot(kind='bar', color='skyblue')
                plt.title('Target Distribution (Count)')
                plt.xlabel('Rainfall Category')
                plt.ylabel('Count')
                plt.xticks(rotation=45)
                
                plt.subplot(1, 2, 2)
                target_dist.plot(kind='pie', autopct='%1.1f%%', colors=['lightcoral', 'skyblue', 'lightgreen', 'gold'])
                plt.title('Target Distribution (Percentage)')
                plt.ylabel('')
                
                plt.tight_layout()
                plt.show()

            # Analyze key categorical columns
            print(f"\nCATEGORICAL COLUMNS ANALYSIS")
            print("=" * 45)
            categorical_cols = ['indicator', 'community', 'district']
            
            for col in categorical_cols:
                if col in self.df.columns:
                    unique_vals = self.df[col].nunique()
                    total_vals = len(self.df[col].dropna())
                    print(f"\n{col.upper()}:")
                    print(f"  Unique values: {unique_vals}")
                    print(f"  Non-null values: {total_vals:,}")
                    
                    if unique_vals <= 20 and total_vals > 0:  # Show values if not too many
                        print("  Top values:")
                        value_counts = self.df[col].value_counts().head(10)
                        for val, count in value_counts.items():
                            pct = (count / total_vals * 100)
                            print(f"    {val}: {count:,} ({pct:.1f}%)")
            
            # User and community insights
            if 'user_id' in self.df.columns:
                print(f"\n FARMER PARTICIPATION ANALYSIS")
                print("=" * 45)
                user_counts = self.df['user_id'].value_counts()
                print(f"Total farmers: {len(user_counts)}")
                print(f"Average submissions per farmer: {user_counts.mean():.1f}")
                print(f"Most active farmer: {user_counts.iloc[0]} submissions")
                print(f"Farmers with 1 submission: {sum(user_counts == 1)}")
                print(f"Farmers with 10+ submissions: {sum(user_counts >= 10)}")
            
            if 'community' in self.df.columns:
                print(f"\nCOMMUNITY COVERAGE")
                print("=" * 30)
                community_counts = self.df['community'].value_counts()
                print(f"Total communities: {len(community_counts)}")
                print("Top 5 communities by submissions:")
                for community, count in community_counts.head().items():
                    pct = (count / len(self.df) * 100)
                    print(f"  {community}: {count:,} ({pct:.1f}%)")
            
            # Time analysis if prediction_time exists
            if 'prediction_time' in self.df.columns:
                print(f"\nTEMPORAL PATTERNS")
                print("=" * 30)
                
                # Convert to datetime
                temp_df = self.df.copy()
                temp_df['prediction_time'] = pd.to_datetime(temp_df['prediction_time'])
                temp_df['hour'] = temp_df['prediction_time'].dt.hour
                temp_df['day_of_week'] = temp_df['prediction_time'].dt.day_name()
                temp_df['month'] = temp_df['prediction_time'].dt.month
                
                # Time range
                print(f"Time range: {temp_df['prediction_time'].min()} to {temp_df['prediction_time'].max()}")
                
                # Hour analysis
                hour_dist = temp_df['hour'].value_counts().sort_index()
                print(f"Peak submission hour: {hour_dist.idxmax()}:00 ({hour_dist.max()} submissions)")
                
                # Create time analysis plots
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                
                # Hour distribution
                hour_dist.plot(kind='bar', ax=axes[0,0], color='lightblue')
                axes[0,0].set_title('Submissions by Hour of Day')
                axes[0,0].set_xlabel('Hour')
                axes[0,0].set_ylabel('Count')
                
                # Day of week
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                day_dist = temp_df['day_of_week'].value_counts().reindex(day_order)
                day_dist.plot(kind='bar', ax=axes[0,1], color='lightgreen')
                axes[0,1].set_title('Submissions by Day of Week')
                axes[0,1].set_xlabel('Day')
                axes[0,1].set_ylabel('Count')
                axes[0,1].tick_params(axis='x', rotation=45)
                
                # Monthly distribution
                month_dist = temp_df['month'].value_counts().sort_index()
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                month_dist.index = [month_names[i-1] for i in month_dist.index]
                month_dist.plot(kind='bar', ax=axes[1,0], color='lightcoral')
                axes[1,0].set_title('Submissions by Month')
                axes[1,0].set_xlabel('Month')
                axes[1,0].set_ylabel('Count')
                axes[1,0].tick_params(axis='x', rotation=45)
                
                # Target by hour (if Target exists)
                if 'Target' in temp_df.columns:
                    target_hour = temp_df.groupby(['hour', 'Target']).size().unstack(fill_value=0)
                    target_hour.plot(kind='bar', stacked=True, ax=axes[1,1])
                    axes[1,1].set_title('Rainfall Predictions by Hour')
                    axes[1,1].set_xlabel('Hour')
                    axes[1,1].set_ylabel('Count')
                    axes[1,1].legend(title='Rainfall Type')
                
                plt.tight_layout()
                plt.show()
            
            # First and last few rows
            print(f"\n SAMPLE DATA")
            print("=" * 20)
            print("First 3 rows:")
            print(self.df.head(3))
            
            print(f"\nData exploration completed!")
            print(f"Summary: {self.df.shape[0]:,} records from {self.df.get('community', pd.Series()).nunique()} communities")
            
            return self.df
            
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found!")
            print("Please check the file path and try again.")
            return None
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    # Preprocessing data
    def preprocess_data(self, df):
        processed_df = df.copy()
        print("=== Preprocess Ghana Rainfall Data ===")

        # 1. Dropping columns with excessive missing values
        missing_threshold = 0.9
        to_drop = []

        for col in processed_df.columns:
            pct  = processed_df[col].isna().mean()
            if pct > missing_threshold:
                to_drop.append(col)
                print(f"Dropping {col}: {pct: .1%} missing values")
        
        if to_drop:
            processed_df = processed_df.drop(columns = to_drop, errors = 'ignore')
        
        # 2. Indicator missing handling
        if 'indicator' in processed_df.columns:
            processed_df['indicator'] = processed_df['indicator'].fillna("no_indicator")

        # 3. Time features
        if 'prediction_time' in processed_df.columns:
            processed_df['prediction_time'] = pd.to_datetime(
                processed_df['prediction_time'], errors = 'coerce'
            )
            processed_df['hour'] = processed_df['prediction_time'].dt.hour
            processed_df['day_of_week'] = processed_df['prediction_time'].dt.day_of_week
            processed_df['month'] = processed_df['prediction_time'].dt.month
            processed_df['is_weekend'] = processed_df['prediction_time'].isin([5, 6]).astype(int)

            # time flags
            processed_df['is_morning'] = processed_df['hour'].between(6, 11, inclusive = 'both').astype(int)
            processed_df['is_afternoon'] = processed_df['hour'].between(12, 17, inclusive = 'both').astype(int)
            processed_df['is_evening'] = processed_df['hour'].between(18, 21, inclusive = 'both').astype(int)

            # night wraps around midningt
            processed_df['is_night'] = ((processed_df['hour'] >= 22) | (processed_df['hour'] < 6)).astype(int)

        # 4. Farmer experience proxy
        if 'user_id' in processed_df.columns:
            user_counts = processed_df['user_id'].value_counts()
            processed_df['farmer_experience_level'] = processed_df['user_id'].map(user_counts).fillna(0).astype(int)
            q75 = processed_df['farmer_experience_level'].quantile(0.75)
            processed_df['is_experienced_farmer'] = (processed_df['farmer_experience_level'] >= q75).astype(int)

        # 5. Community/ District aggregates
        if {'community', 'confidence'}.issubset(processed_df.columns):
            comm_avg = processed_df.groupby('community')['confidence'].mean()
            processed_df['community_avg_confidence'] = processed_df['community'].map(comm_avg)

        if {'district', 'confidence'}.issubset(processed_df.columns):
            dist_avg = processed_df.groupby('district')['confidence'].mean()
            processed_df['district_avg_confidence'] = processed_df['district'].map(dist_avg)
        
        # 6. Indicator availability flag
        if 'indicator' in processed_df.columns:
            processed_df['has_indicator'] = (processed_df['indicator'] != 'no_indicator').astype(int)
        
        # 7) Assemble feature lists (use only columns that exist)
        categorical_features = ['indicator', 'community', 'district', 'Target']  # 'Target' capitalized
        if 'day_of_week' in processed_df.columns:
            categorical_features += ['day_of_week', 'month']
        categorical_features = [c for c in categorical_features if c in processed_df.columns]

        numerical_features = [
            'farmer_experience_level', 'community_avg_confidence', 'district_avg_confidence',
            'predicted_intensity', 'confidence', 'is_weekend', 'is_experienced_farmer', 'forecast_length',
            'has_indicator', 'hour', 'is_morning', 'is_afternoon', 'is_evening', 'is_night'
        ]
        numerical_features = [n for n in numerical_features if n in processed_df.columns]

        print("After dropping sparse columns:")
        print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
        print(f"Numerical features   ({len(numerical_features)}): {numerical_features}")

        # 8 Save into the instance for downstream steps AND return it
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.processed_df = processed_df
        return processed_df

   