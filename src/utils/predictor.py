import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                           f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.feature_selection import SelectKBest, chi2, f_classif, RFE
from sklearn.model_selection import (train_test_split, cross_val_score, StratifiedKFold, 
                                   GridSearchCV, RandomizedSearchCV)
from sklearn.calibration import CalibratedClassifierCV
import warnings
from scipy.stats import randint, uniform

warnings.filterwarnings('ignore')

class EnhancedGhanaRainfallPredictor:
    """
    Enhanced ML pipeline for predicting rainfall intensity with focus on macro-F1 optimization
    """

    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.label_encoder = None
        self.feature_names = None
        self.performance_metrics = {}
        self.best_params = {}
    
    def EnhancedGhanaRainfallPredictor(self, file_path):
        """Load and explore the dataset with comprehensive analysis"""
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

            # Checking for missing values
            print("\nMissing Values")
            missing_counts = self.df.isnull().sum().sort_values(ascending=False)
            missing_pct = (missing_counts / len(self.df) * 100).round(2)
            missing_df = pd.DataFrame({
                'Missing Count': missing_counts,
                'Missing %': missing_pct
            })
            print(missing_df[missing_df['Missing Count'] > 0])

            # Target variable analysis
            if 'Target' in self.df.columns:
                print(f"\nTARGET VARIABLE ANALYSIS")
                print("=" * 40)
                target_dist = self.df['Target'].value_counts().sort_index()
                print("Target distribution:")
                total_samples = len(self.df)
                
                for target, count in target_dist.items():
                    pct = (count / total_samples * 100)
                    print(f"  {target}: {count:,} ({pct:.1f}%)")
                
                # Calculate class imbalance ratio
                max_class = target_dist.max()
                min_class = target_dist.min()
                imbalance_ratio = max_class / min_class
                print(f"\nClass imbalance ratio: {imbalance_ratio:.2f}")
                
                if imbalance_ratio > 3:
                    print("⚠️ Significant class imbalance detected - will need special handling")
                
                # Visualize target distribution
                plt.figure(figsize=(12, 5))
                
                plt.subplot(1, 3, 1)
                target_dist.plot(kind='bar', color='skyblue', alpha=0.8)
                plt.title('Target Distribution (Count)')
                plt.xlabel('Rainfall Category')
                plt.ylabel('Count')
                plt.xticks(rotation=45)
                
                plt.subplot(1, 3, 2)
                target_dist.plot(kind='pie', autopct='%1.1f%%', 
                                colors=['lightcoral', 'skyblue', 'lightgreen', 'gold'])
                plt.title('Target Distribution (Percentage)')
                plt.ylabel('')
                
                # Class balance visualization
                plt.subplot(1, 3, 3)
                ratios = target_dist / target_dist.min()
                ratios.plot(kind='bar', color='orange', alpha=0.7)
                plt.title('Class Balance Ratios')
                plt.xlabel('Rainfall Category')
                plt.ylabel('Ratio to Minority Class')
                plt.xticks(rotation=45)
                
                plt.tight_layout()
                plt.show()

            return self.df
            
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found!")
            return None
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def preprocess_data(self, df):
        """Enhanced preprocessing with feature engineering"""
        processed_df = df.copy()
        print("=== Enhanced Preprocessing Ghana Rainfall Data ===")

        # 1. Handle missing values more strategically
        missing_threshold = 0.7  # More conservative threshold
        to_drop = []

        for col in processed_df.columns:
            pct = processed_df[col].isna().mean()
            if pct > missing_threshold:
                to_drop.append(col)
                print(f"Dropping {col}: {pct:.1%} missing values")
        
        if to_drop:
            processed_df = processed_df.drop(columns=to_drop, errors='ignore')
        
        # 2. Enhanced indicator handling
        if 'indicator' in processed_df.columns:
            processed_df['indicator'] = processed_df['indicator'].fillna("no_indicator")
            # Create indicator frequency encoding
            indicator_counts = processed_df['indicator'].value_counts()
            processed_df['indicator_frequency'] = processed_df['indicator'].map(indicator_counts)

        # 3. Enhanced time features
        if 'prediction_time' in processed_df.columns:
            processed_df['prediction_time'] = pd.to_datetime(
                processed_df['prediction_time'], errors='coerce'
            )
            processed_df['hour'] = processed_df['prediction_time'].dt.hour
            processed_df['day_of_week'] = processed_df['prediction_time'].dt.dayofweek
            processed_df['month'] = processed_df['prediction_time'].dt.month
            processed_df['day_of_year'] = processed_df['prediction_time'].dt.dayofyear
            processed_df['is_weekend'] = processed_df['day_of_week'].isin([5, 6]).astype(int)

            # Enhanced time-based features
            processed_df['is_rainy_season'] = processed_df['month'].isin([4, 5, 6, 7, 8, 9, 10]).astype(int)
            processed_df['is_dry_season'] = processed_df['month'].isin([11, 12, 1, 2, 3]).astype(int)
            
            # Cyclical encoding for temporal features
            processed_df['hour_sin'] = np.sin(2 * np.pi * processed_df['hour'] / 24)
            processed_df['hour_cos'] = np.cos(2 * np.pi * processed_df['hour'] / 24)
            processed_df['month_sin'] = np.sin(2 * np.pi * processed_df['month'] / 12)
            processed_df['month_cos'] = np.cos(2 * np.pi * processed_df['month'] / 12)

        # 4. Enhanced farmer experience features
        if 'user_id' in processed_df.columns:
            user_stats = processed_df.groupby('user_id').agg({
                'confidence': ['mean', 'std', 'count'],
                'predicted_intensity': ['mean', 'std'] if 'predicted_intensity' in processed_df.columns else []
            }).fillna(0)
            
            user_stats.columns = ['_'.join(col).strip() for col in user_stats.columns]
            
            # Map back to original dataframe
            for col in user_stats.columns:
                processed_df[f'farmer_{col}'] = processed_df['user_id'].map(user_stats[col]).fillna(0)

        # 5. Enhanced community/district features
        for geo_col in ['community', 'district']:
            if geo_col in processed_df.columns and 'confidence' in processed_df.columns:
                geo_stats = processed_df.groupby(geo_col).agg({
                    'confidence': ['mean', 'std', 'count'],
                    'predicted_intensity': ['mean', 'std'] if 'predicted_intensity' in processed_df.columns else []
                }).fillna(0)
                
                geo_stats.columns = ['_'.join(col).strip() for col in geo_stats.columns]
                
                for col in geo_stats.columns:
                    processed_df[f'{geo_col}_{col}'] = processed_df[geo_col].map(geo_stats[col]).fillna(0)

        # 6. Interaction features
        if 'confidence' in processed_df.columns and 'predicted_intensity' in processed_df.columns:
            processed_df['confidence_intensity_interaction'] = (
                processed_df['confidence'] * processed_df['predicted_intensity']
            )
        
        if 'forecast_length' in processed_df.columns and 'confidence' in processed_df.columns:
            processed_df['forecast_confidence_ratio'] = (
                processed_df['confidence'] / (processed_df['forecast_length'] + 1)
            )

        # 7. Feature selection based on availability
        categorical_features = []
        numerical_features = []
        
        # Categorical features
        potential_cat = ['indicator', 'community', 'district', 'day_of_week', 'month']
        categorical_features = [c for c in potential_cat if c in processed_df.columns]

        # Numerical features  
        potential_num = [
            'predicted_intensity', 'confidence', 'forecast_length', 'hour',
            'day_of_year', 'is_weekend', 'is_rainy_season', 'is_dry_season',
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
            'indicator_frequency', 'confidence_intensity_interaction',
            'forecast_confidence_ratio'
        ]
        
        # Add farmer and geo features
        farmer_features = [col for col in processed_df.columns if col.startswith('farmer_')]
        community_features = [col for col in processed_df.columns if col.startswith('community_')]
        district_features = [col for col in processed_df.columns if col.startswith('district_')]
        
        potential_num.extend(farmer_features + community_features + district_features)
        numerical_features = [n for n in potential_num if n in processed_df.columns]

        print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
        print(f"Numerical features ({len(numerical_features)}): {numerical_features}")

        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.processed_df = processed_df
        return processed_df
    
    def create_preprocessing_pipeline(self, feature_selection=True, k_best=50):
        """Enhanced preprocessing pipeline with feature selection"""
        # Numerical preprocessing
        numerical_transformer = StandardScaler()

        # Categorical preprocessing
        categorical_transformer = OneHotEncoder(
            handle_unknown='ignore', 
            sparse_output=False,
            drop='if_binary'  # Reduce dimensionality
        )

        # Create column transformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ],
            remainder='drop'
        )

        # Add feature selection if requested
        if feature_selection and len(self.numerical_features) + len(self.categorical_features) > k_best:
            self.feature_selector = SelectKBest(score_func=f_classif, k=k_best)
            return Pipeline([
                ('preprocessor', self.preprocessor),
                ('feature_selection', self.feature_selector)
            ])
        else:
            return self.preprocessor
    
    def prepare_features_and_target(self, df, target_column='Target'):
        """Prepare features and target with enhanced encoding"""
        exclude_features = [target_column, 'ID', 'prediction_time', 'user_id']

        feature_columns = [col for col in df.columns if col not in exclude_features]
        X = df[feature_columns]
        y = df[target_column]

        print(f"Using {len(feature_columns)} features for modeling")
        
        # Enhanced target encoding
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        print(f"\nTarget classes: {self.label_encoder.classes_}")
        print("Target distribution:")
        
        class_counts = np.bincount(y_encoded)
        for i, cls in enumerate(self.label_encoder.classes_):
            count = class_counts[i]
            print(f"  {cls}: {count} ({count/len(y_encoded)*100:.1f}%)")
        
        return X, y_encoded
    
    def optimize_sampling_strategy(self, X_train, y_train, cv_folds=3):
        """Find the best sampling strategy for class imbalance"""
        print("🔍 Optimizing sampling strategy...")
        
        strategies = {
            'none': None,
            'smote': SMOTE(random_state=42),
            'adasyn': ADASYN(random_state=42),
            'smote_enn': SMOTEENN(random_state=42),
            'undersample': RandomUnderSampler(random_state=42)
        }
        
        best_strategy = None
        best_score = 0
        results = {}
        
        # Simple model for strategy comparison
        rf_simple = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced')
        
        for name, strategy in strategies.items():
            try:
                if strategy is None:
                    X_resampled, y_resampled = X_train, y_train
                else:
                    X_resampled, y_resampled = strategy.fit_resample(X_train, y_train)
                
                # Cross-validation with macro-F1
                cv_scores = cross_val_score(
                    rf_simple, X_resampled, y_resampled,
                    cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
                    scoring='f1_macro'
                )
                
                mean_score = cv_scores.mean()
                results[name] = mean_score
                
                print(f"  {name}: {mean_score:.4f} (±{cv_scores.std()*2:.4f})")
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_strategy = strategy
                    
            except Exception as e:
                print(f"  {name}: Failed - {e}")
                results[name] = 0
        
        print(f"\n✅ Best sampling strategy: {[k for k, v in strategies.items() if v is best_strategy][0]} (F1-macro: {best_score:.4f})")
        return best_strategy, results
    
    def build_optimized_models(self):
        """Build and optimize individual models"""
        models = {}
        
        # Enhanced Random Forest
        models['rf'] = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced_subsample',  # Better for imbalanced data
            random_state=42,
            n_jobs=-1
        )
        
        # Enhanced Gradient Boosting
        models['gb'] = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,  # Lower learning rate for better performance
            max_depth=8,
            subsample=0.8,
            random_state=42
        )
        
        # Calibrated SVM for better probabilities
        svm_base = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=42
        )
        models['svm'] = CalibratedClassifierCV(svm_base, cv=3)
        
        # Logistic Regression with balanced weights
        models['lr'] = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
        
        return models
    
    def hyperparameter_tuning(self, X_train, y_train, model_type='rf', n_iter=50):
        """Perform hyperparameter tuning with focus on macro-F1"""
        print(f"🔧 Tuning hyperparameters for {model_type}...")
        
        if model_type == 'rf':
            param_dist = {
                'n_estimators': randint(100, 500),
                'max_depth': randint(10, 30),
                'min_samples_split': randint(2, 10),
                'min_samples_leaf': randint(1, 5),
                'max_features': ['sqrt', 'log2', None]
            }
            model = RandomForestClassifier(random_state=42, class_weight='balanced_subsample', n_jobs=-1)
            
        elif model_type == 'gb':
            param_dist = {
                'n_estimators': randint(100, 300),
                'learning_rate': uniform(0.01, 0.2),
                'max_depth': randint(3, 15),
                'subsample': uniform(0.6, 0.4)
            }
            model = GradientBoostingClassifier(random_state=42)
        
        # RandomizedSearchCV with macro-F1 scoring
        search = RandomizedSearchCV(
            model, 
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='f1_macro',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        search.fit(X_train, y_train)
        
        print(f"Best parameters: {search.best_params_}")
        print(f"Best CV score (F1-macro): {search.best_score_:.4f}")
        
        return search.best_estimator_, search.best_params_
    
    def create_ensemble_model(self, models=None, optimize=True):
        """Create an optimized ensemble model"""
        if models is None:
            models = self.build_optimized_models()
        
        # Create voting classifier with optimized weights
        if optimize:
            # Simple weight optimization based on individual performance
            weights = [1.5, 1.2, 1.0, 0.8]  # Favor RF and GB
        else:
            weights = None
        
        self.model = VotingClassifier(
            estimators=[
                ('rf', models['rf']),
                ('gb', models['gb']),
                ('svm', models['svm']),
                ('lr', models['lr'])
            ],
            voting='soft',
            weights=weights
        )
        
        return self.model
    
    def comprehensive_evaluation(self, y_true, y_pred, y_pred_proba=None):
        """Comprehensive model evaluation with focus on macro-F1"""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro')
        metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted')
        metrics['micro_f1'] = f1_score(y_true, y_pred, average='micro')
        metrics['macro_precision'] = precision_score(y_true, y_pred, average='macro')
        metrics['macro_recall'] = recall_score(y_true, y_pred, average='macro')
        
        # Per-class F1 scores
        f1_per_class = f1_score(y_true, y_pred, average=None)
        for i, f1 in enumerate(f1_per_class):
            metrics[f'f1_{self.label_encoder.classes_[i]}'] = f1
        
        # AUC if probabilities available
        if y_pred_proba is not None and len(np.unique(y_true)) > 2:
            try:
                metrics['auc_ovr'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
                metrics['auc_ovo'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovo')
            except:
                pass
        
        return metrics
    
    def train_model(self, X, y, optimize_hyperparams=False, use_best_sampling=True):
        """Enhanced training with comprehensive optimization"""
        print("🚀 Starting enhanced model training...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create and fit preprocessing pipeline
        preprocessing_pipeline = self.create_preprocessing_pipeline()
        X_train_processed = preprocessing_pipeline.fit_transform(X_train)
        X_test_processed = preprocessing_pipeline.transform(X_test)
        
        # Optimize sampling strategy
        if use_best_sampling:
            best_sampler, sampling_results = self.optimize_sampling_strategy(X_train_processed, y_train)
            if best_sampler is not None:
                X_train_processed, y_train = best_sampler.fit_resample(X_train_processed, y_train)
                print(f"Applied optimal sampling strategy")
        
        # Build models
        if optimize_hyperparams:
            print("🔧 Performing hyperparameter optimization...")
            optimized_rf, rf_params = self.hyperparameter_tuning(X_train_processed, y_train, 'rf', n_iter=30)
            optimized_gb, gb_params = self.hyperparameter_tuning(X_train_processed, y_train, 'gb', n_iter=30)
            
            models = {
                'rf': optimized_rf,
                'gb': optimized_gb,
                'svm': self.build_optimized_models()['svm'],
                'lr': self.build_optimized_models()['lr']
            }
            self.best_params = {'rf': rf_params, 'gb': gb_params}
        else:
            models = self.build_optimized_models()
        
        # Create and train ensemble
        ensemble_model = self.create_ensemble_model(models, optimize=True)
        ensemble_model.fit(X_train_processed, y_train)
        
        # Make predictions
        y_pred = ensemble_model.predict(X_test_processed)
        y_pred_proba = ensemble_model.predict_proba(X_test_processed)
        
        # Comprehensive evaluation
        metrics = self.comprehensive_evaluation(y_test, y_pred, y_pred_proba)
        
        print(f"\n📊 MODEL PERFORMANCE RESULTS")
        print("=" * 50)
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro-F1: {metrics['macro_f1']:.4f} ⭐")
        print(f"Weighted-F1: {metrics['weighted_f1']:.4f}")
        print(f"Macro-Precision: {metrics['macro_precision']:.4f}")
        print(f"Macro-Recall: {metrics['macro_recall']:.4f}")
        
        print(f"\nPer-Class F1 Scores:")
        for cls in self.label_encoder.classes_:
            if f'f1_{cls}' in metrics:
                print(f"  {cls}: {metrics[f'f1_{cls}']:.4f}")
        
        # Cross-validation with macro-F1
        cv_scores_f1 = cross_val_score(
            ensemble_model, X_train_processed, y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='f1_macro'
        )
        
        cv_scores_acc = cross_val_score(
            ensemble_model, X_train_processed, y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='accuracy'
        )
        
        print(f"\nCross-Validation Results:")
        print(f"  Macro-F1: {cv_scores_f1.mean():.4f} (±{cv_scores_f1.std()*2:.4f})")
        print(f"  Accuracy: {cv_scores_acc.mean():.4f} (±{cv_scores_acc.std()*2:.4f})")
        
        # Detailed classification report
        target_names = self.label_encoder.classes_
        class_report = classification_report(y_test, y_pred, target_names=target_names)
        print(f"\nDetailed Classification Report:")
        print(class_report)
        
        # Confusion matrix visualization
        self.plot_confusion_matrix(y_test, y_pred)
        
        # Store results
        self.performance_metrics = metrics
        self.performance_metrics.update({
            'cv_f1_macro_mean': cv_scores_f1.mean(),
            'cv_f1_macro_std': cv_scores_f1.std(),
            'cv_accuracy_mean': cv_scores_acc.mean(),
            'cv_accuracy_std': cv_scores_acc.std(),
            'classification_report': class_report,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        })
        
        # Store model components
        self.final_model = ensemble_model
        self.preprocessor = preprocessing_pipeline
        self.X_test = X_test_processed
        self.y_test = y_test
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        
        return ensemble_model, preprocessing_pipeline
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot an enhanced confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Absolute values
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_,
                   ax=ax1)
        ax1.set_title('Confusion Matrix (Counts)')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # Normalized values
        sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Oranges',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_,
                   ax=ax2)
        ax2.set_title('Confusion Matrix (Normalized)')
        ax2.set_ylabel('True Label')
        ax2.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_feature_importance(self, top_n=20):
        """Enhanced feature importance analysis"""
        try:
            # Get feature importance from Random Forest in ensemble
            rf_model = self.final_model.named_estimators_['rf']
            importances = rf_model.feature_importances_
            
            # Get feature names
            if hasattr(self.preprocessor, 'get_feature_names_out'):
                feature_names = self.preprocessor.get_feature_names_out()
            else:
                # Fallback method
                n_features = len(importances)
                feature_names = [f'feature_{i}' for i in range(n_features)]
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print(f"\nTop {top_n} Most Important Features for Rainfall Prediction:")
            print("=" * 60)
            for i, (_, row) in enumerate(importance_df.head(top_n).iterrows(), 1):
                print(f"{i:2d}. {row['feature']:<40} {row['importance']:.4f}")
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            top_features = importance_df.head(top_n)
            
            plt.barh(range(len(top_features)), top_features['importance'], color='skyblue', alpha=0.8)
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title(f'Top {top_n} Feature Importance for Rainfall Prediction')
            plt.gca().invert_yaxis()  # Highest importance at top
            
            # Add value labels on bars
            for i, v in enumerate(top_features['importance']):
                plt.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=9)
            
            plt.tight_layout()
            plt.show()
            
            return importance_df
            
        except Exception as e:
            print(f"Could not analyze feature importance: {e}")
            return None
    
    def generate_improvement_recommendations(self):
        """Generate specific recommendations for improving model performance"""
        recommendations = []
        
        macro_f1 = self.performance_metrics.get('macro_f1', 0)
        accuracy