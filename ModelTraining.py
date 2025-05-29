import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.impute import SimpleImputer
import joblib
import pickle
import warnings
warnings.filterwarnings('ignore')

class StrokePredictionTrainer:
    def __init__(self, csv_file_path):
        """
        Initialize the stroke prediction trainer
        
        Args:
            csv_file_path (str): Path to the CSV file containing the dataset
        """
        self.csv_file_path = csv_file_path
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_accuracy = 0
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_and_preprocess_data(self):
        """Load and preprocess the stroke dataset"""
        print("Loading and preprocessing data...")
        
        # Load the dataset
        self.df = pd.read_csv(self.csv_file_path)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Dataset info:")
        print(self.df.info())
        print(f"\nMissing values:")
        print(self.df.isnull().sum())
        
        # Handle missing values in BMI
        if 'bmi' in self.df.columns:
            # Replace 'N/A' with NaN and then impute
            self.df['bmi'] = self.df['bmi'].replace('N/A', np.nan)
            self.df['bmi'] = pd.to_numeric(self.df['bmi'], errors='coerce')
            
            # Impute missing BMI values with median
            bmi_imputer = SimpleImputer(strategy='median')
            self.df['bmi'] = bmi_imputer.fit_transform(self.df[['bmi']]).flatten()
        
        # Encode categorical variables
        categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        
        for col in categorical_cols:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = le
        
        # Separate features and target
        if 'stroke' in self.df.columns:
            self.X = self.df.drop(['stroke', 'id'], axis=1, errors='ignore')
            self.y = self.df['stroke']
        else:
            raise ValueError("Target column 'stroke' not found in dataset")
        
        print(f"Features shape: {self.X.shape}")
        print(f"Target distribution:")
        print(self.y.value_counts())
        
        return self.X, self.y
    
    def split_and_scale_data(self, test_size=0.2, random_state=42):
        """Split data into train/test and scale features"""
        print("Splitting and scaling data...")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def initialize_models(self):
        """Initialize different machine learning models"""
        print("Initializing models...")
        
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(kernel='rbf', random_state=42, probability=True),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Naive Bayes': GaussianNB(),
            'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42)
        }
        
        return self.models
    
    def train_and_evaluate_models(self):
        """Train and evaluate all models"""
        print("Training and evaluating models...")
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train the model
            if name in ['Random Forest', 'Decision Tree', 'Gradient Boosting', 'AdaBoost']:
                # Tree-based models can handle unscaled data
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            else:
                # Other models need scaled data
                model.fit(self.X_train_scaled, self.y_train)
                y_pred = model.predict(self.X_test_scaled)
                y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            auc_score = roc_auc_score(self.y_test, y_pred_proba)
            
            # Cross-validation score
            if name in ['Random Forest', 'Decision Tree', 'Gradient Boosting', 'AdaBoost']:
                cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
            else:
                cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'auc_score': auc_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"{name} - Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}, CV Mean: {cv_scores.mean():.4f}")
            
            # Update best model
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_model = model
                self.best_model_name = name
        
        self.results = results
        return results
    
    def hyperparameter_tuning(self, model_name=None):
        """Perform hyperparameter tuning for the best model or specified model"""
        if model_name is None:
            model_name = self.best_model_name
        
        print(f"\nPerforming hyperparameter tuning for {model_name}...")
        
        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            'Gradient Boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'Logistic Regression': {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.001, 0.01]
            }
        }
        
        if model_name in param_grids:
            base_model = self.models[model_name]
            param_grid = param_grids[model_name]
            
            # Use appropriate data for the model
            if model_name in ['Random Forest', 'Decision Tree', 'Gradient Boosting', 'AdaBoost']:
                X_train_data = self.X_train
            else:
                X_train_data = self.X_train_scaled
            
            grid_search = GridSearchCV(
                base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1
            )
            grid_search.fit(X_train_data, self.y_train)
            
            # Update the best model
            self.best_model = grid_search.best_estimator_
            print(f"Best parameters for {model_name}: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
            
            return grid_search.best_estimator_, grid_search.best_params_
        else:
            print(f"Hyperparameter tuning not implemented for {model_name}")
            return None, None
    
    def print_detailed_results(self):
        """Print detailed results for all models"""
        print("\n" + "="*80)
        print("DETAILED MODEL EVALUATION RESULTS")
        print("="*80)
        
        # Sort models by accuracy
        sorted_results = sorted(self.results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        for i, (name, result) in enumerate(sorted_results, 1):
            print(f"\n{i}. {name}")
            print("-" * 50)
            print(f"Accuracy: {result['accuracy']:.4f}")
            print(f"AUC Score: {result['auc_score']:.4f}")
            print(f"Cross-validation: {result['cv_mean']:.4f} (+/- {result['cv_std']*2:.4f})")
            
            print("\nClassification Report:")
            print(classification_report(self.y_test, result['predictions']))
            
            print("Confusion Matrix:")
            print(confusion_matrix(self.y_test, result['predictions']))
        
        print(f"\n🏆 BEST MODEL: {self.best_model_name} with accuracy: {self.best_accuracy:.4f}")
    
    def save_best_model(self, model_path='best_stroke_model.joblib', scaler_path='scaler.joblib', 
                       encoders_path='label_encoders.joblib'):
        """Save the best model, scaler, and encoders"""
        print(f"\nSaving best model ({self.best_model_name}) and preprocessing objects...")
        
        # Save the model
        joblib.dump(self.best_model, model_path)
        print(f"Model saved to: {model_path}")
        
        # Save the scaler
        joblib.dump(self.scaler, scaler_path)
        print(f"Scaler saved to: {scaler_path}")
        
        # Save label encoders
        joblib.dump(self.label_encoders, encoders_path)
        print(f"Label encoders saved to: {encoders_path}")
        
        # Also save using pickle as alternative
        pickle_path = model_path.replace('.joblib', '.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.best_model, f)
        print(f"Model also saved as pickle to: {pickle_path}")
        
        return model_path, scaler_path, encoders_path
    
    def load_model_for_prediction(self, model_path='best_stroke_model.joblib', 
                                 scaler_path='scaler.joblib', encoders_path='label_encoders.joblib'):
        """Load saved model and preprocessing objects for making predictions"""
        print("Loading saved model and preprocessing objects...")
        
        # Load model
        loaded_model = joblib.load(model_path)
        print(f"Model loaded from: {model_path}")
        
        # Load scaler
        loaded_scaler = joblib.load(scaler_path)
        print(f"Scaler loaded from: {scaler_path}")
        
        # Load encoders
        loaded_encoders = joblib.load(encoders_path)
        print(f"Encoders loaded from: {encoders_path}")
        
        return loaded_model, loaded_scaler, loaded_encoders
    
    def predict_stroke_risk(self, patient_data, model_path='best_stroke_model.joblib', 
                           scaler_path='scaler.joblib', encoders_path='label_encoders.joblib'):
        """
        Predict stroke risk for new patient data
        
        Args:
            patient_data (dict): Dictionary with patient information
                Example: {
                    'gender': 'Male',
                    'age': 67,
                    'hypertension': 0,
                    'heart_disease': 1,
                    'ever_married': 'Yes',
                    'work_type': 'Private',
                    'Residence_type': 'Urban',
                    'avg_glucose_level': 228.69,
                    'bmi': 36.6,
                    'smoking_status': 'formerly smoked'
                }
        """
        # Load model and preprocessing objects
        model, scaler, encoders = self.load_model_for_prediction(model_path, scaler_path, encoders_path)
        
        # Convert to DataFrame
        df_patient = pd.DataFrame([patient_data])
        
        # Apply label encoding
        for col, encoder in encoders.items():
            if col in df_patient.columns:
                df_patient[col] = encoder.transform(df_patient[col].astype(str))
        
        # Scale if needed (check if model needs scaling)
        if self.best_model_name in ['Random Forest', 'Decision Tree', 'Gradient Boosting', 'AdaBoost']:
            X_patient = df_patient.values
        else:
            X_patient = scaler.transform(df_patient.values)
        
        # Make prediction
        prediction = model.predict(X_patient)[0]
        probability = model.predict_proba(X_patient)[0]
        
        print(f"\nPrediction for patient:")
        print(f"Stroke Risk: {'HIGH' if prediction == 1 else 'LOW'}")
        print(f"Probability of No Stroke: {probability[0]:.4f}")
        print(f"Probability of Stroke: {probability[1]:.4f}")
        
        return prediction, probability
    
    def run_complete_pipeline(self, perform_tuning=True):
        """Run the complete training pipeline"""
        print("Starting complete stroke prediction model training pipeline...")
        print("="*80)
        
        # Load and preprocess data
        self.load_and_preprocess_data()
        
        # Split and scale data
        self.split_and_scale_data()
        
        # Initialize models
        self.initialize_models()
        
        # Train and evaluate models
        self.train_and_evaluate_models()
        
        # Hyperparameter tuning
        if perform_tuning:
            self.hyperparameter_tuning()
        
        # Print results
        self.print_detailed_results()
        
        # Save best model
        model_path, scaler_path, encoders_path = self.save_best_model()
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Best model: {self.best_model_name}")
        print(f"Best accuracy: {self.best_accuracy:.4f}")
        print(f"Model saved to: {model_path}")
        
        return self.best_model, self.best_model_name, self.best_accuracy

if __name__ == "__main__":
    trainer = StrokePredictionTrainer('data/healthcare-dataset-stroke-data.csv')  # Replace with your CSV file path
    
    best_model, best_model_name, best_accuracy = trainer.run_complete_pipeline()
    
    sample_patient = {
        'gender': 'Male',
        'age': 67,
        'hypertension': 0,
        'heart_disease': 1,
        'ever_married': 'Yes',
        'work_type': 'Private',
        'Residence_type': 'Urban',
        'avg_glucose_level': 228.69,
        'bmi': 36.6,
        'smoking_status': 'formerly smoked'
    }
    
    print("\n" + "="*80)
    print("TESTING PREDICTION ON SAMPLE PATIENT")
    print("="*80)
    prediction, probability = trainer.predict_stroke_risk(sample_patient)