"""
Fake News Detection - Model Training Script
Trains multiple classifiers and evaluates their performance
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import preprocess_text, combine_text


class FakeNewsDetector:
    """
    Main class for training fake news detection models
    """
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.results = {}
        
    def load_data(self, filepath='data.csv'):
        """
        Load data from CSV file
        Expected columns: title, text, label
        """
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Labels distribution:\n{df['label'].value_counts()}\n")
        
        return df
    
    def preprocess_data(self, df):
        """
        Preprocess text data
        Combines title and text, then applies preprocessing
        """
        print("Preprocessing data...")
        
        # Combine title and text
        df['combined_text'] = df.apply(
            lambda row: combine_text(row, ['title', 'text']), 
            axis=1
        )
        
        # Apply preprocessing to combined text
        df['processed_text'] = df['combined_text'].apply(preprocess_text)
        
        print("Preprocessing complete!\n")
        return df
    
    def prepare_features(self, df, max_features=5000, ngram_range=(1, 2)):
        """
        Convert text to TF-IDF features
        """
        print(f"Extracting TF-IDF features (max_features={max_features})...")
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.9,
            stop_words='english'
        )
        
        X = self.vectorizer.fit_transform(df['processed_text'])
        y = df['label'].values
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Vocabulary size: {len(self.vectorizer.get_feature_names_out())}\n")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets
        """
        print(f"Splitting data (test_size={test_size})...")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y
        )
        
        print(f"Training set size: {self.X_train.shape[0]}")
        print(f"Testing set size: {self.X_test.shape[0]}\n")
    
    def train_models(self):
        """
        Train multiple classifiers and evaluate performance
        """
        models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000, 
                random_state=42,
                solver='lbfgs'
            ),
            'Multinomial Naive Bayes': MultinomialNB()
        }
        
        for model_name, model in models.items():
            print(f"Training {model_name}...")
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            auc = roc_auc_score(self.y_test, y_pred_proba)
            
            # Store results
            self.results[model_name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"✓ {model_name} trained")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            print(f"  AUC-ROC:   {auc:.4f}\n")
        
        # Select best model
        best_model_name = max(self.results.items(), key=lambda x: x[1]['f1'])[0]
        self.model = self.results[best_model_name]['model']
        
        print(f"✓ Best model selected: {best_model_name}\n")
        
        return best_model_name
    
    def evaluate_model(self, model_name='Logistic Regression'):
        """
        Print detailed evaluation metrics and confusion matrix
        """
        result = self.results[model_name]
        y_pred = result['predictions']
        
        print(f"\n{'='*60}")
        print(f"DETAILED EVALUATION - {model_name}")
        print(f"{'='*60}\n")
        
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred, 
                                   target_names=['Real News', 'Fake News']))
        
        cm = confusion_matrix(self.y_test, y_pred)
        print("\nConfusion Matrix:")
        print(f"                 Predicted Fake  Predicted Real")
        print(f"Actual Fake              {cm[1, 1]:>3}           {cm[1, 0]:>3}")
        print(f"Actual Real              {cm[0, 1]:>3}           {cm[0, 0]:>3}\n")
    
    def plot_results(self, model_name='Logistic Regression', save_path='results/'):
        """
        Plot confusion matrix and ROC curve
        """
        os.makedirs(save_path, exist_ok=True)
        
        result = self.results[model_name]
        y_pred = result['predictions']
        y_proba = result['probabilities']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                   xticklabels=['Real', 'Fake'],
                   yticklabels=['Real', 'Fake'])
        axes[0].set_ylabel('Actual')
        axes[0].set_xlabel('Predicted')
        axes[0].set_title(f'Confusion Matrix - {model_name}')
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(self.y_test, y_proba)
        auc = self.results[model_name]['auc']
        
        axes[1].plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})', linewidth=2)
        axes[1].plot([0, 1], [0, 1], 'k--', label='Random classifier')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title(f'ROC Curve - {model_name}')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}model_performance.png', dpi=300, bbox_inches='tight')
        print(f"✓ Performance plots saved to {save_path}model_performance.png")
        plt.close()
    
    def save_model(self, model_path='models/fake_news_model.pkl', 
                  vectorizer_path='models/tfidf_vectorizer.pkl'):
        """
        Save trained model and vectorizer to disk
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        
        print(f"✓ Model saved to {model_path}")
        print(f"✓ Vectorizer saved to {vectorizer_path}\n")
    
    def save_results(self, filepath='results/model_performance.txt'):
        """
        Save detailed results to text file
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            f.write("="*70 + "\n")
            f.write("FAKE NEWS DETECTION - MODEL PERFORMANCE REPORT\n")
            f.write("="*70 + "\n\n")
            
            for model_name, result in self.results.items():
                f.write(f"Model: {model_name}\n")
                f.write(f"{'─'*70}\n")
                f.write(f"Accuracy:  {result['accuracy']:.4f}\n")
                f.write(f"Precision: {result['precision']:.4f}\n")
                f.write(f"Recall:    {result['recall']:.4f}\n")
                f.write(f"F1-Score:  {result['f1']:.4f}\n")
                f.write(f"AUC-ROC:   {result['auc']:.4f}\n")
                f.write("\n")
            
            f.write("="*70 + "\n")
            f.write(f"Best Model: Logistic Regression\n")
            f.write("="*70 + "\n")
        
        print(f"✓ Results saved to {filepath}")


def main():
    """
    Main training pipeline
    """
    print("\n" + "="*70)
    print("FAKE NEWS DETECTION - MODEL TRAINING")
    print("="*70 + "\n")
    
    detector = FakeNewsDetector()
    
    # Load data
    df = detector.load_data('data.csv')
    
    # Preprocess data
    df = detector.preprocess_data(df)
    
    # Prepare features
    X, y = detector.prepare_features(df)
    
    # Split data
    detector.split_data(X, y)
    
    # Train models
    best_model = detector.train_models()
    
    # Evaluate
    detector.evaluate_model(best_model)
    
    # Plot results
    detector.plot_results(best_model)
    
    # Save model
    detector.save_model()
    
    # Save results
    detector.save_results()
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
