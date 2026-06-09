"""
Fake News Detection - Prediction Script
Uses trained model to classify new news articles
"""

import joblib
import pandas as pd
import sys
from preprocess import preprocess_text, combine_text
import argparse


class NewsPredictor:
    """
    Class for making predictions on news articles
    """
    
    def __init__(self, model_path='models/fake_news_model.pkl', 
                 vectorizer_path='models/tfidf_vectorizer.pkl'):
        """
        Load pre-trained model and vectorizer
        """
        try:
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            print("✓ Model and vectorizer loaded successfully\n")
        except FileNotFoundError:
            print(f"ERROR: Model files not found!")
            print("Please run train.py first to train the model")
            sys.exit(1)
    
    def predict(self, title, text):
        """
        Predict if a news article is real or fake
        
        Args:
            title (str): Article title
            text (str): Article text
        
        Returns:
            dict: Prediction result with label and confidence
        """
        # Combine title and text
        combined = f"{title} {text}"
        
        # Preprocess
        processed = preprocess_text(combined)
        
        # Vectorize
        X = self.vectorizer.transform([processed])
        
        # Predict
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0]
        
        # Get confidence for predicted class
        confidence = max(probability)
        
        label = 'Fake' if prediction == 1 else 'Real'
        
        result = {
            'prediction': label,
            'confidence': float(confidence),
            'probability_real': float(probability[0]),
            'probability_fake': float(probability[1])
        }
        
        return result
    
    def predict_batch(self, articles):
        """
        Predict on multiple articles
        
        Args:
            articles (list): List of dicts with 'title' and 'text' keys
        
        Returns:
            list: List of prediction results
        """
        results = []
        
        for i, article in enumerate(articles, 1):
            title = article.get('title', '')
            text = article.get('text', '')
            
            result = self.predict(title, text)
            result['article_id'] = i
            results.append(result)
        
        return results
    
    def predict_from_csv(self, input_path, output_path, title_col='title', text_col='text'):
        """
        Read articles from CSV, predict, and save results
        
        Args:
            input_path (str): Path to input CSV
            output_path (str): Path to output CSV with predictions
            title_col (str): Column name for titles
            text_col (str): Column name for text
        """
        print(f"Reading articles from {input_path}...")
        df = pd.read_csv(input_path)
        
        print(f"Found {len(df)} articles. Making predictions...\n")
        
        predictions = []
        
        for idx, row in df.iterrows():
            title = str(row[title_col]) if title_col in df.columns else ""
            text = str(row[text_col]) if text_col in df.columns else ""
            
            result = self.predict(title, text)
            predictions.append(result)
            
            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{len(df)} articles...")
        
        # Create results dataframe
        results_df = pd.DataFrame(predictions)
        
        # Add original data
        for col in df.columns:
            results_df.insert(0, col, df[col].values)
        
        # Save results
        results_df.to_csv(output_path, index=False)
        print(f"\n✓ Results saved to {output_path}")
        
        # Print summary
        fake_count = (results_df['prediction'] == 'Fake').sum()
        real_count = (results_df['prediction'] == 'Real').sum()
        
        print(f"\nSummary:")
        print(f"  Total articles: {len(results_df)}")
        print(f"  Fake news: {fake_count} ({fake_count/len(results_df)*100:.1f}%)")
        print(f"  Real news: {real_count} ({real_count/len(results_df)*100:.1f}%)")
        print(f"  Average confidence: {results_df['confidence'].mean():.4f}")
        
        return results_df


def predict_news(title, text):
    """
    Convenience function for single prediction
    Can be imported and used in other scripts
    
    Usage:
        from predict import predict_news
        result = predict_news("Title", "Article text...")
        print(result)
    """
    predictor = NewsPredictor()
    return predictor.predict(title, text)


def predict_batch(articles):
    """
    Convenience function for batch prediction
    Can be imported and used in other scripts
    
    Usage:
        from predict import predict_batch
        articles = [{"title": "...", "text": "..."}, ...]
        results = predict_batch(articles)
    """
    predictor = NewsPredictor()
    return predictor.predict_batch(articles)


def main():
    """
    Command-line interface for predictions
    """
    parser = argparse.ArgumentParser(description='Fake News Detection - Prediction')
    
    parser.add_argument('--title', type=str, help='Article title')
    parser.add_argument('--text', type=str, help='Article text')
    parser.add_argument('--input', type=str, help='Input CSV file path')
    parser.add_argument('--output', type=str, default='predictions.csv', 
                       help='Output CSV file path')
    parser.add_argument('--title-col', type=str, default='title',
                       help='Column name for titles in CSV')
    parser.add_argument('--text-col', type=str, default='text',
                       help='Column name for text in CSV')
    
    args = parser.parse_args()
    
    predictor = NewsPredictor()
    
    print("\n" + "="*70)
    print("FAKE NEWS DETECTION - PREDICTION")
    print("="*70 + "\n")
    
    if args.title and args.text:
        # Single prediction
        print(f"Article Title: {args.title}\n")
        result = predictor.predict(args.title, args.text)
        
        print("PREDICTION RESULT:")
        print(f"  Label: {result['prediction']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Probability Real: {result['probability_real']:.4f}")
        print(f"  Probability Fake: {result['probability_fake']:.4f}\n")
        
    elif args.input:
        # Batch prediction from CSV
        predictor.predict_from_csv(args.input, args.output, 
                                  args.title_col, args.text_col)
    else:
        print("ERROR: Please provide either:")
        print("  --title and --text (for single prediction)")
        print("  --input (for CSV batch prediction)")
        print("\nExample:")
        print("  python predict.py --title 'News Title' --text 'Article content...'")
        print("  python predict.py --input articles.csv --output predictions.csv")
        parser.print_help()


if __name__ == "__main__":
    main()
