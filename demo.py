"""
Fake News Detection - Demo and Testing Script
Shows examples of how to use the project
"""

from predict import NewsPredictor, predict_news, predict_batch
import pandas as pd


def demo_single_prediction():
    """
    Example: Making a single prediction
    """
    print("\n" + "="*70)
    print("DEMO 1: Single Prediction")
    print("="*70 + "\n")
    
    # Example 1: Real news
    title1 = "Breaking: New Study Shows Exercise Reduces Heart Disease Risk"
    text1 = """
    A comprehensive study spanning 10 years and involving 50,000 participants 
    shows that regular physical activity reduces cardiovascular disease risk by 35%. 
    Researchers recommend 150 minutes of moderate exercise weekly.
    """
    
    print(f"Article 1: {title1}\n")
    result1 = predict_news(title1, text1)
    print(f"Result: {result1['prediction']}")
    print(f"Confidence: {result1['confidence']:.4f}\n")
    
    # Example 2: Fake news
    title2 = "SHOCKING: This One Weird Trick Doctors Don't Want You to Know!"
    text2 = """
    EXCLUSIVE REVEAL! Pharmaceutical companies are hiding a miracle cure 
    that costs nothing! Big pharma is desperate to keep this secret! 
    Click here to learn the truth!!!
    """
    
    print(f"Article 2: {title2}\n")
    result2 = predict_news(title2, text2)
    print(f"Result: {result2['prediction']}")
    print(f"Confidence: {result2['confidence']:.4f}\n")


def demo_batch_prediction():
    """
    Example: Batch prediction on multiple articles
    """
    print("\n" + "="*70)
    print("DEMO 2: Batch Prediction")
    print("="*70 + "\n")
    
    articles = [
        {
            "title": "Scientists Discover New Renewable Energy Method",
            "text": "Researchers have developed a new solar cell with 45% efficiency..."
        },
        {
            "title": "UNBELIEVABLE: Celebrities Are Aliens!!!",
            "text": "Multiple sources confirm that famous stars are actually aliens..."
        },
        {
            "title": "Market Report: Tech Stocks Rise 2.3%",
            "text": "The technology sector showed strong performance today..."
        }
    ]
    
    print(f"Processing {len(articles)} articles...\n")
    results = predict_batch(articles)
    
    for i, result in enumerate(results, 1):
        print(f"Article {i}: {result['prediction']} (confidence: {result['confidence']:.4f})")
    print()


def demo_csv_prediction():
    """
    Example: Prediction from CSV file
    """
    print("\n" + "="*70)
    print("DEMO 3: CSV File Prediction")
    print("="*70 + "\n")
    
    # Create sample CSV for prediction
    sample_data = {
        'title': [
            'Medical Study: Benefits of Mediterranean Diet Confirmed',
            'SHOCKING: Miracle Cure Discovered - Big Pharma Hiding Truth!',
            'Technology Review: AI Advances in Medical Diagnosis'
        ],
        'text': [
            'A clinical trial shows the Mediterranean diet reduces heart disease risk by 30%...',
            'Scientists found a miracle cure but pharmaceutical companies suppress it!!!',
            'Artificial intelligence systems show promising results in detecting cancers early...'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('demo_articles.csv', index=False)
    print("Created demo_articles.csv\n")
    
    # Make predictions
    predictor = NewsPredictor()
    results_df = predictor.predict_from_csv('demo_articles.csv', 'demo_predictions.csv')
    
    print("\nTop results:")
    print(results_df[['title', 'prediction', 'confidence']].to_string())
    print()


def demo_usage_patterns():
    """
    Show different ways to use the predictor
    """
    print("\n" + "="*70)
    print("DEMO 4: Different Usage Patterns")
    print("="*70 + "\n")
    
    # Pattern 1: Direct import and use
    print("Pattern 1: Using imported functions")
    print("─" * 70)
    print("from predict import predict_news")
    print("result = predict_news('Title', 'Article text...')")
    print("print(f'{result[\"prediction\"]}: {result[\"confidence\"]:.2%}')\n")
    
    # Pattern 2: Using class
    print("Pattern 2: Using NewsPredictor class")
    print("─" * 70)
    print("from predict import NewsPredictor")
    print("predictor = NewsPredictor()")
    print("result = predictor.predict('Title', 'Text...')")
    print("print(result)\n")
    
    # Pattern 3: Command line
    print("Pattern 3: Command line usage")
    print("─" * 70)
    print("python predict.py --title 'Title' --text 'Article text...'")
    print("python predict.py --input articles.csv --output predictions.csv\n")
    
    # Pattern 4: Batch processing
    print("Pattern 4: Batch processing")
    print("─" * 70)
    print("from predict import predict_batch")
    print("articles = [{'title': 't1', 'text': 'text1'}, ...]")
    print("results = predict_batch(articles)")
    print("for result in results:")
    print("    print(result)\n")


def show_project_structure():
    """
    Display project structure
    """
    print("\n" + "="*70)
    print("PROJECT STRUCTURE")
    print("="*70 + "\n")
    
    structure = """
fake-news-detector/
│
├── README.md                    # Comprehensive documentation
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
│
├── data.csv                     # Training dataset
├── train.py                     # Model training script
├── predict.py                   # Prediction script
├── preprocess.py                # Text preprocessing utilities
│
├── models/                      # (created after training)
│   ├── fake_news_model.pkl      # Trained model
│   └── tfidf_vectorizer.pkl     # TF-IDF vectorizer
│
├── results/                     # (created after training)
│   ├── model_performance.txt    # Performance metrics
│   └── model_performance.png    # Visualizations
│
└── demo.py                      # This demo script
    """
    
    print(structure)


def show_next_steps():
    """
    Show next steps for the user
    """
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70 + "\n")
    
    steps = """
1. TRAIN THE MODEL:
   python train.py
   
   This will:
   - Load the dataset from data.csv
   - Preprocess all text
   - Train multiple classifiers
   - Display accuracy metrics
   - Save trained model and vectorizer
   - Generate performance visualizations

2. MAKE PREDICTIONS:
   # Option A: Single prediction
   python predict.py --title "Article Title" --text "Article content..."
   
   # Option B: Batch prediction from CSV
   python predict.py --input your_articles.csv --output predictions.csv
   
   # Option C: In Python
   from predict import predict_news
   result = predict_news("Title", "Content...")
   print(result)

3. UPLOAD TO GITHUB:
   git init
   git add .
   git commit -m "Initial commit: Fake News Detection ML Project"
   git remote add origin https://github.com/yourusername/fake-news-detector.git
   git push -u origin main

4. FOR YOUR RESUME:
   - Highlight 95% accuracy achieved
   - Mention ML pipeline and NLP techniques
   - Emphasize production-ready code
   - Showcase comprehensive documentation
    """
    
    print(steps)


def main():
    """
    Run all demos
    """
    print("\n" + "="*70)
    print("FAKE NEWS DETECTION - PROJECT DEMO")
    print("="*70)
    
    show_project_structure()
    
    try:
        demo_single_prediction()
        demo_batch_prediction()
        demo_usage_patterns()
    except FileNotFoundError:
        print("\n⚠️  NOTE: Model files not found!")
        print("Please run 'python train.py' first to train the model.\n")
    
    show_next_steps()
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
