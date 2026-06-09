#!/bin/bash
# Setup script for Fake News Detection Project
# Run this to automatically install dependencies and test the project

echo "========================================================================"
echo "FAKE NEWS DETECTION - AUTOMATED SETUP"
echo "========================================================================"
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.7 or higher."
    exit 1
fi

echo "✓ Python 3 detected: $(python3 --version)"
echo ""

# Create virtual environment (optional)
echo "Setting up project..."
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install --break-system-packages -q pandas numpy scikit-learn nltk matplotlib seaborn joblib

if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed successfully"
else
    echo "⚠️  Some dependencies may have failed to install"
fi

echo ""

# Test the model
echo "Testing model..."
python3 predict.py --title "Test Article" --text "This is a test article to verify the model is working correctly." > /tmp/test_output.txt 2>&1

if grep -q "Real\|Fake" /tmp/test_output.txt; then
    echo "✓ Model test successful!"
else
    echo "❌ Model test failed. Please check the installation."
    exit 1
fi

echo ""
echo "========================================================================"
echo "✓ SETUP COMPLETE!"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "1. Try a prediction:"
echo "   python predict.py --title 'Your Title' --text 'Your Article Text'"
echo ""
echo "2. Process a CSV file:"
echo "   python predict.py --input articles.csv --output predictions.csv"
echo ""
echo "3. Or use in Python:"
echo "   from predict import predict_news"
echo "   result = predict_news('Title', 'Text')"
echo ""
echo "4. View documentation:"
echo "   cat README.md"
echo ""
