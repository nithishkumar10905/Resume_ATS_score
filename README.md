# Resume-Job Matching System

An AI-based application that evaluates how well a candidate's resume matches a job description.

## Features

- **Text Processing**: Cleans and preprocesses resume and job description text
- **AI Matching**: Uses TF-IDF vectorization and cosine similarity for matching
- **Skills Analysis**: Identifies matched and missing skills
- **Clean UI**: Simple interface for input and results display
- **Real-time Results**: Fast processing without data storage

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python app.py
```

3. Open your browser and go to `http://localhost:5000`

## Usage

1. Paste resume content in the first text area
2. Paste job description in the second text area
3. Click "Analyze Match" to get results
4. View match percentage, matched skills, and missing skills

## How It Works

- **Text Preprocessing**: Removes punctuation, converts to lowercase, removes stop words
- **Similarity Calculation**: Uses TF-IDF vectorization and cosine similarity
- **Skills Extraction**: Pattern matching for common technical and soft skills
- **Results Display**: Shows percentage match and skill analysis

The system provides immediate feedback without storing any user data.