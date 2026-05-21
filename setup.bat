@echo off
echo 🚀 Setting up Enhanced Resume-Job Matching System...

REM Create virtual environment
echo 📦 Creating virtual environment...
python -m venv venv

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️ Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo 📥 Installing dependencies...
pip install -r requirements.txt

REM Download NLTK data
echo 📚 Downloading NLTK data...
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

REM Download spaCy model
echo 🧠 Downloading spaCy model...
python -m spacy download en_core_web_sm

REM Create necessary directories
echo 📁 Creating directories...
if not exist cache mkdir cache
if not exist reports mkdir reports
if not exist logs mkdir logs

REM Test imports
echo 🧪 Testing imports...
python -c "from semantic_matcher import SemanticMatcher; from resume_parser import ResumeParser; from utils import SkillNormalizer; from report_generator import ReportGenerator; print('✅ All imports successful!')"

echo.
echo ✅ Setup complete!
echo.
echo To run the application:
echo   python app_enhanced.py
echo.
echo Or with Docker:
echo   docker-compose up --build
echo.
echo Access the application at: http://localhost:5000
pause
