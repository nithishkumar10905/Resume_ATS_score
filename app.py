from flask import Flask, render_template, request, jsonify
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import PyPDF2
from docx import Document
import io

app = Flask(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

def extract_text_from_pdf(file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except:
        return ""

def extract_text_from_docx(file):
    """Extract text from Word document"""
    try:
        doc = Document(file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except:
        return ""

def extract_text_from_file(file):
    """Extract text from uploaded file based on extension"""
    filename = file.filename.lower()
    if filename.endswith('.pdf'):
        return extract_text_from_pdf(file)
    elif filename.endswith(('.docx', '.doc')):
        return extract_text_from_docx(file)
    else:
        return ""

def preprocess_text(text):
    """Clean and preprocess text"""
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in tokens if word not in stop_words and len(word) > 2])

def extract_skills(text):
    """Extract potential skills from text"""
    skills_pattern = r'\b(?:python|java|javascript|react|node|sql|html|css|aws|docker|kubernetes|git|machine learning|ai|data science|project management|leadership|communication|teamwork|problem solving|analytical|creative|strategic|technical|programming|development|testing|debugging|database|api|framework|library|algorithm|data structure|software|hardware|network|security|cloud|devops|agile|scrum|kanban|ci/cd|automation|analytics|visualization|reporting|excel|powerbi|tableau|salesforce|crm|erp|sap|oracle|microsoft|adobe|google|amazon|facebook|linkedin|twitter|github|stackoverflow|certification|degree|bachelor|master|phd|experience|years|senior|junior|lead|manager|director|engineer|developer|analyst|consultant|specialist|coordinator|administrator|architect|designer|researcher|scientist|technician|intern|freelance|contractor|remote|onsite|hybrid|full-time|part-time|contract|permanent|temporary)\b'
    return set(re.findall(skills_pattern, text.lower()))

def calculate_match(resume_text, job_text):
    """Calculate match percentage and identify skills"""
    # Preprocess texts
    resume_clean = preprocess_text(resume_text)
    job_clean = preprocess_text(job_text)
    
    # Calculate similarity using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_clean, job_clean])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    match_percentage = round(similarity * 100, 2)
    
    # Extract skills
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_text)
    
    matched_skills = list(resume_skills.intersection(job_skills))
    missing_skills = list(job_skills - resume_skills)
    
    return {
        'match_percentage': match_percentage,
        'matched_skills': matched_skills,
        'missing_skills': missing_skills
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Handle file upload or text input
        resume_text = ''
        job_text = ''
        
        # Check if it's a form submission (file upload)
        if request.content_type and 'multipart/form-data' in request.content_type:
            # Try to get resume from file
            if 'resume_file' in request.files:
                resume_file = request.files['resume_file']
                if resume_file and resume_file.filename:
                    resume_text = extract_text_from_file(resume_file)
            
            # If no file or file is empty, try text input
            if not resume_text:
                resume_text = request.form.get('resume_text', '')
            
            # Get job description from form
            job_text = request.form.get('job_description', '')
        
        # Handle JSON request
        elif request.is_json:
            data = request.get_json()
            resume_text = data.get('resume', '')
            job_text = data.get('job_description', '')
        
        # Validate inputs
        if not resume_text or not resume_text.strip():
            return jsonify({'error': 'Resume is required. Please upload a file or paste resume text.'}), 400
        
        if not job_text or not job_text.strip():
            return jsonify({'error': 'Job description is required. Please paste the job description.'}), 400
        
        # Calculate match
        result = calculate_match(resume_text, job_text)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)