import os
import re
import spacy
import PyPDF2
from groq import Groq
from docx import Document
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy model for NLP tasks
try:
    nlp = spacy.load("en_core_web_lg")
except:
    os.system("python -m spacy download en_core_web_lg")
    nlp = spacy.load("en_core_web_lg")

class ResumeAnalyzer:
    def __init__(self, api_key=None):
        """Initialize the ResumeAnalyzer with optional API key."""
        # Try to get API key from environment or parameter
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        
        # Initialize LLM client if API key is available
        self.llm_client = None
        if self.api_key:
            try:
                self.llm_client = Groq(api_key=self.api_key)
                print("LLM client initialized successfully.")
            except Exception as e:
                print(f"Warning: Failed to initialize LLM client: {str(e)}")
                print("Rating feature will be unavailable.")
                
        # Common skills dictionary for different domains
        self.skill_domains = {
            "programming": ["python", "java", "javascript", "c++", "ruby", "php", "swift", "kotlin", "r", "sql"],
            "data_science": ["machine learning", "deep learning", "data analysis", "statistics", "pandas", "numpy", 
                             "scikit-learn", "tensorflow", "pytorch", "nlp", "computer vision"],
            "web_dev": ["html", "css", "react", "angular", "vue", "node.js", "django", "flask", "express", "rest api"],
            "devops": ["docker", "kubernetes", "jenkins", "ci/cd", "aws", "azure", "gcp", "terraform", "ansible"],
            "soft_skills": ["leadership", "communication", "teamwork", "problem solving", "project management", 
                            "time management", "critical thinking", "creativity", "adaptability"]
        }
        
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file."""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    text += pdf_reader.pages[page_num].extract_text()
            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {str(e)}")
            return None
    
    def extract_text_from_docx(self, docx_path):
        """Extract text from a DOCX file."""
        try:
            doc = Document(docx_path)
            return " ".join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            print(f"Error extracting text from DOCX: {str(e)}")
            return None
    
    def extract_text(self, file_path):
        """Extract text from a resume file based on its extension."""
        _, ext = os.path.splitext(file_path.lower())
        
        if ext == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif ext == '.docx':
            return self.extract_text_from_docx(file_path)
        elif ext == '.txt':
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    return file.read()
            except Exception as e:
                print(f"Error reading text file: {str(e)}")
                return None
        else:
            print(f"Unsupported file format: {ext}")
            return None
    
    def preprocess_text(self, text):
        """Clean and preprocess the extracted text."""
        if not text:
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Replace multiple newlines with single newline
        text = re.sub(r'\n+', '\n', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def extract_roles(self, text):
        """Extract job roles and titles from resume."""
        if not text:
            return []
            
        # Common job title words
        job_title_indicators = [
            "engineer", "developer", "analyst", "manager", "director", 
            "specialist", "consultant", "coordinator", "administrator",
            "architect", "designer", "scientist", "lead", "head", "chief",
            "officer", "ceo", "cto", "cfo", "vp", "president", "intern"
        ]
        
        roles = []
        
        # Use NLP to find job titles
        doc = nlp(text)
        
        # Look for job titles in noun chunks
        for chunk in doc.noun_chunks:
            if any(indicator in chunk.text.lower() for indicator in job_title_indicators):
                roles.append(chunk.text.strip())
                
        # Also check for job titles with NER
        for ent in doc.ents:
            if ent.label_ == "WORK_OF_ART" or ent.label_ == "ORG":
                if any(indicator in ent.text.lower() for indicator in job_title_indicators):
                    roles.append(ent.text.strip())
        
        # Clean up roles
        roles = list(set(roles))  # Remove duplicates
        roles = [role for role in roles if len(role.split()) <= 5]  # Remove too long phrases
        
        return roles[:10]  # Return top 10 most likely roles
    
    def extract_skills(self, text):
        """Extract skills from the resume text."""
        if not text:
            return {}
            
        skills_found = {domain: [] for domain in self.skill_domains}
        
        # Create a document with the preprocessed text
        doc = nlp(text.lower())
        
        # Extract skills by domain
        for domain, skills in self.skill_domains.items():
            for skill in skills:
                if re.search(r'\b' + re.escape(skill) + r'\b', text.lower()):
                    skills_found[domain].append(skill)
        
        return skills_found
    
    def calculate_job_match(self, resume_text, job_description):
        """Calculate similarity between resume and job description."""
        if not resume_text or not job_description:
            return 0
            
        try:
            vectorizer = CountVectorizer().fit_transform([resume_text.lower(), job_description.lower()])
            vectors = vectorizer.toarray()
            return cosine_similarity([vectors[0]], [vectors[1]])[0][0] * 100
        except Exception as e:
            print(f"Error calculating similarity: {str(e)}")
            return 0
    
    def rate_resume(self, text, job_description=None):
        """Use LLM to rate resume on a scale of 1-100."""
        if not self.llm_client or not text:
            return "LLM rating unavailable. Please provide a valid API key."
            
        try:
            prompt_parts = [
                "You are a professional resume evaluator. Rate this resume on a scale of 1-100 based on:",
                "- Relevant skills",
                "- Clear role descriptions",
                "- Overall quality and presentation",
                "Provide your rating as a number followed by a brief explanation (max 3 sentences).",
            ]
            
            if job_description:
                prompt_parts.extend([
                    "Also consider how well this resume matches the following job description:",
                    f"\nJob Description:\n{job_description}\n"
                ])
                
            prompt_parts.append(f"\nResume Text:\n{text}")
            
            prompt = "\n".join(prompt_parts)
            
            response = self.llm_client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=300
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error performing resume rating: {str(e)}"
    
    def analyze_resume(self, file_path, job_description=None):
        """Perform simplified analysis of the resume focusing on roles and skills."""
        # Extract text from file
        raw_text = self.extract_text(file_path)
        if not raw_text:
            return {"error": "Failed to extract text from file"}
            
        # Preprocess text
        clean_text = self.preprocess_text(raw_text)
        
        # Analyze core elements
        roles = self.extract_roles(clean_text)
        skills = self.extract_skills(clean_text)
        
        # Calculate match with job description
        job_match = 0
        if job_description:
            job_match = self.calculate_job_match(clean_text, job_description)
        
        # Get rating if available
        rating = self.rate_resume(clean_text, job_description) if self.llm_client else "Rating unavailable"
        
        # Compile results
        analysis_results = {
            'roles': roles,
            'skills': skills,
            'job_match': job_match,
            'rating': rating
        }
        
        return analysis_results

    def print_analysis(self, analysis_results):
        """Print analysis results in a simplified format."""
        if 'error' in analysis_results:
            print(f"Error: {analysis_results['error']}")
            return
            
        print("\n" + "="*50)
        print("RESUME ANALYSIS RESULTS")
        print("="*50)
        
        # Print detected roles
        print("\n📌 DETECTED ROLES:")
        if analysis_results['roles']:
            for role in analysis_results['roles']:
                print(f"  • {role}")
        else:
            print("  No clear roles detected")
        
        # Print skills
        print("\n📌 SKILLS IDENTIFIED:")
        for domain, skills in analysis_results['skills'].items():
            if skills:
                print(f"  {domain.replace('_', ' ').title()}: {', '.join(skills)}")
            
        # Print job match if available
        if analysis_results['job_match'] > 0:
            match = analysis_results['job_match']
            print(f"\n📌 JOB DESCRIPTION MATCH: {match:.1f}%")
                
        # Print rating
        print("\n📌 RESUME RATING:")
        print(analysis_results['rating'])
        
        print("\n" + "="*50)


# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = ResumeAnalyzer()
    
    # Get resume file path
    resume_path = input("Enter the path to your resume file (.pdf, .docx, .txt): ")
    
    # Get optional job description
    use_job = input("Do you want to compare with a job description? (y/n): ").lower()
    job_description = None
    if use_job == 'y':
        job_path = input("Enter the path to job description file or paste the description directly: ")
        
        # Check if it's a file path or direct text
        if os.path.isfile(job_path):
            with open(job_path, 'r') as file:
                job_description = file.read()
        else:
            job_description = job_path
    
    # Analyze resume
    results = analyzer.analyze_resume(resume_path, job_description)
    
    # Print results
    analyzer.print_analysis(results)