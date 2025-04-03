import random
import os
from groq import Groq

# Try to get the API key from environment variable first
api_key = os.getenv("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")

# If not found in environment, prompt the user to enter it
if not api_key:
    print("GROQ_API_KEY environment variable not found.")
    api_key = input("Please enter your Groq API key: ")
    
    if not api_key.strip():
        raise ValueError("API key cannot be empty. Please provide a valid Groq API key.")

try:
    client = Groq(api_key=api_key)
except Exception as e:
    raise Exception(f"Failed to initialize Groq client: {str(e)}")

# ------------------------------ Case Study & Problem-Solving Questions ------------------------------
case_questions = {
    "Team Collaboration": [
        "How would you handle a conflict with a colleague?",
        "Describe how you would onboard a new team member.",
        "How do you approach giving feedback to peers?"
    ],
    "Project Management": [
        "Describe a time you had to meet a tight deadline. How did you manage it?",
        "How do you prioritize tasks when you have multiple deadlines?",
        "How would you handle scope creep in a project?"
    ],
    "Problem Solving": [
        "How would you approach a complex problem with limited resources?",
        "Describe a situation where you had to make a decision with incomplete information.",
        "How do you validate your solution to a problem?"
    ]
}

def evaluate_case_study_answer(question, user_answer, topic):
    """Evaluate the case study answer using Groq API."""
    try:
        prompt = f"""
        You are a business analyst evaluating a candidate's response to a case study question.
        
        Question: {question}
        Topic: {topic}
        Candidate's Answer: {user_answer}

        Provide:
        - A score from 0-100
        - Specific feedback
        - An ideal model answer
        """

        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error evaluating answer: {str(e)}"

def ask_case_study_question():
    try:
        topic = random.choice(list(case_questions.keys()))
        question = random.choice(case_questions[topic])
        
        print(f"\n📌 {topic} Question: {question}")
        user_answer = input("\nYour Answer:\n")
        
        if not user_answer.strip():
            print("Error: Answer cannot be empty.")
            return
        
        evaluation = evaluate_case_study_answer(question, user_answer, topic)
        print("\nEvaluation:", evaluation)
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    print("Welcome to the Case Study Interview Simulator!")
    print("----------------------------------------")
    ask_case_study_question()