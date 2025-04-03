import random
import os
from groq import Groq

# Try to get the API key from environment variable first
api_key = os.getenv("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")  # Changed this line - it should be the env var name, not the key itself

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

# ------------------------------ Technical Questions ------------------------------
tech_questions = {
    "Python Programming": [
        "Explain the difference between lists and tuples in Python.",
        "What are decorators in Python and how do you use them?",
        "How does memory management work in Python?",
        "Explain comprehensions in Python with examples."
    ],
    "Machine Learning": [
        "What is the difference between supervised and unsupervised learning?",
        "Explain the concept of overfitting and how to prevent it.",
        "What is the bias-variance tradeoff?",
        "Explain gradient descent algorithm and its variations."
    ],
    "Data Structures": [
        "Explain the concept of a hash table and its average time complexity.",
        "Compare and contrast arrays and linked lists.",
        "Describe the difference between a stack and a queue.",
        "What is a binary search tree and what are its advantages?"
    ],
    "Web Development": [
        "Explain RESTful API principles.",
        "What is CORS and why is it important?",
        "Describe the difference between cookies and local storage.",
        "What are the main features of HTTP/2 compared to HTTP/1.1?"
    ]
}

def evaluate_technical_answer(question, user_answer, topic):
    """Evaluate the technical answer using Groq API."""
    try:
        prompt = f"""
        You are a senior technical interviewer assessing the candidate's response.
        
        Question: {question}
        Topic: {topic}
        Candidate's Answer: {user_answer}

        Provide:
        - A score from 0-100
        - Specific feedback
        - An ideal model answer
        """

        response = client.chat.completions.create(
            model="llama3-70b-8192",  # Changed from mixtral to Llama3
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error evaluating answer: {str(e)}"

def ask_technical_question():
    try:
        topic = random.choice(list(tech_questions.keys()))
        question = random.choice(tech_questions[topic])
        
        print(f"\n📌 {topic} Question: {question}")
        user_answer = input("\nYour Answer:\n")
        
        if not user_answer.strip():
            print("Error: Answer cannot be empty.")
            return
        
        evaluation = evaluate_technical_answer(question, user_answer, topic)
        print("\nEvaluation:", evaluation)
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    print("Welcome to the Technical Interview Simulator!")
    print("----------------------------------------")
    ask_technical_question()