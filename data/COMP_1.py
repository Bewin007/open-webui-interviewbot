import gradio as gr
import PyPDF2
import torch
from llama_cpp import Llama
import difflib
import spacy
import pytextrank
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure CUDA is available
assert torch.cuda.is_available(), "CUDA is not available. Please check your GPU setup."

# Set the device to CUDA
device = torch.device("cuda")
torch.cuda.set_device(0)  # Use the first GPU
print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# Initialize the Llama model with CUDA support
llm = Llama.from_pretrained(
    repo_id="hugging-quants/Llama-3.2-3B-Instruct-Q8_0-GGUF",
    filename="llama-3.2-3b-instruct-q8_0.gguf",
    n_gpu_layers=-1,  # Use all GPU layers
    n_ctx=2048,  # Adjust context size as needed
    device=device
)

# Load spaCy's English model and add PyTextRank to the pipeline
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textrank")

# Initialize TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Optimized Job Roles Dictionary
job_roles = {
    "Computer Science": ["Software Developer", "Systems Analyst", "Database Administrator"],
    "Information Technology": ["IT Support Specialist", "Network Administrator", "Cybersecurity Analyst"],
    "Electrical Engineering": ["Electrical Design Engineer", "Power Systems Engineer", "Control Systems Engineer"],
    "Mechanical Engineering": ["Mechanical Design Engineer", "HVAC Engineer", "Automotive Engineer"],
    "Civil Engineering": ["Structural Engineer", "Construction Manager", "Geotechnical Engineer"],
    "Biotechnology": ["Biomedical Engineer", "Research Scientist", "Quality Control Analyst"],
    "Electronics and Communication": ["Electronics Design Engineer", "Telecommunications Engineer", "Embedded Systems Engineer"],
    "Chemical Engineering": ["Process Engineer", "Chemical Plant Manager", "Environmental Engineer"],
    "Aerospace Engineering": ["Aerodynamics Engineer", "Flight Systems Engineer", "Aerospace Project Manager"],
    "Data Science": ["Data Scientist", "Machine Learning Engineer", "Data Analyst"],
    "Artificial Intelligence": ["AI Research Scientist", "AI Engineer", "Robotics Engineer"],
    "Robotics": ["Robotics Engineer", "Automation Engineer", "Mechatronics Engineer"],
    "Management Studies": ["Business Analyst", "Project Manager", "Operations Manager"],
    "Physics": ["Physicist", "Laboratory Technician"],
    "Chemistry": ["Chemist", "Chemical Analyst", "Pharmaceutical Scientist"],
    "Mathematics": ["Mathematician", "Statistician", "Actuary"],
    "Environmental Science": ["Environmental Consultant", "Sustainability Specialist", "Environmental Scientist"]
}

# Function to extract text from PDF
def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text() + '\n'
        return text.strip()
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"

# Function to analyze domain from resume text
def analyze_domain(resume_text):
    for domain in job_roles:
        if domain.lower() in resume_text.lower():
            return domain
    return "General"

def extract_keywords_textrank(text):
    doc = nlp(text)
    return [phrase.text for phrase in doc._.phrases[:10]]  # Get top 10 phrases

def generate_hr_questions(domain, job_role, job_description):
    prompt = f"Generate 5 high-quality Technical HR interview questions for a candidate specializing in {domain} for the role of {job_role} with the following job description:\n{job_description}\nFocus on advanced concepts and industry best practices."
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
        temperature=0.7,
    )
    questions = response['choices'][0]['message']['content'].strip().split('\n')
    return [q.strip() for q in questions if q.strip()]

def generate_answer(question):
    prompt = f"Provide a concise and informative answer to the following technical question:\n{question}\nInclude relevant concepts, methodologies, and examples where applicable."
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.5,
    )
    return response['choices'][0]['message']['content'].strip()

def provide_feedback(question, user_answer, expected_answer):
    user_answer_lower = user_answer.lower()
    expected_answer_lower = expected_answer.lower()
    question_lower = question.lower()

    user_keywords = set(extract_keywords_textrank(user_answer_lower))
    expected_keywords = set(extract_keywords_textrank(expected_answer_lower))
    question_keywords = set(extract_keywords_textrank(question_lower))

    relevant_keywords = question_keywords.intersection(expected_keywords)
    user_relevant_keywords = user_keywords.intersection(relevant_keywords)
    keyword_relevance = len(user_relevant_keywords) / len(relevant_keywords) if relevant_keywords else 0

    tfidf_matrix = tfidf_vectorizer.fit_transform([user_answer_lower, expected_answer_lower])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    final_score = (0.6 * keyword_relevance + 0.4 * cosine_sim) * 10
    rating = round(final_score)

    if rating == 10:
        suggestions = ["Excellent answer! You've fully addressed the question and demonstrated deep understanding."]
    elif rating >= 8:
        suggestions = ["Great answer! You've covered most key points. Consider elaborating on some concepts for a perfect score."]
    elif rating >= 6:
        suggestions = ["Good answer, but there's room for improvement. Try to include more specific details and examples."]
    elif rating >= 4:
        suggestions = ["Fair attempt, but your answer lacks depth. Focus on key concepts mentioned in the question and provide more specific examples."]
    else:
        suggestions = ["Your answer needs significant improvement. Review the question carefully and try to address its main points."]

    present_keywords = user_relevant_keywords
    missing_keywords = relevant_keywords - user_relevant_keywords
    feedback_details = f"Relevant Keywords Present: {', '.join(present_keywords)}\n"
    feedback_details += f"Relevant Keywords Missing: {', '.join(missing_keywords)}\n"

    return rating, suggestions + [feedback_details]

def provide_feedback_for_all(questions, user_answers):
    feedback_summary = ""
    for i, (question, user_answer) in enumerate(zip(questions, user_answers), start=1):
        expected_answer = generate_answer(question)
        rating, suggestions = provide_feedback(question, user_answer, expected_answer)
        feedback_summary += f"**Question {i}:** {question}\n"
        feedback_summary += f"**Your Answer:** {user_answer}\n"
        feedback_summary += f"**Expected Answer:** {expected_answer}\n"
        feedback_summary += f"**Rating:** <span style='color: red;'>{rating}/10</span>\n"
        feedback_summary += f"**Suggestions:** {' '.join(suggestions)}\n\n"
    return feedback_summary if feedback_summary else "No feedback available. Please answer all the questions first."

def upload_resume(file):
    resume_text = extract_text_from_pdf(file)
    if "Error" in resume_text:
        return resume_text, "N/A", gr.update(choices=[], value=None)
    detected_domain = analyze_domain(resume_text)
    if detected_domain in job_roles:
        relevant_roles = job_roles[detected_domain]
    else:
        relevant_roles = []
    all_roles = [role for roles in job_roles.values() for role in roles]
    return "Resume uploaded and analyzed.", detected_domain, gr.update(choices=relevant_roles + all_roles, value=None)

def handle_chat(user_message, selected_domain, selected_job_role, job_description, chat_history, questions, current_question_index, user_answers):
    response_message = ""
    if not questions:
        response_message = "Please generate questions first."
    elif current_question_index < len(questions):
        if user_message.strip().lower() in ["skip", "i don't know", "don't know", "idk", "i am sorry"]:
            user_answers.append(user_message)
            response_message = f"You indicated: {user_message}.\n"
            current_question_index += 1
            if current_question_index < len(questions):
                response_message += f"Question {current_question_index + 1}: {questions[current_question_index]}"
            else:
                response_message += "All questions answered!"
        else:
            response_message = f"Your answer: {user_message}\n"
            user_answers.append(user_message)
            current_question_index += 1
            if current_question_index < len(questions):
                response_message += f"Question {current_question_index + 1}: {questions[current_question_index]}"
            else:
                response_message += "All questions answered!"
    chat_history.append((user_message, response_message))
    return chat_history, chat_history, questions, current_question_index, user_answers

def start_generate_questions(selected_domain, selected_job_role, job_description, chat_history):
    questions = generate_hr_questions(selected_domain, selected_job_role, job_description)
    response_message = f"Question 1: {questions[0]}" if questions else "No questions generated."
    chat_history.append(("", response_message))
    return chat_history, chat_history, questions, 0, []

def generate_recent_answer(questions, current_question_index, chat_history):
    if current_question_index < len(questions):
        question = questions[current_question_index]
        answer = generate_answer(question)
        response_message = f"Answer: {answer}"
    else:
        response_message = "No question available to generate an answer."
    chat_history.append(("", response_message))
    return chat_history, chat_history

css = """
body {
    background-color: #f0f0f0;
    font-family: 'Arial', sans-serif;
}
.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}
.gradio-container {
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.user-msg {
    background-color: #25D366;
    color: white;
    border-radius: 15px;
    padding: 8px 12px;
    margin: 5px;
    text-align: right;
    max-width: 60%;
    float: right;
    clear: both;
}
.bot-msg {
    background-color: #262626;
    color: white;
    border-radius: 15px;
    padding: 8px 12px;
    margin: 5px;
    text-align: left;
    max-width: 60%;
    float: left;
    clear: both;
}
.clearfix::after {
    content: "";
    clear: both;
    display: table;
}
.generate-btn {
    background-color: #FF69B4 !important;
    color: white !important;
}
.feedback-btn {
    background-color: #4169E1 !important;
    color: white !important;
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("# 🎓 KITS - Interview Prep Bot")
    
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Accordion("Resume Analysis", open=False):
                file_input = gr.File(label="📄 Upload your resume (PDF)", file_types=['pdf'])
                upload_button = gr.Button("📤 Upload and Analyze Resume")
                upload_status = gr.Textbox(label="Status")
                detected_domain = gr.Textbox(label="🎯 Detected Specialization")
                job_role_dropdown = gr.Dropdown(label="🔍 Select Job Role", choices=[])
                job_description_input = gr.Textbox(label="📋 Enter Job Description (max 200 words)", max_lines=10)
            
            generate_button = gr.Button("🔄 Generate Questions", elem_classes=["generate-btn"])
            feedback_button = gr.Button("📝 Provide Feedback", elem_classes=["feedback-btn"])

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="💬 Chat")
            chat_input = gr.Textbox(label="Type your answer", placeholder="Type here or click 'Skip' to proceed")
            with gr.Row():
                chat_button = gr.Button("📨 Send")
                skip_button = gr.Button("🔄 Skip")
                generate_answer_button = gr.Button("💡 Generate Answer")

    chat_history = gr.State([])
    questions = gr.State([])
    current_question_index = gr.State(0)
    user_answers = gr.State([])

    upload_button.click(upload_resume, inputs=file_input, outputs=[upload_status, detected_domain, job_role_dropdown])
    generate_button.click(start_generate_questions, inputs=[detected_domain, job_role_dropdown, job_description_input, chat_history], outputs=[chatbot, chat_history, questions, current_question_index, user_answers])
    chat_button.click(handle_chat, inputs=[chat_input, detected_domain, job_role_dropdown, job_description_input, chat_history, questions, current_question_index, user_answers], outputs=[chatbot, chat_history, questions, current_question_index, user_answers])
    skip_button.click(lambda domain, role, desc, chat_hist, ques, curr_idx, user_ans: handle_chat("skip", domain, role, desc, chat_hist, ques, curr_idx, user_ans), inputs=[detected_domain, job_role_dropdown, job_description_input, chat_history, questions, current_question_index, user_answers], outputs=[chatbot, chat_history, questions, current_question_index, user_answers])
    generate_answer_button.click(generate_recent_answer, inputs=[questions, current_question_index, chat_history], outputs=[chatbot, chat_history])
    feedback_button.click(provide_feedback_for_all, inputs=[questions, user_answers], outputs=[chatbot])

if __name__ == "__main__":
    print("Initializing GPU...")
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    dummy_input = dummy_input.to(device)
    print("GPU initialized and ready.")
    demo.launch()