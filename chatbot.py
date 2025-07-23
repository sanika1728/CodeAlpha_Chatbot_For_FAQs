from tkinter import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import string

nltk.download('stopwords')

from nltk.corpus import stopwords

faq_questions = {
    "What is Python?": "Python is an interpreted, high-level programming language.",
    "What is Java?": "Java is a platform-independent, object-oriented programming language.",
    "What is Git?": "Git helps developers track changes in source code during software development.",
    "What is an API?": "An API allows software apps to interact with each other.",
    "What is a database?": "A database stores data in a structured way for easy retrieval.",
    "What is debugging?": "Debugging is fixing errors in your code to make it work correctly.",
    "What is machine learning?": "Machine learning enables systems to learn from data without being explicitly programmed.",
    "What is frontend dev?": "Frontend means the part of websites/apps that users interact with.",
    "What is backend dev?": "Backend handles data, databases, server logic, and APIs.",
    "What is IDE?": "IDE stands for Integrated Development Environment, used to write and test code."
}

def preprocess(text):
    text = text.lower()
    tokens = text.split()
    filtered = []
    stop = set(stopwords.words("english"))
    for word in tokens:
        if word not in stop and word not in string.punctuation:
            filtered.append(word)
    return ' '.join(filtered)

questions_list = list(faq_questions.keys())
processed_qs = [preprocess(q) for q in questions_list]
vec = TfidfVectorizer()
matrix = vec.fit_transform(processed_qs)

def find_answer(user_query):
    cleaned = preprocess(user_query)
    q_vec = vec.transform([cleaned])
    result = cosine_similarity(q_vec, matrix)
    index = result.argmax()
    score = result[0][index]

    if score > 0.3:
        return faq_questions[questions_list[index]]
    else:
        return "Hmm... I couldn't get that. Try rephrasing?"

def respond():
    msg = user_input.get()
    if not msg.strip():
        return
    chat_win.config(state=NORMAL)
    chat_win.insert(END, "You: " + msg + "\n")
    bot = find_answer(msg)
    chat_win.insert(END, "Bot: " + bot + "\n\n")
    chat_win.config(state=DISABLED)
    user_input.delete(0, END)

def wipe_chat():
    chat_win.config(state=NORMAL)
    chat_win.delete("1.0", END)
    chat_win.config(state=DISABLED)

def exit_now():
    window.destroy()

def save_log():
    chat_win.config(state=NORMAL)
    content = chat_win.get("1.0", END).strip()
    with open("chat_output.txt", "w", encoding="utf-8") as f:
        f.write(content)
    chat_win.config(state=DISABLED)

window = Tk()
window.title("Code ChatBot - Software FAQs")
window.geometry("510x600")
window.config(bg="#eaf4fc")

Label(window, text="CodeAlpha FAQ Bot 🧠", font=("Verdana", 15, "bold"), bg="#eaf4fc").pack(pady=12)

frm = Frame(window)
frm.pack(padx=10, pady=5)

chat_win = Text(frm, height=20, width=60, font=("Consolas", 10))
chat_win.pack(side=LEFT)
chat_win.config(state=DISABLED)

sb = Scrollbar(frm, command=chat_win.yview)
sb.pack(side=RIGHT, fill=Y)
chat_win.config(yscrollcommand=sb.set)

user_input = Entry(window, font=("Arial", 12), width=40)
user_input.pack(padx=10, pady=10, side=LEFT)

Button(window, text="Send", command=respond, bg="#00796b", fg="white", padx=10, pady=5).pack(side=LEFT)

btns = Frame(window, bg="#eaf4fc")
btns.pack(pady=12)

Button(btns, text="Clear", command=wipe_chat, bg="#ff9800", fg="white", padx=12, pady=5).grid(row=0, column=0, padx=10)
Button(btns, text="Save", command=save_log, bg="#3f51b5", fg="white", padx=12, pady=5).grid(row=0, column=1, padx=10)
Button(btns, text="Exit", command=exit_now, bg="#f44336", fg="white", padx=12, pady=5).grid(row=0, column=2, padx=10)

window.mainloop()
