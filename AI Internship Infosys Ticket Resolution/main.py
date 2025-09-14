import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Create chat completion
chat_completion = client.chat.completions.create(
    model="llama-3.1-8b-instant",   # ✅ correct name
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "How is Yash?"},
    ],
    temperature=0.6,
)
# Print the output
print(chat_completion.choices[0].message.content)


data = {
    "ticket_id":"34vervewe3t",
    "ticket_content":"i have issue with my phone",
    "ticket_category" :"",
    "ticket_timestamp":"205-08-22:00:00:00:IST",
    "ticket_by":"email"
}

categories = [
    "maintainance",
    "product support",
    "refund",
    "high_priority_product"
]

def categorization_builder (data:dict):

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "hey can you help me with health suppose cold?",
            },
            {
                "role":"system",
                "content":"You know financial advisor and this is your cfo info {rag_chunk}"
            }
        ]
    )
