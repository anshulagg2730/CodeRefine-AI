from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

class CodeRequest(BaseModel):
    code: str
    instruction: str | None = None


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    with open("../frontend/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/analyze")
def analyze(data: CodeRequest):

    base_prompt = f"""
You are CodeRefine AI.

Perform ALL of the following clearly:

1️⃣ Run Explanation (What will this code do?)
2️⃣ Professional Code Review
3️⃣ Bugs (if any)
4️⃣ Improvements
5️⃣ Performance
6️⃣ Security
7️⃣ Clean Improved Version

Code:
{data.code}
"""

    if data.instruction:
        base_prompt += f"""

Also apply this transformation instruction:
{data.instruction}

Then provide:
8️⃣ Transformed Final Code
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": base_prompt}],
            temperature=0.4,
            max_tokens=1500
        )

        return {
            "output": response.choices[0].message.content
        }

    except Exception as e:
        return {"output": f"Error: {str(e)}"}

