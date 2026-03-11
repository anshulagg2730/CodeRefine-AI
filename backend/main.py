from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import httpx

load_dotenv()

app = FastAPI()

print("API KEY:", os.getenv("GROQ_API_KEY"))

class CodeRequest(BaseModel):
    code: str
    instruction: str | None = None


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    with open("../frontend/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/analyze")
async def analyze(data: CodeRequest):

    prompt = f"""
You are CodeRefine AI — a senior software engineer and debugging expert.

Analyze the code and follow the output structure STRICTLY.

Run Explanation
Explain the program in numbered points.

Professional Code Review
Write code quality review in numbered points.

Bugs
List syntax and logical errors in numbered points.

Improvements
Write suggested improvements in numbered points.

Performance
Write performance related notes in numbered points.

Security
Write security related observations in numbered points.

Clean Improved Code

First write ONE line explaining what the corrected code does and in which language it runs.

Then provide ONLY the corrected code.

IMPORTANT RULES:

• Fix ALL syntax errors  
• Always return FULL working code  
• Use correct headers

HEADER RULES:

If language is C → use
#include <stdio.h>

If language is C++ → use
#include <iostream>
using namespace std;

If language is Python → no headers

If language is Java → return full class

VERY IMPORTANT:

• Do NOT print language name like c, cpp, python above code  
• Return ONLY code inside the code block  
• Do NOT add explanation inside code
"""

    if data.instruction:
        prompt += f"""

Transformation Instruction

Apply the instruction AFTER fixing the code.

Instruction:
{data.instruction}

Return the FULL transformed code.
"""

    prompt += f"""

Source Code:
{data.code}
"""

    try:

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 800
                }
            )

        data = response.json()

        return {
            "output": data["choices"][0]["message"]["content"]
        }

    except Exception as e:
        print("GROQ ERROR:", e)
        return {"output": str(e)}