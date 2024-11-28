from fastapi import FastAPI
from pydantic import BaseModel
import openai
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
app = FastAPI()
client = OpenAI()

# OpenAI API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")

# 입력 모델 정의
class InputData(BaseModel):
    prompt: str

class Prompt(BaseModel):
    prompt: str

@app.get("/")
async def read_root():
    return {"message": "Hello Python"}

# @app.post("/process")
# async def process_prompt(prompt: Prompt):
#     # 여기서는 간단히 입력받은 prompt를 그대로 반환합니다.
#     # 실제 LLM 처리는 이 부분에 구현하면 됩니다.
#     return {"response": f"Processed: {prompt.prompt}"}

@app.post("/process")
async def process_input(data: InputData):
    try:
        # ChatGPT API 호출
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # 또는 사용하고자 하는 모델
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": data.prompt}
            ]
        )
        # 응답 반환
        return {"response": response.choices[0].message.content.strip()}
    except Exception as e:
        print(f"Error: {str(e)}")
        return {"error": "An error occurred while processing your request."}
