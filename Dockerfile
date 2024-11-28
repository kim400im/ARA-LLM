# 베이스 이미지로 Python 3.9 사용
FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 패키지 복사
COPY ./requirements.txt /app/requirements.txt

# 필요한 패키지 설치
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# 앱 소스 코드 복사
COPY ./main.py /app/

# FastAPI 서버 실행 (포트 5000)
EXPOSE 5000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
