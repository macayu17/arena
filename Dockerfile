FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY predict.py grade.py ./
COPY model.pkl ./

ENTRYPOINT ["python", "grade.py"]
