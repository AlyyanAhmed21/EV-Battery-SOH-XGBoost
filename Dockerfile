# 1. Base Image
FROM python:3.10-slim

# 2. System Deps (GCC for XGBoost, libgomp for parallelism)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 3. Work Directory
WORKDIR /app

# 4. Copy Source Code
# (The .dockerignore file ensures we don't copy the huge data folder)
COPY . /app

# 5. Install Dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6. Expose Port
EXPOSE 8080

# 7. Start Command
CMD ["python", "app.py"]