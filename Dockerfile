FROM python:3.13-slim

# 1. Create a non-root user (Hugging Face security requirement)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# 2. Set the working directory inside the user's home space
WORKDIR $HOME/app

# 3. Install system dependencies (Switch to root temporarily, then switch back)
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*
USER user

# 4. Copy requirements and install python dependencies
COPY --chown=user requirements.docker.txt .
RUN pip install --no-cache-dir -r requirements.docker.txt

# 5. Copy the rest of your application code with correct ownership
COPY --chown=user . .

# 6. Expose the mandatory Hugging Face Port
EXPOSE 7860
ENV PORT=7860

# 7. Start Chainlit properly on host 0.0.0.0 and port 7860
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "7860"]
