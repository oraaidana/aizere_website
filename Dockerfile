# 1. Use an official lightweight Python image
FROM python:3.10-slim

# 2. Set environment variables
# Prevents Python from writing pyc files and ensures logs are sent to terminal
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Install SYSTEM dependencies
# Necessary for OpenCV, Matplotlib, and MRI processing (Nibabel)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Create a non-root user for security (Hugging Face Requirement)
# Uses User ID 1000, which is standard for HF Spaces
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# 5. Set the working directory to user home
WORKDIR $HOME/app

# 6. Copy and install Python requirements first
# We do this before copying the whole project to leverage Docker caching
COPY --chown=user:user requirements.txt .

# Install CPU-only versions of Torch to save ~2GB of space
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 7. Copy the entire project
# This includes main.py, model.py, and your checkpoints folder
COPY --chown=user:user . .

# 8. Expose the port Hugging Face expects
EXPOSE 7860

# 9. Start the FastAPI app using Uvicorn
# 0.0.0.0 is mandatory for the app to be reachable externally
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
