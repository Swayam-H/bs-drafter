FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# We use uv for fast installation if possible, otherwise pip.
# To keep it standard, we'll install uv and use it to sync.
RUN pip install --no-cache-dir uv

# Install dependencies using uv
RUN uv pip install --system -r pyproject.toml

# Copy application files
COPY . .

# Create user to run the app securely
RUN useradd -m -u 1000 user
RUN chown -R user:user /app
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Expose port 7860 (default for Hugging Face Spaces)
EXPOSE 7860

# Run the server
CMD ["python", "server.py"]
