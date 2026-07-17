FROM python:3.11-slim

# Create user to run the app
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy dependency files
COPY --chown=user:user pyproject.toml .
# We use uv for fast installation if possible, otherwise pip.
# To keep it standard, we'll install uv and use it to sync.
RUN pip install --no-cache-dir uv
COPY --chown=user:user uv.lock .

# Install dependencies using uv
RUN uv pip install --system -r pyproject.toml

# Copy application files
COPY --chown=user:user . .

# Expose port 7860 (default for Hugging Face Spaces)
EXPOSE 7860

# Run the server
CMD ["python", "server.py"]
