# Small official Python image (from the AWS mirror to avoid Docker Hub rate limits)
FROM public.ecr.aws/docker/library/python:3.9-slim

WORKDIR /app

# Install libraries first so Docker can cache this layer
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy the code and the trained model
COPY src/ src/
COPY app/ app/
COPY models/ models/

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
