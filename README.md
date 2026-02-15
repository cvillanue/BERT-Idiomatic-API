# BERT Idiomatic API

Containerized FastAPI inference service for **token-level idiom span detection** using a fine-tuned **BERT (bert-base-uncased)** model (BIO tagging).  
Returns idiom spans with character offsets + extracted text.

## Quickstart (Docker: recommended)

-Model weights are packaged inside the published Docker image.  
-The GitHub repo excludes `model/` artifacts to keep the repository lightweight.

### 1) Run the container
```bash
docker run --rm -p 8000:8000 cvillanue/idiomatic-bert-api
