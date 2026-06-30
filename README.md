# Intelligent Candidate Discovery & Ranking Challenge

This repository contains the code to rank candidates for the AI Engineer role based on the Redrob Hackathon rules. 

The pipeline is split into two parts: pre-computation step (which generates the embeddings and caches models) and a highly optimized ranking step (which runs completely offline, uses only CPU, and finishes under the 5-minute limit).

## Setup Instructions

1. Ensure you have Python 3.9+ installed.
2. Install the required dependencies from the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

3. Ensure the dataset (`candidates.jsonl`) and the job description (`job_description.docx`) are placed inside the `data/` directory.

## Reproduction Steps

### Step 1: Pre-computation (Offline Setup)
This step processes all 100,000 candidates to generate semantic embeddings using the `all-MiniLM-L6-v2` base model and caches the `ms-marco` cross-encoder. 
*Note: This step requires internet to install the models the first time and it exceeds the 5-minute window (hackathon rules (Section 10.3)). On a standard machine, it takes ~90 minutes.*

Run the following command:
```bash
python src/precompute.py --candidates data/candidates.jsonl --jd data/job_description.docx --out data/embeddings.npz
```

### Step 2: Ranking (Under 5 Minutes)
This step executes the final ranking pipeline. It uses the pre-computed `embeddings.npz` to perform the heuristic BM25 scoring and runs the Cross-Encoder re-ranker on the top candidates entirely offline. 

Run the following command to generate the final output CSV:
```bash
python src/rank.py --candidates data/candidates.jsonl --jd data/job_description.docx --embeddings data/embeddings.npz --out TryHards.csv
```

Once completed, the exact Top 100 candidates will be saved in `TryHards.csv` at the root of the project.


##Sandbox link:
<a href='https://colab.research.google.com/drive/1L7wmBvWu05oZSYmo8pMI7T1_ii6BqE94?usp=sharing'>Sample code running in a sandbox</a>
