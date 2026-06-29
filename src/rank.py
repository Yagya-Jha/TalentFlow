import json
import csv
import argparse
import math
from collections import Counter
import re
import time
import unicodedata
import numpy as np
import random

try:
    import docx
except ImportError:
    pass

def normalize_text(text: str) -> str:
    if not text:
        return ""
    return unicodedata.normalize('NFKD', str(text)).encode('ascii', 'ignore').decode('ascii')

def load_jd(jd_path: str) -> str:
    if jd_path.endswith('.docx'):
        doc = docx.Document(jd_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        text = '\n'.join(full_text)
    else:
        with open(jd_path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
    return normalize_text(text).lower()

def safe_get(d, *keys, default=None):
    for k in keys:
        if d is None:
            return default
        d = d.get(k)
    return d if d is not None else default

def stem(word):
    if word.endswith('ing') and len(word) > 5:
        return word[:-3]
    if word.endswith('ed') and len(word) > 4:
        return word[:-2]
    if word.endswith('es') and len(word) > 4:
        return word[:-2]
    if word.endswith('s') and len(word) > 3:
        return word[:-1]
    return word

def extract_tokens(text: str, stemmer=True, n_grams=True):
    words = re.findall(r'\w+', str(text).lower())
    stopwords = {'and', 'the', 'is', 'in', 'to', 'with', 'for', 'of', 'a', 'an', 'on', 'at', 'by', 'as', 'this', 'that'}
    words = [w for w in words if w not in stopwords and len(w) > 2]
    
    if stemmer:
        words = [stem(w) for w in words]
        
    tokens = list(words)
    if n_grams:
        for i in range(len(words) - 1):
            tokens.append(f"{words[i]}_{words[i+1]}")
        for i in range(len(words) - 2):
            tokens.append(f"{words[i]}_{words[i+1]}_{words[i+2]}")
            
    return tokens

def extract_candidate_text_with_weights(c):
    tokens_weighted = []
    profile = c.get('profile', {})
    
    headline_tokens = extract_tokens(profile.get('headline', ''))
    for t in headline_tokens:
        tokens_weighted.append((t, 3.0))
        
    title_tokens = extract_tokens(profile.get('current_title', ''))
    for t in title_tokens:
        tokens_weighted.append((t, 3.0))
        
    summary_tokens = extract_tokens(profile.get('summary', ''))
    for t in summary_tokens:
        tokens_weighted.append((t, 1.5))
        
    for exp in c.get('career_history', []):
        for t in extract_tokens(exp.get('title', '')):
            tokens_weighted.append((t, 2.0))
        for t in extract_tokens(exp.get('description', '')):
            tokens_weighted.append((t, 1.0))
            
    for skill in c.get('skills', []):
        for t in extract_tokens(skill.get('name', '')):
            tokens_weighted.append((t, 2.5))
            
    for proj in c.get('projects', []):
        for t in extract_tokens(proj.get('description', '')):
            tokens_weighted.append((t, 1.0))
            
    return tokens_weighted

def compute_cosine_similarity(vecA, vecB):
    dot = np.dot(vecA, vecB)
    normA = np.linalg.norm(vecA)
    normB = np.linalg.norm(vecB)
    if normA == 0 or normB == 0:
        return 0.0
    return dot / (normA * normB)

def generate_reasoning(c: dict, rank: int, jd_budget: float, honeypot_penalty: float) -> str:
    yoe = safe_get(c, 'profile', 'years_of_experience', default=0)
    title = safe_get(c, 'profile', 'current_title', default='Professional')
    
    skills = [s.get('name') for s in c.get('skills', []) if s.get('name')]
    top_skills = ", ".join(skills[:3]) if skills else "relevant industry skills"
    
    signals = c.get('redrob_signals', {})
    notice_days = signals.get('notice_period_days', 60)
    salary = signals.get('expected_salary_range_inr_lpa', {})
    max_salary = salary.get('max', 0)
    
    days_since_active = 0
    last_active = signals.get('last_active_date')
    if last_active:
        try:
            parts = last_active.split('-')
            days_since_active = (2026 - int(parts[0])) * 365 + (6 - int(parts[1])) * 30 + (1 - int(parts[2]))
        except:
            pass
            
    concerns = []
    if notice_days > 60:
        concerns.append(f"a {notice_days}-day notice period")
    if max_salary > jd_budget * 1.2:
        concerns.append(f"high salary expectations ({max_salary} LPA)")
    if days_since_active > 180:
        concerns.append("platform inactivity")
    if honeypot_penalty < 1.0:
        concerns.append("inconsistencies in their timeline")
        
    # Extract adjectives based on rank
    if rank <= 10:
        tones = ["an exceptional match", "a top-tier candidate", "highly recommended", "an outstanding fit", "a premium profile"]
        tone_sentences = ["Exceptional match.", "Top-tier candidate.", "Highly recommended.", "Outstanding fit.", "A premium profile."]
    elif rank <= 40:
        tones = ["a solid profile", "a strong contender", "a good alignment", "a reliable match", "a great potential fit"]
        tone_sentences = ["Solid profile.", "Strong contender.", "Good alignment.", "Reliable match.", "Great potential."]
    elif rank <= 80:
        tones = ["a moderate fit", "a reasonable candidate", "acceptable but lacking some depth", "an acceptable profile"]
        tone_sentences = ["Moderate fit.", "Reasonable candidate.", "Has potential but lacks some depth.", "Acceptable profile."]
    else:
        tones = ["a marginal fit", "a borderline candidate", "a weaker alignment", "falling short on core requirements"]
        tone_sentences = ["Marginal fit.", "Borderline candidate.", "Weaker alignment.", "Falls short on core requirements."]
        
    tone_adj = random.choice(tones)
    tone_sent = random.choice(tone_sentences)
    
    # Variation in concerns/closing
    if concerns:
        concern_phrases = [
            f"We should note {', '.join(concerns)}.",
            f"Keep in mind their {', '.join(concerns)}.",
            f"However, they have {', '.join(concerns)}.",
            f"One caveat is the {', '.join(concerns)}.",
            f"Note: {', '.join(concerns)}."
        ]
        closing = random.choice(concern_phrases)
    else:
        if rank <= 50:
            closing_phrases = [
                "Signals indicate they are highly engaged and available.",
                "No major red flags detected in their behavioral signals.",
                "Aligns cleanly with the JD without any obvious behavioral drawbacks.",
                "Shows prompt engagement metrics and good availability.",
                "Seems highly responsive and ready for the role."
            ]
        else:
            closing_phrases = [
                "They lack the deeper specialization seen in top candidates.",
                "Doesn't quite match the strict technical density we want.",
                "While capable, other candidates offer more direct experience.",
                "Outcompeted by peers on specific technical depth."
            ]
        closing = random.choice(closing_phrases)
        
    structure_type = random.randint(1, 3)
    
    if structure_type == 1:
        # Structure 1 (The Classic): Independent sentences
        exp_phrases = [
            f"Brings {yoe} years as a {title} with deep knowledge in {top_skills}.",
            f"A {title} with {yoe} YOE, demonstrating expertise across {top_skills}.",
            f"Leverages {yoe} years of background as a {title}, particularly skilled in {top_skills}.",
            f"With {yoe} years under their belt, this {title} stands out for {top_skills}.",
            f"Shows a strong foundation in {top_skills} over a {yoe}-year career as a {title}.",
            f"Their {yoe} years in {title} roles heavily feature {top_skills}."
        ]
        exp_text = random.choice(exp_phrases)
        if random.random() > 0.5:
            return f"{tone_sent} {exp_text} {closing}"
        else:
            return f"{exp_text} {tone_sent} {closing}"
            
    elif structure_type == 2:
        # Structure 2 (The Fluid Integration): Combines tone and experience into a single fluid sentence
        fluid_phrases = [
            f"Specializing in {top_skills}, this {title}'s {yoe} years of experience makes them {tone_adj}.",
            f"Standing out as {tone_adj}, this {title} offers {yoe} years of experience focusing on {top_skills}.",
            f"With {yoe} years as a {title}, their strong background in {top_skills} positions them as {tone_adj}.",
            f"Their {yoe}-year track record as a {title} with {top_skills} makes them {tone_adj} for the role."
        ]
        return f"{random.choice(fluid_phrases)} {closing}"
        
    else:
        # Structure 3 (The Assessment Lead): Starts with the assessment/skills directly
        lead_phrases = [
            f"Considered {tone_adj} primarily due to their {yoe} years of experience in {top_skills} as a {title}.",
            f"They are {tone_adj}. A {yoe}-year {title} whose core strengths include {top_skills}.",
            f"Due to their solid {yoe} years working as a {title} with {top_skills}, they rank as {tone_adj}."
        ]
        return f"{random.choice(lead_phrases)} {closing}"

def process_candidates(candidates, jd_text: str, embeddings_path: str):
    print("Loading precomputed embeddings...")
    data = np.load(embeddings_path)
    emb_cand_ids = data['candidate_ids']
    emb_matrix = data['embeddings']
    jd_embedding = data['jd_embedding']
    
    # Map embeddings to candidates
    emb_dict = {}
    for idx, cid in enumerate(emb_cand_ids):
        emb_dict[cid] = emb_matrix[idx]
        
    print("Extracting JD keywords...")
    jd_tokens = extract_tokens(jd_text)
    
    doc_freq = Counter()
    candidate_tf_list = []
    candidate_lens = []
    
    # 1. Behavioral Twin Pass
    behavior_hashes = Counter()
    for c in candidates:
        sig = c.get('redrob_signals', {})
        h = f"{sig.get('interview_completion_rate', '')}_{sig.get('offer_acceptance_rate', '')}_{sig.get('notice_period_days', '')}_{sig.get('last_active_date', '')}"
        h += f"_{len(c.get('career_history', []))}"
        behavior_hashes[h] += 1
    
    for c in candidates:
        tokens_weighted = extract_candidate_text_with_weights(c)
        tf = Counter()
        unique_tokens = set()
        doc_len = 0
        for t, w in tokens_weighted:
            tf[t] += w
            unique_tokens.add(t)
            doc_len += w
            
        candidate_tf_list.append(tf)
        candidate_lens.append(doc_len)
        doc_freq.update(unique_tokens)
        
    N = len(candidates) + 1
    avgdl = sum(candidate_lens) / max(1, len(candidate_lens))
    
    jd_tokens = extract_tokens(jd_text)
    if 'rag' in jd_tokens:
        jd_tokens.extend(['vector', 'databas', 'pinecon', 'milvu', 'semantic', 'search', 'retriev'])
    if 'llm' in jd_tokens:
        jd_tokens.extend(['generat', 'ai', 'foundat', 'model'])

    # BM25 Parameters
    k1 = 1.5
    b = 0.75
    
    idf = {}
    for word in jd_tokens:
        df = doc_freq.get(word, 0)
        # Standard BM25 IDF formulation
        idf_val = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
        idf[word] = max(0.01, idf_val)
    
    jd_budget_match = re.search(r'budget.*?(\d+)(?:\s*lpa)?', jd_text, re.IGNORECASE)
    jd_budget = float(jd_budget_match.group(1)) if jd_budget_match else 20.0

    scored_candidates = []
    print("Scoring candidates (Stage 1 + Stage 2 + Semantic)...")
    
    for i, c in enumerate(candidates):
        cid = c.get('candidate_id')
        tf_dict = candidate_tf_list[i]
        doc_len = candidate_lens[i]
        
        # Stage 1: BM25 Similarity
        bm25_score = 0.0
        for w in set(jd_tokens):
            if w in tf_dict:
                tf = tf_dict[w]
                num = tf * (k1 + 1)
                den = tf + k1 * (1 - b + b * (doc_len / avgdl))
                bm25_score += idf.get(w, 0.0) * (num / den)
                
        # Scale BM25 to ~0-100. BM25 scores typically range 0-50 depending on query length.
        skills_score = min(bm25_score * 3.0, 100.0)
        
        # Semantic Score (replacing LLM)
        semantic_score = 0.0
        if cid in emb_dict:
            cos_sim = compute_cosine_similarity(jd_embedding, emb_dict[cid])
            semantic_score = max(0.0, min(100.0, cos_sim * 100.0))
            # Adjust baseline since cosine similarity between dense vectors can be high
            # We scale it slightly to spread out scores.
            semantic_score = ((semantic_score - 20) / 80.0) * 100.0 if semantic_score > 20 else semantic_score
            semantic_score = max(0.0, min(100.0, semantic_score))
        
        # Stage 2: Signals and Redrob Data
        signals = c.get('redrob_signals', {})
        assessments = signals.get('skill_assessment_scores', {})
        if assessments:
            avg_assessment = sum(assessments.values()) / len(assessments)
            if avg_assessment < 50:
                skills_score *= 0.8
        
        yoe = safe_get(c, 'profile', 'years_of_experience', default=0)
        exp_score = 50
        if 4 <= yoe <= 10:
            exp_score = 95
        elif yoe > 10:
            exp_score = 80
        elif yoe >= 2:
            exp_score = 65
            
        response_rate = float(signals.get('recruiter_response_rate', 0.5))
        github_score = float(signals.get('github_activity_score', -1))
        
        cult_score = 70 + (response_rate * 20)
        if github_score > 0:
            cult_score += min(10, github_score / 10)
            
        days_since_active = 0
        last_active = signals.get('last_active_date')
        if last_active:
            try:
                parts = last_active.split('-')
                days_since_active = (2026 - int(parts[0])) * 365 + (6 - int(parts[1])) * 30 + (1 - int(parts[2]))
            except:
                pass
        
        if days_since_active > 180:
            cult_score *= 0.5
            
        edu_score = 75
        elite_pattern = re.compile(r'\b(IIT|NIT|BITS|Stanford|MIT)\b', re.IGNORECASE)
        bachelor_end_year = None
        integrated_end_year = None
        
        for edu in c.get('education', []):
            tier = edu.get('tier', 'unknown')
            inst = edu.get('institution', '')
            
            if tier == 'tier_1' or elite_pattern.search(inst):
                edu_score = 95
            elif tier == 'tier_2':
                edu_score = max(edu_score, 85)
                
            degree = edu.get('degree', '').lower()
            if 'integrated' in degree or 'dual' in degree or 'm.tech (int' in degree:
                integrated_end_year = edu.get('end_year')
            elif 'b.' in degree or 'bsc' in degree or 'bachelor' in degree:
                bachelor_end_year = edu.get('end_year')
                
            if yoe < 2:
                grade = str(edu.get('grade', ''))
                gpa_match = re.search(r'(\d+\.\d+)', grade)
                if gpa_match:
                    gpa = float(gpa_match.group(1))
                    if gpa > 8.0:
                        edu_score = min(100, edu_score + 10)
                elif '%' in grade:
                    perc_match = re.search(r'(\d+)', grade)
                    if perc_match:
                        perc = float(perc_match.group(1))
                        if perc > 80:
                            edu_score = min(100, edu_score + 10)
                
        honeypot_penalty = 1.0
        career_history = c.get('career_history', [])
        
        career_len = sum(exp.get('duration_months', 0) for exp in career_history) / 12
        if yoe > 5 and career_len < 1:
            honeypot_penalty *= 0.5
            
        # 1. Consulting Firm Filter
        consulting_firms = {'tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini'}
        all_consulting = True
        for exp in career_history:
            company = exp.get('company', '').lower()
            if not any(cf in company for cf in consulting_firms):
                all_consulting = False
                break
        if career_history and all_consulting:
            honeypot_penalty *= 0.6
            
        # 2. Pure Research Filter
        all_research = True
        for exp in career_history:
            title = exp.get('title', '').lower()
            if not any(kw in title for kw in ['research', 'postdoc', 'phd', 'academic', 'scientist']):
                all_research = False
                break
        if career_history and all_research:
            honeypot_penalty *= 0.6
            
        # 3. Non-engineering Title Trap
        curr_title = safe_get(c, 'profile', 'current_title', default='').lower()
        non_eng_titles = ['hr ', 'human resources', 'marketing', 'designer', 'recruiter', 'sales', 'content writer', 'business analyst', 'operations']
        if any(t in curr_title for t in non_eng_titles):
            honeypot_penalty *= 0.2
            
        # 4. Shallow AI / LangChain Wrappers
        has_shallow_ai = False
        deep_ml = False
        for skill in c.get('skills', []):
            name = skill.get('name', '').lower()
            dur = skill.get('duration_months', 0)
            if ('langchain' in name or 'openai' in name or 'gpt' in name) and dur < 12:
                has_shallow_ai = True
            if any(kw in name for kw in ['pytorch', 'tensorflow', 'ranking', 'recommendation', 'machine learning', 'nlp', 'deep learning']) and dur >= 24:
                deep_ml = True
        if has_shallow_ai and not deep_ml:
            honeypot_penalty *= 0.8
            
        if bachelor_end_year:
            years_since_grad = 2026 - bachelor_end_year
            if yoe > (years_since_grad + 2):
                honeypot_penalty *= 0.4
                
        # New honeypot check: Impossible total experience based on graduation year
        grad_year = integrated_end_year if integrated_end_year else bachelor_end_year
        if grad_year:
            # typical grad age: 22 for bachelors, 23 for integrated.
            max_exp = (2026 - grad_year + 5) if integrated_end_year else (2026 - grad_year + 4)
            if yoe > max_exp:
                honeypot_penalty *= 0.3
                
        current_jobs = sum(1 for exp in career_history if exp.get('is_current'))
        if current_jobs >= 3:
            honeypot_penalty *= 0.7
            
        for skill in c.get('skills', []):
            endorsements = skill.get('endorsements', 0)
            duration = skill.get('duration_months', 1)
            if endorsements > 50 and duration < 3:
                honeypot_penalty *= 0.8
                break
                
        # 5. Keyword Stuffer Trap
        if len(c.get('skills', [])) > 40 and yoe < 3:
            honeypot_penalty *= 0.2
            
        # 6. Behavioral Twin Trap
        sig = c.get('redrob_signals', {})
        h = f"{sig.get('interview_completion_rate', '')}_{sig.get('offer_acceptance_rate', '')}_{sig.get('notice_period_days', '')}_{sig.get('last_active_date', '')}"
        h += f"_{len(c.get('career_history', []))}"
        if behavior_hashes[h] > 1:
            honeypot_penalty *= 0.1
                
        interview_rate = signals.get('interview_completion_rate', 0.5)
        offer_rate = signals.get('offer_acceptance_rate', 0.5)
        if interview_rate < 0: interview_rate = 0.5
        if offer_rate < 0: offer_rate = 0.5
        closing_multiplier = (interview_rate + offer_rate) / 2.0
        
        notice_days = signals.get('notice_period_days', 60)
        if notice_days <= 30:
            closing_multiplier *= 1.2
        elif notice_days > 60:
            closing_multiplier *= 0.8
            
        salary = signals.get('expected_salary_range_inr_lpa', {})
        max_salary = salary.get('max', 0)
        
        if max_salary > jd_budget * 1.2:
            closing_multiplier *= 0.7
            
        # Blend Stage 1 and Stage 2 scores into heuristic base
        stage1_score = ((skills_score * 0.4) + (exp_score * 0.3) + (edu_score * 0.1) + (cult_score * 0.2))
        stage1_score = stage1_score * honeypot_penalty * closing_multiplier
        stage1_score = min(max(stage1_score, 0.0), 100.0)
        
        stage2_signals_score = cult_score * closing_multiplier * honeypot_penalty
        stage2_signals_score = min(max(stage2_signals_score, 0.0), 100.0)
        
        # Blended Final Score: 0.40 * stage1_tfidf + 0.35 * semantic + 0.25 * stage2_signals
        final_score = 0.40 * stage1_score + 0.35 * semantic_score + 0.25 * stage2_signals_score
        final_score = min(max(final_score, 0.0), 100.0)
        
        scored_candidates.append({
            'candidate': c,
            'candidate_id': cid,
            'score': round(final_score, 2),
            'jd_budget': jd_budget,
            'honeypot_penalty': honeypot_penalty
        })
        
    return scored_candidates

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', required=True)
    parser.add_argument('--jd', required=True)
    parser.add_argument('--embeddings', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    print("Loading JD...")
    jd_text = load_jd(args.jd)
    
    print("Loading candidates...")
    candidates = []
    with open(args.candidates, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if line:
                c = json.loads(line)
                candidates.append(c)
            
    print(f"Loaded {len(candidates)} candidates. Processing...")
    scored = process_candidates(candidates, jd_text, args.embeddings)
    
    print("Sorting candidates...")
    scored.sort(key=lambda x: (-x['score'], int(x['candidate_id'].split("_")[1])))
    
    # Stage 4: Cross-Encoder Reranking
    top_300 = scored[:300]
    print("Running Cross-Encoder re-ranking on Top 300...")
    from sentence_transformers import CrossEncoder
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    def get_cand_text_for_ce(c):
        parts = [c.get('profile', {}).get('headline', ''), c.get('profile', {}).get('summary', '')]
        for exp in c.get('career_history', []):
            parts.append(exp.get('title', ''))
            parts.append(exp.get('description', ''))
        return " ".join(p for p in parts if p)
        
    pairs = [[jd_text, get_cand_text_for_ce(info['candidate'])] for info in top_300]
    ce_scores = cross_encoder.predict(pairs, show_progress_bar=False)
    
    import math
    for i, info in enumerate(top_300):
        # CE scores roughly range from -10 to +10. Sigmoid to 0-100 scale.
        ce_val = 1 / (1 + math.exp(-ce_scores[i])) * 100.0
        # Blend original Stage 1-3 score with Stage 4 CE score
        info['score'] = round((info['score'] * 0.4) + (ce_val * 0.6), 2)
        
    print("Final sort...")
    top_300.sort(key=lambda x: (-x['score'], int(x['candidate_id'].split("_")[1])))
    
    # Take exactly top 100
    top_100 = top_300[:100]
    
    for idx, c_info in enumerate(top_100):
        c_info['rank'] = idx + 1
        c_info['reasoning'] = generate_reasoning(
            c_info['candidate'],
            c_info['rank'],
            c_info['jd_budget'],
            c_info['honeypot_penalty']
        )

    fieldnames = ['candidate_id', 'rank', 'score', 'reasoning']

    print(f"Writing exactly 100 output rows to {args.out}...")

    with open(args.out, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for c_info in top_100:
            row = {
                'candidate_id': c_info['candidate_id'],
                'rank': c_info['rank'],
                'score': c_info['score'],
                'reasoning': c_info['reasoning']
            }
            writer.writerow(row)
            
    print("Done!")

if __name__ == "__main__":
    main()
