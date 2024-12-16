import requests 
import json
import os
import sqlite3
import pandas as pd
import torch.nn.functional as F
from ragatouille.data import CorpusProcessor
from ragatouille import RAGPretrainedModel
from ftfy import fix_text
corpus_processor = CorpusProcessor()

# Document processing
urls = {
    '1':  'https://raw.githubusercontent.com/vishalbakshi/fastbook-benchmark/refs/heads/main/data/01_intro.ipynb',
    '2':  'https://raw.githubusercontent.com/vishalbakshi/fastbook-benchmark/refs/heads/main/data/02_production.ipynb',
    '4':  'https://raw.githubusercontent.com/vishalbakshi/fastbook-benchmark/refs/heads/main/data/04_mnist_basics.ipynb',
    '8':  'https://raw.githubusercontent.com/vishalbakshi/fastbook-benchmark/refs/heads/main/data/08_collab.ipynb',
    '9':  'https://raw.githubusercontent.com/vishalbakshi/fastbook-benchmark/refs/heads/main/data/09_tabular.ipynb',
    '10': 'https://raw.githubusercontent.com/vishalbakshi/fastbook-benchmark/refs/heads/main/data/10_nlp.ipynb',
    '13': 'https://raw.githubusercontent.com/vishalbakshi/fastbook-benchmark/refs/heads/main/data/13_convolutions.ipynb'
}

nbs = {
    '1': '01_intro.ipynb',
    '2': '02_production.ipynb',
    '4': '04_mnist_basics.ipynb',
    '8': '08_collab.ipynb',
    '9': '09_tabular.ipynb',
    '10': '10_nlp.ipynb',
    '13': '13_convolutions.ipynb'
}

def download_file(url, fn): 
    with open(fn, 'wb') as file: file.write(requests.get(url).content)

def download_data():
    for chapter, nb in nbs.items(): download_file(urls[chapter], fn=nb)
    return nbs

def chunk_string(text, n):
    skip = int(len(text) / n)
    return [text[i:i + skip] for i in range(0, len(text), skip)]

def notebook_to_string(path):
    with open(path, 'r', encoding='utf-8') as f: notebook = json.load(f)
    all_text = ''
    
    for cell in notebook['cells']:
        if cell['cell_type'] == 'markdown' and any('## Questionnaire' in line for line in cell['source']): break
        if cell['cell_type'] in ['markdown', 'code']: all_text += ''.join(cell['source']) + '\n'

    return all_text

def get_data(nbs):
    data = {}
    n_chars = 0
  
    for chapter, nb in nbs.items():
        data[chapter] = chunk_string(notebook_to_string(nb), 2)
        for c in data[chapter]: n_chars += len(c)
  
    assert n_chars == 503769
    return data

def process_documents(text, chunk_size):
    documents = corpus_processor.process_corpus(text, chunk_size=chunk_size)
    documents = [doc['content'] for doc in documents]
    return documents

# Full text search
def delete_db(): 
    if os.path.exists("fastbook.db"): os.remove("fastbook.db")

def load_data(documents, db_path, chapter=1):
    # create virtual table
    if not os.path.exists(db_path):
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            cur.execute("""
                    CREATE VIRTUAL TABLE fastbook_text
                    USING FTS5(chapter, text);
                    """)
            conn.commit()

    # load in the chunks for each chapter
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        for doc in documents: cur.execute("INSERT INTO fastbook_text(chapter, text) VALUES (?, ?)", (chapter, doc))
        conn.commit()
        res = cur.execute("SELECT * FROM fastbook_text WHERE chapter = ?", (chapter,)).fetchall()
        assert len(res) == len(documents)

    return True

def full_text_search(kw_df, limit=10):
    all_results = []
    with sqlite3.connect('fastbook.db') as conn:
        cur = conn.cursor()

        for _, row in kw_df.iterrows():
            chapter = row['chapter']
            keywords = row['keywords'].replace('"', '').split()
            matchstr = ' OR '.join([f'"{kw.strip(",")}"' for kw in keywords])

            q = f"""
                    SELECT text, rank
                    FROM fastbook_text
                    WHERE fastbook_text MATCH ?
                    AND chapter = ?
                    ORDER BY rank
                    LIMIT ?
            """
            res = cur.execute(q, (matchstr, str(chapter), limit)).fetchall()
            res = [item[0] for item in res]
            assert len(res) <= limit
            all_results.append(res)

    assert len(all_results) == len(kw_df)
    return all_results

def load_benchmark():
    url = 'https://raw.githubusercontent.com/vishalbakshi/fastbook-benchmark/refs/heads/main/fastbook-benchmark.json'
    download_file(url=url, fn="fastbook-benchmark.json")
    with open('fastbook-benchmark.json', 'r') as f: benchmark = json.load(f)
    assert len(benchmark['questions']) == 191
    return benchmark

def load_keywords():
    url = 'https://raw.githubusercontent.com/vishalbakshi/fastbook-benchmark/refs/heads/main/examples/fts_keywords.csv'
    kw_df = pd.read_csv(url)
    assert kw_df.shape == (191,4)
    return kw_df

# Scoring retrieval results
def modified_mrr(question, results, cutoff=10):
    retrieved_passages = results[:cutoff]
    highest_rank = 0
    for ac in question["answer_context"]: 
        ctxs = ac.get("context", [])
        component_answered = False

        for rank, passage in enumerate(retrieved_passages, start=1):
            if any(fix_text(ctx) in fix_text(passage) for ctx in ctxs): 
                highest_rank = max(highest_rank, rank)
                component_answered = True
                break

        if not component_answered: return 0.0

    return 1.0/highest_rank if highest_rank > 0 else 0.0

def modified_recall(question, results):
    components_found = []
    for ac in question["answer_context"]: 
        ctxs = ac.get("context", [])
        found = False

        for rank, passage in enumerate(results, start=1):
            if any(fix_text(ctx) in fix_text(passage) for ctx in ctxs): 
                found = True
                break

        components_found.append(found)

    return sum(components_found) / len(components_found)

def score_retrieval(benchmark, results):
    mrrs = []
    recalls = []

    for i,q in enumerate(benchmark['questions']):
        mrr = modified_mrr(q, results[i])
        recall = modified_recall(q, results[i])
        mrrs.append(mrr)
        recalls.append(recall)

    assert len(mrrs) == len(benchmark["questions"])
    assert len(recalls) == len(benchmark["questions"])

    mrrs = pd.Series(mrrs)
    recalls = pd.Series(recalls)
    
    return mrrs, recalls

# Single vector search
def prep_questions(benchmark):
    questions = {}
    for q in benchmark["questions"]: questions.setdefault(str(q["chapter"]), []).append(q["question_text"].strip('"\''))
    return questions

def single_vector_retrieval(nbs, all_docs, data_embs, qs_embs, topk=10):
    results = []
    for chapter in nbs.keys():
        idxs = F.cosine_similarity(qs_embs[chapter].unsqueeze(1), data_embs[chapter].unsqueeze(0), dim=2).sort(descending=True)[1]
        top_k_idxs = idxs[:, :topk]
        top_k_chunks = [[all_docs[chapter][idx.item()] for idx in row_idxs]for row_idxs in top_k_idxs]
        results.extend(top_k_chunks)

    assert len(results) == 191
    for res in results: assert len(res) <= topk
    return results

# ColBERT search
def index_free_retrieval(nbs, data, questions, model_nm="colbert-ir/colbertv2.0", topk=10, chunk_size=500):
    results = []
    for chapter in nbs.keys():
        chapter_results = []
        RAG = RAGPretrainedModel.from_pretrained(model_nm)
        documents = process_documents(data[chapter], chunk_size=chunk_size)
        RAG.encode(documents, document_metadatas=[{"chapter": chapter} for _ in range(len(documents))])
        top_k = min(topk, len(documents))
        for q in questions[chapter]: 
            res = RAG.search_encoded_docs(query = q.strip('"\''), k=top_k)
            res = [r['content'] for r in res]
            chapter_results.append(res)
        results.extend(chapter_results)

    assert len(results) == 191
    for res in results: assert len(res) <= 10
    return results
