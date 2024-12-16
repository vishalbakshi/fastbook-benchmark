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

def download_data():
    for chapter, nb in nbs.items(): 
        download_file(urls[chapter], fn=nb)
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
