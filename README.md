# fastbook-benchmark
Information Retrieval QA Dataset

I created this dataset to evaluate different retrieval methods. I use the following code to load this dataset:

```python
def download_file(url, fn): 
    with open(fn, 'wb') as file: file.write(requests.get(url).content)

url = 'https://raw.githubusercontent.com/vishalbakshi/fastbook-benchmark/refs/heads/main/fastbook-benchmark.json'
download_file(url=url, fn="fastbook-benchmark.json")

def load_benchmark():
    with open('fastbook-benchmark.json', 'r') as f: benchmark = json.load(f)
    return benchmark

benchmark = load_benchmark()
assert len(benchmark['questions']) == 191
```

