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

This dataset currently contains 191 questions (from [fastbook](https://github.com/fastai/fastbook/tree/master) end-of-chapter Questionnaires) for 7 chapters (1, 2, 4, 8, 9, 10, and 13). The `gold_standard_answer` for each question is verbatim from the chapter's corresponding solutions Wiki on the fastai Forums:

- [Chapter 1 Solutions](https://forums.fast.ai/t/fastbook-chapter-1-questionnaire-solutions-wiki/65647)
- [Chapter 2 Solutions](https://forums.fast.ai/t/fastbook-chapter-2-questionnaire-solutions-wiki/66392)
- [Chapter 4 Solutions](https://forums.fast.ai/t/fastbook-chapter-4-questionnaire-solutions-wiki/67253)
- [Chapter 8 Solutions](https://forums.fast.ai/t/fastbook-chapter-8-questionnaire-solutions-wiki/69926)
- [Chapter 9 Solutions](https://forums.fast.ai/t/fastbook-chapter-9-questionnaire-solutions-wiki/69932)
- [Chapter 10 Solutions](https://forums.fast.ai/t/fastbook-chapter-10-questionnaire-solutions-wiki/70506)
- [Chapter 13 Solutions](https://forums.fast.ai/t/fastbook-chapter-13-questionnaire-wiki/91761)

