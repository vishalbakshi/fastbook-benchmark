# fastbook-benchmark
Information Retrieval QA Dataset

## Background

I created this dataset to evaluate different retrieval methods. I use the following code to load the `main` version of this dataset:

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

## Dataset Structure

Each dataset item has the following structure:

```json
{
    "chapter": 0,
    "question_number": 0,
    "question_text": "...",
    "gold_standard_answer": "...",
    "answer_context": [
        {
            "answer_component": "...", 
            "scoring_type": "simple",
            "context": [
                "...",
                "...",
                "..."
            ],
            "explicit_context": "true",
            "extraneous_answer": "false"
        },
        {
            "answer_component": "...", 
            "scoring_type": "simple",
            "context": [
                "...",
                "...",
                "..."
            ],
            "explicit_context": "false",
            "extraneous_answer": "true"
        }
    ],
        "question_context": []
}
```

Each dataset item represents one question/answer pair. 

`answer_context` contains the passages from the chapter relevant to the `gold_standard_answer`. 

Each `context` contains one or more passages relevant to the corresponding `answer_component`. (Ex: Ch4 Q30 has multiple strings in `context`; Q20 has many `answer_component`s). 

I tagged some `answer_component`s as an `extraneous_answer` since I felt they were extraneous to the goal of the question. (Ex: Ch13, Q38). 

Some `answer_component`s are flagged with `"explicit_context" = "false"` if the `context`s do not explicitly address the corresponding `answer_component` (Ex: Ch4, Q11) or if `context` is empty (Ex: Ch4, Q2). 

Some dataset items contain `question_context`, which is some passage from the chapter which addresses the `question_text`. (Ex: Ch4, Q27). 

## Video Series

1. [Introducing the fastbook-benchmark Information Retrieval QA Dataset](https://www.youtube.com/watch?v=VsVIy8k9rMU): Dataset and modified metrics overview.
2. Document Processing (coming soon): Converting notebooks to searchable chunks.
3. Full Text Search (coming soon): Basic search implementation.
4. Single Vector Search (coming soon): Dense retrieval methods.
5. ColBERT Search (coming soon): Late interaction retrieval approaches (ColBERTv2 and answerai-colbert-small-v1).

## Calculating Metrics

Since each question/answer pair has one or more `answer_component`s, I have chosen to modify the MRR@k and Recall@k calculations in my experiments and call them _Answer Component MRR@k_ and _Answer Component Recall@k_.

#### Modified MRR@k

The rank of the n-th passage, in the top-k passages, by which one or more `context`s of all `answer_component`s are retrieved. For example, if k=10 and a question has 4 `answer_component`s, and the corresponding `context`s are retrieved by the 9th-retrieved passage, the Modified MRR@10 is 1/9. If k=10 and only 3 of the `answer_component`s' `context`s are retrieved, Modified MRR@10 is 0.

```python
from ftfy import fix_text

def calculate_mrr(question, retrieved_passages, cutoff=10):
    retrieved_passages = retrieved_passages[:cutoff]
    highest_rank = 0

    for ans_comp in question["answer_context"]:
        contexts = ans_comp.get("context", [])
        component_found = False

        for rank, passage in enumerate(retrieved_passages, start=1):
            if any(fix_text(context) in fix_text(passage) for context in contexts):
                highest_rank = max(highest_rank, rank)
                component_found = True
                break

        if not component_found:
            return 0.0

    return 1.0/highest_rank if highest_rank > 0 else 0.0
```

#### Modified Recall@k

The percentage of `answer_component`s for which one or more `context`s are retrieved in the top-k passages. For example, if k=10 and a question has 4 `answer_component`s, and the corresponding `context`s for only 3 of them are retrieved in the top-10 passages, the Modified Recall@10 is 0.75. In this way, Modified Recall@k is more lenient than Modified MRR@k.

```python
from ftfy import fix_text

def calculate_recall(question, retrieved_passages, cutoff=10):
    retrieved_passages = retrieved_passages[:cutoff]
    ans_comp_found = []

    for ans_comp in question["answer_context"]:
        contexts = ans_comp.get("context", [])
        found = False

        for passage in retrieved_passages:
            if any(fix_text(context) in fix_text(passage) for context in contexts):
                found = True
                break

        ans_comp_found.append(found)

    return sum(ans_comp_found) / len(ans_comp_found)
```

## Dataset Statistics

The following key statistics are calculated in [this Colab notebook](https://colab.research.google.com/drive/1KCgmVljX4aURRyFGmnnK3U2cv_3BSqs7?usp=sharing). I'll do my best to update this section after dataset updates.

### Number of Questions per Chapter

|Chapter|# of Questions|
|:-:|:-:|
|1|30|
|2|26|
|4|31|
|8|23|
|9|27|
|10|20|
|13|34|
|**Total**|**191**|

### Number of `answer_component`s per Chapter

|Chapter|# of `answer_components`|
|:-:|:-:|
|1|78|
|2|58|
|4|73|
|8|31|
|9|48|
|10|27|
|13|42|
|**Total**|**357**|

### Average Number of `answer_component`s per Question

|Chapter|Avg # of `answer_components` per Question|
|:-:|:-:|
|1|2.6
|2|2.2
|4|2.4
|8|1.3
|9|1.8
|10|1.4
|13|1.2
|**Overall**|**1.9**

### Number of Empty `answer_component.context`s per Chapter

|Chapter|# of Empty `answer_component.context`s|
|:-:|:-:|
|1|8|
|2|5|
|4|8|
|8|1|
|9|1|
|10|1|
|13|1|
|**Total**|**25**|

### Number of `answer_component.explicit_context = "false"` per Chapter 

|Chapter|# of Implicit `answer_component`s|
|:-:|:-:|
|1|8
|2|5
|4|10
|8|3
|9|4
|10|4
|13|7
|**Total**|**41**|


### Number of `answer_component.extraneous_answer = "true"` per Chapter

|Chapter|# of Extraneous `answer_component`s|
|:-:|:-:|	
|1|7
|2|2
|4|8
|8|1
|9|0
|10|0
|13|1
|**Total**|**19**|

