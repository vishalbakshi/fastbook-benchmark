# fastbook-benchmark
Information Retrieval QA Dataset

## Background

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

## Dataset Metrics

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

