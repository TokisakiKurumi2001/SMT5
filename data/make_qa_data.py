from datasets import load_dataset
from tqdm import tqdm
import pandas as pd

if __name__ == "__main__":
    dataset = load_dataset('mkqa')
    langs = ['ar', 'de', 'en', 'es', 'fi', 'fr', 'ja', 'ko', 'pt', 'ru', 'sv', 'th', 'tr', 'vi', 'zh_cn']
    data_dict = {"query": [], "answer": [], "idx": [], "lang": []}
    cnt = 0
    for example in tqdm(dataset['train']):
        flag = False
        # queries
        queries = example["queries"]
        answers = example['answers']
        for lang in langs:
            query = queries[lang]
            answer = answers[lang][0]['text']
            if query is None or answer is None:
                flag = True
                break
            data_dict['query'].append(query)
            data_dict['lang'].append(lang)
            data_dict['idx'].append(cnt)
            data_dict['answer'].append(answer)
        if flag:
            continue
        cnt += 1
    df = pd.DataFrame(data_dict)
    df = df.sort_values(['lang'], ascending=True)
    df.to_csv('mkqa.train.csv', index=False)
