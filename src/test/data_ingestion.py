from datasets import list_datasets, list_metrics, load_dataset, load_metric
from time import time
import re
import string
from copy import deepcopy


RAW_DATA_DIR = '~/BKTeam/NarrativeQA/data/raw/'
REMOVE_COLS = ['document.file_size', 'document.word_count', 'document.summary.text', 'document.summary.tokens', \
                'document.summary.url', 'document.summary.title']
RENAME_COLS = {'document.id': 'id',
                'document.url': 'url',
                'document.kind': 'kind',
                'document.start': 'start',
                'document.end': 'end',
                'document.text': 'context_text',
                'question.text': 'question_text',
                'question.tokens': 'question_tokens'}
REGEX_HTML_TAG = r'<[\w\/\s=\"\'\-\#\&\%\*\(\)\!\\\[\]\}\{\?\`\~.,;:]+>'
REGEX_BOOK_START1 = r'START OF (THIS|THE)?\s?PROJECT GUTENBERG'
REGEX_BOOK_START2 = r'THE SMALL PRINT! FOR PUBLIC DOMAIN'
REGEX_BOOK_END = r'END OF (THE|THIS)?\s?PROJECT GUTENBERG'


def preprocess_websource(text):
    ### FROM TOP
    is_lowercase = True
    idx = text.find('<body')
    print(idx)
    if idx != -1:
        text = text[idx:]
    idx = text.find('<BODY')
    print(idx)
    if idx != -1:
        text = text[idx:]
    idx = text.find("<pre>")
    print(idx)
    if idx == -1:
        is_lowercase = False
        idx = text.find("<PRE>")
        print(idx)
    text = text[idx+5:]
    if is_lowercase:
        idx = text.find("<pre>")
        print(idx)
        if idx != -1:
            text = text[idx+5:]
    else:
        idx = text.find("<PRE>")
        print(idx)
        if idx != -1:
            text = text[idx+5:]
    print(is_lowercase)

    ### FROM BOTTOM
    idx = text.find('</body>')
    text = text[:idx]
    if is_lowercase:
        idx = text.find("</pre>")
        text = text[:idx]
    else:
        idx = text.find("</pre>")
        text = text[:idx]
    idx = text.find("<!--Ad Banner")
    if idx != -1:
        text = text[:idx]
    return text

def remove_html_tags(sentence):
    tags = re.findall(REGEX_HTML_TAG, sentence)
    for tag in tags:
        sentence = sentence.replace(tag, "")
    sentence = sentence.strip()
    return sentence


def remove_punctuations(sentence):
    sentence = sentence.translate(str.maketrans("", "", string.punctuation))
    return sentence


def ingest_data(split):
    data = load_dataset('narrativeqa', split=split, cache_dir=RAW_DATA_DIR)
    print(f"Total data length: {len(data)} rows")
    # print(data.column_names)
    return data


def clean_data(data):
    processed_data = data.flatten()
    processed_data = processed_data.remove_columns(REMOVE_COLS)
    for col in list(RENAME_COLS.keys()):
        processed_data = processed_data.rename_column(col, RENAME_COLS.get(col))
    return processed_data


####################################################
### Preprocessing MOVIE data
### - Removing html tags
### - Keeping context from start_tokens to end_tokens
####################################################
def preprocess_movie(movie):
    start_tokens = movie['start']
    end_tokens = movie['end']
    print(movie['url'])
    print(start_tokens)
    print(end_tokens)
    context = movie['context_text']
    processed_context_lines = []
    start_tokens = remove_punctuations(start_tokens).strip()
    end_tokens = remove_punctuations(end_tokens).strip()

    print(context[:1000])
    print("\n===============\n")
    context = preprocess_websource(context)
    print(context[:1000])
    for line in context.split("\n"):
        text = remove_html_tags(line)
        if text:
            processed_context_lines.append(text)
    start_idx = -1
    for i in range(0, round(len(processed_context_lines)/2)):
        if start_tokens.split(" ")[0] in processed_context_lines[i]:
            check_text = " ".join(processed_context_lines[i:i+3])
            check_text = remove_punctuations(check_text).strip()
            if check_text.startswith(start_tokens):
                start_idx = i
                break
    end_idx = -1
    for i in range(len(processed_context_lines)-1, round(len(processed_context_lines)/2), -1):
        if end_tokens.split(" ")[0] in processed_context_lines[i]:
            check_text = " ".join(processed_context_lines[i-2:i+1])
            check_text = remove_punctuations(check_text).strip()
            if check_text.endswith(end_tokens):
                end_idx = i
                break
    
    if start_idx != -1:
        processed_context_lines = processed_context_lines[start_idx:]
    if end_idx != -1:
        processed_context_lines = processed_context_lines[:end_idx+1]
    processed_context = "\n".join(processed_context_lines)
    movie["context_text"] = processed_context
    # print(processed_context[500:1000])
    # print("\n\n")
    # print(processed_context[:200])
    # print("----------------------------------")
    return movie


####################################################
### Preprocessing BOOK data
### - Removing empty lines
### - Removing notes part like "<<this is electronic version...>>"
### - Identifying start_tokens and end_tokens
### - Keeping context from start_tokens to end_tokens
####################################################
def preprocess_book(book): 
    context = book['context_text']
    processed_context = ""

    for line in context.split('\n'):
        line = line.replace("_", "")
        if line:
            processed_context = processed_context + "\n" + line.strip()
    processed_context = processed_context[1:]

    flag = True
    while flag:
        idx1 = processed_context.find("<<THIS ELECTRONIC VERSION ")
        if idx1 != -1:
            idx2 = processed_context.find("FOR MEMBERSHIP.>>")
            processed_context = processed_context[:idx1] + processed_context[idx2+17:]
        else: 
            flag = False

    start_tokens = re.search(REGEX_BOOK_START1, processed_context)
    if not start_tokens:
        start_tokens = re.search(REGEX_BOOK_START2, processed_context)
    start_idx = processed_context.find(start_tokens.group())
    processed_context = processed_context[start_idx:]
    start_idx = processed_context.find("\n") + 1

    context_uppercase = deepcopy(processed_context).upper()
    
    end_tokens = re.search(REGEX_BOOK_END, context_uppercase)
    end_idx = context_uppercase.find(end_tokens.group(), round(len(context_uppercase)/2))
    end_idx = end_idx - 8 + context_uppercase[end_idx-8:end_idx].find("\n")
    
    processed_context = processed_context[start_idx:end_idx]
    book['context_text'] = processed_context
    return book


def preprocess_record(record):
    if record["kind"] == "movie":
        return preprocess_movie(record)
    else:
        return preprocess_book(record)


if __name__ == '__main__':
    start_time = time() 

    print("\n\nSTEP 1: Ingesting data...")
    raw_data = ingest_data('train')
    
    print("\n\nSTEP 2: Cleaning data...")
    cleaned_data = clean_data(raw_data)

    idx = [7119]
    print("\n\nSTEP 3: Preprocessing data...")
    cleaned_data = cleaned_data.select(idx)
    preprocessed_data = cleaned_data.map(preprocess_record)

    for i in range(0, len(idx)):
        print("\n------------------\n")
        print(f"{i}\n")
        print(preprocessed_data[i]["context_text"][:200])
        print("\n------------------\n")
        print(preprocessed_data[i]["context_text"][-200:])

    # prev_url = ""
    # for idx in range(9000, 10000):
    #     url = preprocessed_data[idx]["url"]
    #     if url != prev_url:
    #         prev_url = url
    #         print("\n\n")
    #         print(f"`````````` NUMBER{idx}")
    #         print(preprocessed_data[idx]["url"])
    #         # print(preprocessed_data[idx]["context_text"])
    #         print(preprocessed_data[idx]["context_text"][:200])
    #         print("----------")
    #         print(preprocessed_data[idx]["context_text"][-200:])

    # print("\n\nSTEP 3: Preprocessing MOVIE data...")
    # movie_data = cleaned_data.filter(lambda row: row['kind']=='movie')
    # # movie_data = movie_data.select([0,55])
    # print(f"Movie data length: {len(movie_data)} rows")
    # # processed_movie_data = movie_data.map(preprocess_movie)

    # print("\n\nSTEP 4: Preprocessing BOOK data...")
    # book_data = cleaned_data.filter(lambda row: row['kind']=='gutenberg')
    # print(f"Book data length: {len(book_data)} rows")
    # book_data = book_data.select([245, 5464, 6721, 2556])
    # processed_book_data = book_data.map(preprocess_book)


    wasted_seconds= round(time()-start_time)
    wasted_minutes = round((time()-start_time)/60)
    print(f"\n\nProgram takes {wasted_minutes} minutes {wasted_seconds} seconds\n")

