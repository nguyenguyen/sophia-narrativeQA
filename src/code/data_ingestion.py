from datasets import list_datasets, list_metrics, load_dataset, load_metric
from time import time
import re
import string


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
    context = movie['context_text']
    processed_context_lines = []
    start_tokens = remove_punctuations(start_tokens).strip()
    end_tokens = remove_punctuations(end_tokens).strip()

    for line in context.split("\n"):
        text = remove_html_tags(line)
        if text:
            processed_context_lines.append(text)

    start_idx = 0
    for i in range(0, len(processed_context_lines)):
        if start_tokens.split(" ")[0] in processed_context_lines[i]:
            check_text = " ".join(processed_context_lines[i:i+3])
            check_text = remove_punctuations(check_text).strip()
            if check_text.startswith(start_tokens):
                start_idx = i
                break
    end_idx = 0
    for i in range(len(processed_context_lines)-1, round(len(processed_context_lines)/2), -1):
        if end_tokens.split(" ")[0] in processed_context_lines[i]:
            check_text = " ".join(processed_context_lines[i-2:i+1])
            check_text = remove_punctuations(check_text).strip()
            if check_text.endswith(end_tokens):
                end_idx = i
                break
        
    processed_context = "\n".join(processed_context_lines[start_idx:end_idx+1])
    movie["context_text"] = processed_context
    return movie


if __name__ == '__main__':
    start_time = time() 

    print("\n\nSTEP 1: Ingesting data...")
    raw_data = ingest_data('train')
    
    print("\n\nSTEP 2: Cleaning data...")
    cleaned_data = clean_data(raw_data)

    print("\n\nSTEP 3: Preprocessing MOVIE data...")
    movie_data = cleaned_data.filter(lambda row: row['kind']=='movie')
    movie_data = movie_data.select([0,55])
    print(f"Movie data length: {len(movie_data)} rows")
    processed_movie_data = movie_data.map(preprocess_movie)

    print("\n\nSTEP 4: Preprocessing BOOK data...")
    book_data = cleaned_data.filter(lambda row: row['kind']=='gutenberg')
    print(f"Book data length: {len(book_data)} rows")


    wasted_seconds= round(time()-start_time)
    wasted_minutes = round((time()-start_time)/60)
    print(f"\n\nProgram takes {wasted_minutes} minutes {wasted_seconds} seconds\n")

