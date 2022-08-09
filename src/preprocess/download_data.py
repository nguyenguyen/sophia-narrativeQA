import os
import re
import random
import logging
import spacy
from spacy import displacy
from langdetect import detect
from bs4 import BeautifulSoup
from datasets import load_dataset, Dataset, set_caching_enabled
import unidecode
import time
from utils import append_text, write_pickle

logging.basicConfig(level=logging.INFO)

sequences_distribution = [i for i in range(1, 1000000)]
entity_vocab = {}
entity_vocab_per_doc = {}
NER = spacy.load("en_core_web_sm")

NOT_ALLOWED_ENITITY_TYPE = ["DATE", "TIME", "CARDINAL"]

url_regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
rm_lst = [url_regex, "\t"]

start_token_extra_lst = ["produced by", "start of the project", "start of this project", "written by"]
end_token_extra_lst = ["start full license", "end of the project", "end of project"]

count = 0


def is_html(text, thresh_hold=2):
    html_tag = BeautifulSoup(text, "html.parser").find()
    if html_tag is None:
        return False
    elif len(html_tag) <= thresh_hold:
        return False
    else:
        return True


def remove_accent(text):
    return unidecode.unidecode(text)


def mask_entity(text):
    global entity_vocab_per_doc
    global sequences_distribution

    entities = NER(text)
    for word in entities.ents:
        if re.search(r"[A-Z][a-z]+", word.text) and word.label_ not in NOT_ALLOWED_ENITITY_TYPE:
            if word.text not in entity_vocab_per_doc:
                number = random.choice(sequences_distribution)
                sequences_distribution.remove(number)
                mask = f"[ent{number}]"
                entity_vocab_per_doc[word.text] = mask
            else:
                mask = entity_vocab_per_doc[word.text]

            text = text.replace(word.text, mask)
    return text


def slice_content(raw_text, start_token, end_token):
    start_token_lst = [re.sub('[\W ]+', ' ', i.lower()).strip() for i in start_token_extra_lst + [start_token]]
    end_token_lst = [re.sub('[\W ]+', ' ', i.lower()).strip() for i in end_token_extra_lst + [end_token]]

    start_pos_lst = [raw_text.lower().find(i) for i in start_token_lst]
    end_pos_lst = [raw_text.lower().rfind(i) for i in end_token_lst]

    min_start_pos = max(start_pos_lst)
    min_end_pos = len(raw_text)
    end_token1 = ""

    for start_pos in start_pos_lst:
        if start_pos != -1 and start_pos <= min_start_pos:
            min_start_pos = start_pos

    if min_start_pos == -1:
        min_start_pos = 0

    for index, end_pos in enumerate(end_pos_lst):
        if end_pos != -1 and end_pos <= min_end_pos:
            min_end_pos = end_pos
            end_token1 = end_token_lst[index]

    return raw_text[min_start_pos:min_end_pos + len(end_token1)]


def re_sub_text(text):
    text = re.sub("'ll", " will", text)
    text = re.sub("n't", " not", text)

    # remove special str
    for pattern in rm_lst:
        text = re.sub(pattern, "", text)

    # replace ?, ! by .
    text = re.sub(r'[!?]', '.', text).strip()

    # remove other punc and lower text
    # text = re.sub(r'[^a-zA-Z0-9\s.,()!?&\']', ' ', text.lower()).strip()
    text = re.sub(r'[^a-zA-Z\s.\']', ' ', text).strip()

    return text


def clean_sentence(sen):
    token = " [SEP] "
    sen = re.sub(r"\s+", " ", sen)
    sen = mask_entity(sen)
    sen = sen.replace(".", token)
    if not sen.strip().endswith("[SEP]"):
        sen = sen + token
    return sen.lower()


def preprocess_qa(sen):
    sen = mask_entity(sen)
    sen = sen.replace('?', '').replace('.', '') + " [SEP] "
    return sen.lower()


def split_and_concat_paragpraph(sliced_text):
    split_lst = sliced_text.split("\n")
    return_lst = []
    text = ""
    for sent in split_lst:
        if sent == "":
            if text != "":
                mask_text = clean_sentence(text)
                return_lst.append(mask_text)
                text = ""
        else:
            text = text + " " + sent

    if text.strip() != "":
        mask_text = clean_sentence(text)
        return_lst.append(mask_text)

    return return_lst


def preprocess_doc(raw_text, start_token, stop_token):
    if is_html(raw_text):
        scrtext_html = BeautifulSoup(raw_text, features="lxml").body.find('td', attrs={'class': 'scrtext'})
        if scrtext_html is not None:
            raw_text = scrtext_html.text
        else:
            raw_text = BeautifulSoup(raw_text, "lxml").text
    preprocessed_text = re_sub_text(raw_text)
    sliced_text = slice_content(preprocessed_text, start_token, stop_token)
    paragraphs = split_and_concat_paragpraph(sliced_text)
    return paragraphs


def clean_data(set_name, cleaned_data_path):
    global count
    logging.info(f"=== Cleaning {set_name} set and save to \'{cleaned_data_path}\'")

    dataset = load_dataset('narrativeqa', split=f'{set_name}')
    doc_id_set = set()
    for text in dataset:
        doc = text["document"]

        # doc
        if doc["id"] != "3ee65995071a0e70027e74a9b7735a734ba43bc7":  # french
            if doc["id"] not in doc_id_set:
                count += 1
                print("====== Processing doc id:", doc["id"], doc["kind"], count)
                doc_id_set.add(doc["id"])

                global entity_vocab
                global entity_vocab_per_doc

                entity_vocab_per_doc = {}

                if doc["id"] == "37fa67ed55fc62766b9a5f0edcafcc360131aebb":  # unicode format failed
                    correct_text = "".join([remove_accent(char) for char in doc["text"]])
                    correct_text = re.sub("AC/AA", "", correct_text)
                    correct_text = re.sub('i>>\?A-A>>A\?', "", correct_text)
                    correct_text = re.sub("_", "", correct_text)
                    clean_doc = preprocess_doc(correct_text, doc["start"], doc["end"])
                else:
                    clean_doc = preprocess_doc(doc["text"], doc["start"], doc["end"])

                for part in clean_doc:
                    if part.strip() != "":
                        append_text(part.strip(), cleaned_data_path, f"{doc['id']}.txt")
                if len(entity_vocab_per_doc) > 0:
                    entity_vocab[doc["id"]] = entity_vocab_per_doc

            # qa
            # print("=== Processing qa:", doc["id"], doc["kind"])
            qa_lst = []
            mask_text = preprocess_qa(text["question"]["text"])
            qa_lst.append(mask_text.replace('?', ''))
            for ans in text["answers"]:
                mask_text = preprocess_qa(ans["text"])
                qa_lst.append(mask_text)
            append_text("".join(qa_lst), f"{cleaned_data_path}/qa", f"{doc['id']}.txt")


if __name__ == '__main__':
    dataset_splits = ["train"]
    # dataset_splits = ["train", "test", "validation"]
    for split in splits:
        count = 0
        folder_path = f"./data/raw/{split}"
        # vocab_folder = "./data/vocab/"
        if not os.path.isdir(folder_path):
            clean_data(split, folder_path)
        # if len(entity_vocab) > 0:
        #     write_pickle(entity_vocab, vocab_folder, "entities")




