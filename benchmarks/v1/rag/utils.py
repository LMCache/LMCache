# Standard
from enum import Enum
from logging import Logger
import collections
import json
import logging
import re
import string

# Third Party
from rouge_score import rouge_scorer


def build_format(color):
    reset = "\x1b[0m"
    underline = "\x1b[3m"
    return (
        f"{color}[%(asctime)s] %(levelname)s:{reset} %(message)s "
        + f"{underline}(%(filename)s:%(lineno)d:%(name)s){reset}"
    )


class CustomFormatter(logging.Formatter):
    grey = "\x1b[1m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    FORMATS = {
        logging.DEBUG: build_format(grey),
        logging.INFO: build_format(green),
        logging.WARNING: build_format(yellow),
        logging.ERROR: build_format(red),
        logging.CRITICAL: build_format(bold_red),
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def init_logger(name: str, log_level=logging.DEBUG) -> Logger:
    logger = logging.getLogger(name)

    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)
    logger.setLevel(logging.DEBUG)

    return logger


class PromptBuildMethodType(Enum):
    QA = 0
    FEW_SHOT = 1


def load_dataset(dataset_path):
    print("Loading dataset:", dataset_path)
    with open(dataset_path) as f:
        return json.load(f)


def normalize_question(question):
    if not question.endswith("?"):
        question = question + "?"

    return question[0].lower() + question[1:]


def parse_generation(s):
    s = s.lstrip("\n").split("\n")[0]
    if s.startswith("Yes") or s.startswith("yes"):
        s = "Yes"
    elif (s.split()[0]).startswith("No") or (s.split()[0]).startswith("no"):
        s = "No"
    return s


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def build_qa_prompt(example, query_prompt):
    q = normalize_question(example["question"])
    doc_prompts = [f"{ctx['title']}\n\n{ctx['text']}\n\n" for ctx in example["ctxs"]]
    q_prompt = f"{query_prompt}{q}\nAnswer:"
    return doc_prompts, q_prompt


def build_fewshot_prompt(example):
    q = "\n\n" + example["question"]
    doc_prompts = [f"{ctx['text']}" for ctx in example["ctxs"]]
    q_prompt = f"{q}"
    return doc_prompts, q_prompt


def compute_f1(a_pred, a_gold, tokenizer):
    a_pred = parse_generation(a_pred)
    gold_toks = tokenizer.encode(normalize_answer(a_gold))[1:]
    pred_toks = tokenizer.encode(normalize_answer(a_pred))[1:]
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_rl(pred, gold):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rougeL = scorer.score(gold, pred)["rougeL"].fmeasure
    return rougeL


def build_rag_prompt_tokens(
    system_prompt_tokens,
    doc_tokens_list,
    query_prompt_tokens,
    separator_tokens,
):
    """Build RAG prompt using token IDs instead of strings for offline processing"""
    # Combine all tokens: system_prompt + separator + docs + separator + query_prompt
    all_tokens = system_prompt_tokens[:]
    
    for doc_tokens in doc_tokens_list:
        all_tokens.extend(separator_tokens)
        all_tokens.extend(doc_tokens)
    
    all_tokens.extend(separator_tokens)
    all_tokens.extend(query_prompt_tokens)
    
    return all_tokens


def build_rag_prompt(
    system_prompt,
    example,
    query_prompt,
    separator: str,
    prompt_build_method: PromptBuildMethodType,
):
    doc_prompts = None
    q_prompt = None
    if prompt_build_method == PromptBuildMethodType.FEW_SHOT:
        doc_prompts, q_prompt = build_fewshot_prompt(example)
    elif prompt_build_method == PromptBuildMethodType.QA:
        doc_prompts, q_prompt = build_qa_prompt(example, query_prompt)
    else:
        raise ValueError(f"Invalid prompt build method {prompt_build_method}")
    final_prompt = separator.join([system_prompt] + doc_prompts + [q_prompt])
    return final_prompt, doc_prompts
