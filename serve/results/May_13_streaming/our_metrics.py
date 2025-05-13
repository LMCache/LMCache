from rouge_score import rouge_scorer

def evaluate_answer(generated_answer, reference_answer):
    """
    Evaluate the model-generated answer vs. reference using the ROUGE metric.
    Args:
        generated_answer (str): The generated answer from the model.
        reference_answer (str): The reference answer for evaluation.

    Returns:
        dict: A dictionary containing ROUGE-1, ROUGE-2, and ROUGE-L scores.
    """
    if not reference_answer:
        return None  # No reference available for evaluation
    
    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Calculate ROUGE scores
    scores = scorer.score(reference_answer, generated_answer)
    # Format the scores to return
    return scores["rougeL"].fmeasure

from collections import Counter

def f1_score(prediction, ground_truth):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
