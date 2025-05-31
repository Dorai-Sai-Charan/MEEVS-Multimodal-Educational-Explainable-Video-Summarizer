from transformers import BartTokenizer, BartForConditionalGeneration
from datasets import load_dataset
import evaluate
import matplotlib.pyplot as plt
from tqdm import tqdm

# ✅ Load trained model
model_path = "bart_large_arxiv_model"  # ← update this path if different
tokenizer = BartTokenizer.from_pretrained(model_path)
model = BartForConditionalGeneration.from_pretrained(model_path)

# ✅ Load test data
dataset = load_dataset("ccdv/arxiv-summarization")
test_data = dataset["test"]

# ✅ Load metrics
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")
bertscore = evaluate.load("bertscore")

# ✅ Store results
N = 100
predictions, references = [], []
rouge_1s, rouge_2s, rouge_Ls, rouge_Lsums = [], [], [], []
meteor_scores = []
bert_precisions, bert_recalls, bert_f1s = [], [], []

print(f"\n🔍 Evaluating on {N} samples...\n")
for i in tqdm(range(N)):
    article = test_data[i]["article"]
    reference = test_data[i]["abstract"]

    # Tokenize + generate
    inputs = tokenizer(article, return_tensors="pt", truncation=True, max_length=1024)
    summary_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        num_beams=4,
        length_penalty=1.1,
        no_repeat_ngram_size=3
    )
    pred = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    predictions.append(pred)
    references.append(reference)

    # Compute metrics
    rouge_result = rouge.compute(predictions=[pred], references=[reference])
    rouge_1s.append(rouge_result["rouge1"])
    rouge_2s.append(rouge_result["rouge2"])
    rouge_Ls.append(rouge_result["rougeL"])
    rouge_Lsums.append(rouge_result["rougeLsum"])

    meteor_result = meteor.compute(predictions=[pred], references=[reference])
    meteor_scores.append(meteor_result["meteor"])

    bert_result = bertscore.compute(predictions=[pred], references=[reference], lang="en")
    bert_precisions.append(bert_result["precision"][0])
    bert_recalls.append(bert_result["recall"][0])
    bert_f1s.append(bert_result["f1"][0])

# ✅ Print average scores
print("\n📊 Final Average Evaluation Scores:")
print("\n🔹 ROUGE:")
print(f"ROUGE-1:     {sum(rouge_1s)/N:.4f}")
print(f"ROUGE-2:     {sum(rouge_2s)/N:.4f}")
print(f"ROUGE-L:     {sum(rouge_Ls)/N:.4f}")
print(f"ROUGE-Lsum:  {sum(rouge_Lsums)/N:.4f}")

print(f"\n🔹 METEOR:   {sum(meteor_scores)/N:.4f}")

print("\n🔹 BERTScore:")
print(f"Precision:   {sum(bert_precisions)/N:.4f}")
print(f"Recall:      {sum(bert_recalls)/N:.4f}")
print(f"F1 Score:    {sum(bert_f1s)/N:.4f}")
