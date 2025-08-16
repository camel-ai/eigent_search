#%%
# load json
import json
from datasets import load_dataset, Dataset

new_result_json = "/Users/yuqichengzhu/Documents/Research/DeepResearch/results/version1.0_simpleqa_from=0_to=50.json"
with open(new_result_json, "r") as f:
    new_result = json.load(f)

main_result_json = "/Users/yuqichengzhu/Documents/Research/DeepResearch/results/versionmain_simpleqa_from=0_to=30.json"
with open(main_result_json, "r") as f:
    main_result = json.load(f)

# accuracy evaluation
def evaluate_accuracy(results, length=10):
    results = results[:length]
    correct = 0
    total = len(results)
    for result in results:
        if result["grade"] == "CORRECT":
            correct += 1
    return correct / total if total > 0 else 0

def evaluate_accuracy_givenattempt(results, length=10):
    results = results[:length]
    correct = 0
    attempt = 0
    for result in results:
        if result["grade"] == "CORRECT":
            correct += 1
        if result["grade"] != "NOT_ATTEMPTED":
            attempt += 1
    return correct / attempt if attempt > 0 else 0

def evaluate_attempt_rate(results, length=10):
    results = results[:length]
    attempted = 0
    total = len(results)
    for result in results:
        if result["grade"] != "NOT_ATTEMPTED":
            attempted += 1
    return attempted / total if total > 0 else 0

print(f"New Result Accuracy: {evaluate_accuracy(new_result, 30):.2f}")
print(f"Main Result Accuracy: {evaluate_accuracy(main_result, 30):.2f}")

print(f"New Result Attempted Accuracy: {evaluate_accuracy_givenattempt(new_result, 30):.2f}")
print(f"Main Result Attempted Accuracy: {evaluate_accuracy_givenattempt(main_result, 30):.2f}")

print(f"New Result Attempt Rate: {evaluate_attempt_rate(new_result, 30):.2f}")
print(f"Main Result Attempt Rate: {evaluate_attempt_rate(main_result, 30):.2f}")

dataset = load_dataset("basicv8vc/SimpleQA")

test_samples = list(dataset["test"])[:30]

questions = [sample["problem"] for sample in test_samples]
main_answers = [sample["grade"] for sample in main_result[:30]]
new_answers = [sample["grade"] for sample in new_result[:30]]

import pandas as pd

df = pd.DataFrame({
    "Question": questions,
    "Main Result": main_answers,
    "New Result": new_answers
})
# save as excel file
df.to_excel("simpleqa_comparison.xlsx", index=False)
# %%
