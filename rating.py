import bert_score
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default="./eval_output.json")
parser.add_argument('--output', type=str, default="./score_output.json")
args = parser.parse_args()

with open(args.input, "r", encoding="utf-8") as f:
    test_data = json.load(f)

test_data["scores"] = dict()
test_data["scores"]["en"] = dict()
test_data["scores"]["zh"] = dict()
test_data["scores"]["all"] = dict()
en_scorer = bert_score.BERTScorer(lang="en", batch_size=3)
zh_scorer = bert_score.BERTScorer(lang="zh", batch_size=3)

en_sum_p = 0
en_sum_r = 0
en_sum_f1 = 0
for data in test_data["en"]:
    print("For instruction: " + data["instruction"])
    (P, R, F1) = en_scorer.score([data["output"].strip()], [data["output_gpt"].strip()], return_hash=False)
    en_sum_p += P.mean().item()
    en_sum_r += R.mean().item()
    en_sum_f1 += F1.mean().item()
    data["score"] = dict()
    data["score"]["P"] = P.mean().item()
    data["score"]["R"] = R.mean().item()
    data["score"]["F1"] = F1.mean().item()
    print(f"Precision = {P.mean()}, Recall = {R.mean()}, F1 = {F1.mean()}")
    print()

size = len(test_data["en"])
avg_p = en_sum_p/size
avg_r = en_sum_r/size
avg_f1 = en_sum_f1/size
test_data["scores"]["en"]["P"] = avg_p
test_data["scores"]["en"]["R"] = avg_r
test_data["scores"]["en"]["F1"] = avg_f1
print(f"EN Average: Precision = {avg_p}, Recall = {avg_r}, F1 = {avg_f1}")

zh_sum_p = 0
zh_sum_r = 0
zh_sum_f1 = 0
for data in test_data["zh"]:
    print("For instruction: " + data["instruction"])
    (P, R, F1) = zh_scorer.score([data["output"].strip()], [data["output_gpt"].strip()], return_hash=False)
    zh_sum_p += P.mean().item()
    zh_sum_r += R.mean().item()
    zh_sum_f1 += F1.mean().item()
    data["score"] = dict()
    data["score"]["P"] = P.mean().item()
    data["score"]["R"] = R.mean().item()
    data["score"]["F1"] = F1.mean().item()
    print(f"Precision = {P.mean()}, Recall = {R.mean()}, F1 = {F1.mean()}")
    print()

size = len(test_data["zh"])
avg_p = zh_sum_p/size
avg_r = zh_sum_r/size
avg_f1 = zh_sum_f1/size
test_data["scores"]["zh"]["P"] = avg_p
test_data["scores"]["zh"]["R"] = avg_r
test_data["scores"]["zh"]["F1"] = avg_f1
print(f"ZH Average: Precision = {avg_p}, Recall = {avg_r}, F1 = {avg_f1}")

size = len(test_data["en"]) + len(test_data["zh"])
avg_p = (en_sum_p + zh_sum_p)/size
avg_r = (en_sum_r + zh_sum_r)/size
avg_f1 = (en_sum_f1 + zh_sum_f1)/size
test_data["scores"]["all"]["P"] = avg_p
test_data["scores"]["all"]["R"] = avg_r
test_data["scores"]["all"]["F1"] = avg_f1

print(f"Average: Precision = {avg_p}, Recall = {avg_r}, F1 = {avg_f1}")
open(args.output, "w+").write(json.dumps(test_data, ensure_ascii=False))
