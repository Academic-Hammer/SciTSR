import json

from scitsr.eval import json2Relations, eval_relations


def example():
  json_path = "/home/czwin32768/res/SciTSR/SciTSR/test/structure/1010.1982v1.2.json"
  with open(json_path) as fp: json_obj = json.load(fp)
  ground_truth_relations = json2Relations(json_obj, splitted_content=True)
  # your_relations should be a List of Relation.
  # Here we directly use the ground truth relations as the results.
  your_relations = ground_truth_relations
  precision, recall = eval_relations(
    gt=[ground_truth_relations], res=[your_relations], cmp_blank=True)
  f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0
  print("P: %.2f, R: %.2f, F1: %.2f" % (precision, recall, f1))


if __name__ == "__main__":
  example()  