import json
from collections import Counter

# 定义允许的类型
ALLOWED_TYPES = {'Fruit', 'Compound', 'Flavor', 'Enzyme', 'Reaction', 'Metabolism'}

def verify_types(file_path):
    total_triples = 0
    correct_triples = 0
    incorrect_triples = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            for triple in data['triples']:
                total_triples += 1
                subject_type = triple['subject_type']
                object_type = triple['object_type']
                
                # 只要有一个类型不在允许列表中就算不正确
                if subject_type not in ALLOWED_TYPES or object_type not in ALLOWED_TYPES:
                    incorrect_triples += 1
                else:
                    correct_triples += 1
    
    return total_triples, correct_triples, incorrect_triples

if __name__ == "__main__":
    file_path = "data/qwen_cluster_mapped.jsonl"
    total, correct, incorrect = verify_types(file_path)
    
    print(f"总三元组数量: {total}")
    print(f"正确的三元组数量: {correct}")
    print(f"不正确的三元组数量: {incorrect}")
