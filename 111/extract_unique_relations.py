import json
from typing import Set, Dict, Any

# 定义允许的类型
ALLOWED_TYPES = {'Fruit', 'Compound', 'Flavor', 'Enzyme', 'Reaction', 'Metabolism'}

def extract_unique_relations(input_file: str, output_file: str) -> None:
    """
    从输入的jsonl文件中提取唯一的关系类型，并输出到新的jsonl文件中
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
    """
    unique_relations: Set[str] = set()
    unique_relation_dicts: Dict[str, Dict[str, str]] = {}
    
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            if 'triples' in data:
                for triple in data['triples']:
                    subject_type = triple['subject_type']
                    object_type = triple['object_type']
                    
                    # 检查类型是否在允许列表中
                    if subject_type in ALLOWED_TYPES and object_type in ALLOWED_TYPES:
                        # 创建关系字典
                        relation_dict = {
                            'subject_type': subject_type,
                            'relation_type': triple['relation_type'],
                            'object_type': object_type
                        }
                        # 将字典转换为字符串用于去重
                        relation_str = json.dumps(relation_dict, sort_keys=True)
                        if relation_str not in unique_relations:
                            unique_relations.add(relation_str)
                            unique_relation_dicts[relation_str] = relation_dict
    
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for relation_dict in unique_relation_dicts.values():
            f.write(json.dumps(relation_dict) + '\n')
    
    print(f"提取到 {len(unique_relation_dicts)} 个唯一关系")
    print(f"唯一关系已提取到 {output_file}")

if __name__ == '__main__':
    input_file = 'data/qwen_cluster_mapped.jsonl'
    output_file = 'data/unique_relations.jsonl'
    extract_unique_relations(input_file, output_file) 