import json
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Dict, List, Tuple, Set, DefaultDict
import hdbscan
from sklearn.preprocessing import normalize
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA  # 添加PCA导入
import warnings
import os
from tqdm import tqdm
# 忽略特定的警告
warnings.filterwarnings('ignore', category=FutureWarning)

class EnhancedRelationNormalizer:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', 
                 min_cluster_size: int = 2, 
                 min_samples: int = 1,
                 similarity_threshold: float = 0.70,
                 cluster_selection_epsilon: float = 0.3):  # 添加新的参数
        np.random.seed(2025)  # 添加这行
        print(f"正在加载模型 {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon

    def _normalize_relation_name(self, relation: str) -> str:
        """统一关系名称的格式"""
        return relation.lower().strip().replace(' ', '_')

    def _can_be_in_same_group(self, rel1: Tuple[str, str, str], rel2: Tuple[str, str, str]) -> bool:
        """判断是否属于同个实体类型组"""
        return (rel1[0], rel1[2]) == (rel2[0], rel2[2])

    def _create_relation_text(self, entity1: str, relation: str, entity2: str) -> str:


        return f"{entity1} {relation} {entity2}"

    def _encode_texts(self, texts: List[str]) -> np.ndarray:

        try:
            # 批量处理以提高效率
            batch_size = 32  
            all_embeddings = []
            
            for i in tqdm(range(0, len(texts), batch_size), desc="生成文本嵌入"):
                batch = texts[i:i + batch_size]
                embeddings = self.model.encode(batch, 
                                            normalize_embeddings=True,  # 确保输出的嵌入向量是标准化的
                                            show_progress_bar=False,    # 关闭进度条避免过多输出
                                            convert_to_numpy=True)      # 直接输出numpy数组
                all_embeddings.append(embeddings)
            
            return np.vstack(all_embeddings)
        except Exception as e:
            print(f"警告：生成文本嵌入时出错: {str(e)}")
            raise

    def _cluster_relations(self, encodings: np.ndarray, relation_details: List[Dict]) -> Dict[int, List[int]]:
        """改进的聚类方法，确保ID连续"""
        # 确保输入数据是有限的
        if not np.all(np.isfinite(encodings)):
            encodings = np.nan_to_num(encodings, nan=0.0, posinf=1.0, neginf=-1.0)
            
        normalized_encodings = normalize(encodings, copy=True)
        
        # 第一阶段：基于实体类型的粗分组
        entity_groups: DefaultDict[int, Set[int]] = defaultdict(set)
        group_assignment = {}
        current_group = 0
        
        # 构建初始分组
        print("正在进行实体类型粗分组...")
        for idx, detail in enumerate(tqdm(relation_details, desc="实体类型分组")):
            parts = detail['parts']
            matched = False
            # 尝试匹配现有分组
            for group_id in entity_groups:
                for member_idx in entity_groups[group_id]:
                    if self._can_be_in_same_group(parts, relation_details[member_idx]['parts']):
                        # 计算相似度
                        similarity = np.dot(normalized_encodings[idx], normalized_encodings[member_idx])
                        if similarity >= self.similarity_threshold:
                            entity_groups[group_id].add(idx)
                            group_assignment[idx] = group_id
                            matched = True
                            break
                if matched:
                    break
            # 新建分组
            if not matched:
                entity_groups[current_group].add(idx)
                group_assignment[idx] = current_group
                current_group += 1

        # 第二阶段：精细聚类
        clusters: Dict[int, List[int]] = {}
        cluster_id_counter = 0
        
        print("正在进行精细聚类...")
        for group_id, members in tqdm(entity_groups.items(), desc="精细聚类"):
            member_list = list(members)
            if len(member_list) < 2:
                # 单元素直接作为独立簇
                clusters[cluster_id_counter] = member_list
                cluster_id_counter += 1
                continue
                
            # 执行HDBSCAN聚类
            group_encodings = normalized_encodings[member_list]
            
            # 确保组编码数据是有限的
            if not np.all(np.isfinite(group_encodings)):
                group_encodings = np.nan_to_num(group_encodings, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 计算实际的最小簇大小
            actual_min_cluster_size = min(self.min_cluster_size, len(member_list))
            if actual_min_cluster_size < 2:
                actual_min_cluster_size = 2
            
            actual_min_samples = min(self.min_samples, actual_min_cluster_size - 1)
            if actual_min_samples < 1:
                actual_min_samples = 1
                
            try:
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=actual_min_cluster_size,
                    min_samples=actual_min_samples,
                    metric='euclidean',
                    core_dist_n_jobs=-1,
                    cluster_selection_epsilon=self.cluster_selection_epsilon,
                    cluster_selection_method='eom'
                )
                
                labels = clusterer.fit_predict(group_encodings)
                
                # 检查是否所有点都被标记为噪声
                if np.all(labels == -1):
                    # 如果所有点都是噪声，将整个组作为一个簇
                    clusters[cluster_id_counter] = member_list
                    cluster_id_counter += 1
                    continue
                
            except Exception as e:
                print(f"警告：聚类过程出现错误，将整个组作为一个簇处理: {str(e)}")
                clusters[cluster_id_counter] = member_list
                cluster_id_counter += 1
                continue
            
            # 处理聚类结果
            label_map = defaultdict(list)
            for idx, label in zip(member_list, labels):
                label_map[label].append(idx)
            
            # 分配簇ID
            for label, indices in label_map.items():
                if label == -1:
                    # 噪声点作为独立簇
                    for noise_idx in indices:
                        clusters[cluster_id_counter] = [noise_idx]
                        cluster_id_counter += 1
                else:
                    # 有效簇
                    clusters[cluster_id_counter] = indices
                    cluster_id_counter += 1

        return clusters

    def _plot_clusters(self, encodings: np.ndarray, clusters: Dict[int, List[int]], relation_details: List[Dict], output_file: str):
        """Plot top 10 clusters visualization"""
        try:
            # 确保输入数据是有限的
            if not np.all(np.isfinite(encodings)):
                encodings = np.nan_to_num(encodings, nan=0.0, posinf=1.0, neginf=-1.0)
                
            # 获取前20个簇的索引和对应的标签
            all_indices = []
            cluster_labels = []
            cluster_sizes = {}
            
            # 计算每个簇的大小
            for cluster_id, indices in clusters.items():
                cluster_sizes[cluster_id] = len(indices)
            
            # 按簇的大小排序，选择前20个最大的簇
            top_20_clusters = dict(sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)[:20])
            
            # 获取前20个簇的数据
            for cluster_id in top_20_clusters.keys():
                indices = clusters[cluster_id]
                all_indices.extend(indices)
                cluster_labels.extend([cluster_id] * len(indices))
            
            if not all_indices:
                raise ValueError("No valid cluster data found")
            
            # 使用PCA进行降维
            print("Performing dimensionality reduction...")
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(encodings[all_indices])
            
            # 创建图形
            print("Creating scatter plot...")
            plt.figure(figsize=(12, 8))
            
            # 使用更分散的颜色方案
            import colorsys
            
            def generate_distinct_colors(n):
                colors = []
                for i in range(n):
                    hue = i / n
                    saturation = 0.7 + (i % 3) * 0.1  # 在0.7-0.9之间变化
                    value = 0.8 + (i % 2) * 0.1      # 在0.8-0.9之间变化
                    colors.append(colorsys.hsv_to_rgb(hue, saturation, value))
                return colors
            
            colors = generate_distinct_colors(len(top_20_clusters))

            # 设置白色背景和网格
            plt.style.use('default')
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.gca().set_facecolor('white')
            plt.gcf().set_facecolor('white')
            
            # 绘制置信椭圆
            from matplotlib.patches import Ellipse
            import scipy.stats as stats
            
            for i, (cluster_id, size) in enumerate(top_20_clusters.items()):
                mask = np.array(cluster_labels) == cluster_id
                cluster_points = embeddings_2d[mask]
                
                if len(cluster_points) > 2:  # 至少需要3个点才能计算椭圆
                    # 计算均值和协方差
                    mean = np.mean(cluster_points, axis=0)
                    cov = np.cov(cluster_points.T)
                    
                    # 计算特征值和特征向量
                    eigenvals, eigenvecs = np.linalg.eigh(cov)
                    
                    # 计算椭圆角度
                    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
                    
                    # 创建95%置信椭圆
                    chi2_val = stats.chi2.ppf(0.95, 2)
                    scale = np.sqrt(chi2_val)
                    width, height = 2 * scale * np.sqrt(eigenvals)
                    
                    ellip = Ellipse(xy=mean, width=width, height=height, angle=angle,
                                  facecolor=colors[i], alpha=0.2, edgecolor=None)
                    plt.gca().add_patch(ellip)
            
            # 然后绘制散点
            for i, (cluster_id, size) in enumerate(top_20_clusters.items()):
                mask = np.array(cluster_labels) == cluster_id
                
                # 获取该簇的一个示例关系作为标签
                sample_idx = clusters[cluster_id][0]
                relation = relation_details[sample_idx]['parts'][1]
                label = f'Cluster {i+1}: {relation}(size={size})'
                
                plt.scatter(
                    embeddings_2d[mask, 0],
                    embeddings_2d[mask, 1],
                    color=colors[i],  
                    label=label,
                    s=80,
                    alpha=0.7,
                    edgecolors='white',
                    linewidth=0.5
                )
            
            # 调整轴的范围以减少空白
            x_min, x_max = embeddings_2d[:, 0].min(), embeddings_2d[:, 0].max()
            y_min, y_max = embeddings_2d[:, 1].min(), embeddings_2d[:, 1].max()
            
            # 计算边距（数据范围的8%）
            x_margin = (x_max - x_min) * 0.08
            y_margin = (y_max - y_min) * 0.08
            
            # 设置显示范围，稍微扩大以显示完整的簇
            plt.xlim(x_min - x_margin, x_max + x_margin)
            plt.ylim(y_min - y_margin, y_max + y_margin)
            
            # 设置标题和轴标签
            plt.title('PCA Scores Plot', fontsize=15, pad=20)
            plt.xlabel(f'PC1', fontsize=13)
            plt.ylabel(f'PC2', fontsize=13)
            
            # 设置图例
            plt.legend(
                title='Clusters',
                title_fontsize=13,
                fontsize=11,
                bbox_to_anchor=(1.05, 1),
                loc='upper left',
                borderaxespad=0,
                frameon=True,
                facecolor='white',
                edgecolor='gray',
                framealpha=0.8
            )
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图片
            print(f"Saving visualization to: {os.path.abspath(output_file)}")
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            plt.savefig(
                output_file,
                dpi=300,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none',
                pad_inches=0.4
            )
            plt.close()
            print(f"Visualization saved successfully to: {os.path.abspath(output_file)}")
            
        except Exception as e:
            print(f"Warning: Error generating visualization: {str(e)}")
            import traceback
            print(f"Error details: {traceback.format_exc()}")
            print("Continuing with other steps...")

    def normalize_relations(self, input_file: str, output_file: str):
        """完整的归一化流程"""
        try:
            # 读取输入文件
            relations = []
            original_data = []  # 存储原始数据，包括id
            print("正在读取输入文件...")
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="读取数据"):
                    data = json.loads(line.strip())
                    original_data.append(data)  # 保存原始数据
                    if 'triples' in data:
                        for triple in data['triples']:
                            relation_type = f"{triple['subject_type']} -> {triple['relation_type']} -> {triple['object_type']}"
                            relations.append({
                                'relation_type': relation_type,
                                'relation': triple['relation_type'],
                                'count': 1,
                                'subject_name': triple['subject_name'],
                                'object_name': triple['object_name'],
                                'subject_type': triple['subject_type'],
                                'object_type': triple['object_type']
                            })

            if not relations:
                raise ValueError("没有找到任何关系数据")

            normalized_relations = {}
            processed_indices = set()

            # 准备关系数据
            relation_texts = []
            relation_details = []
            
            print("正在处理关系数据...")
            for rel in tqdm(relations, desc="处理关系"):
                try:
                    if not isinstance(rel, dict) or 'relation_type' not in rel:
                        print(f"警告：跳过无效的关系数据: {rel}")
                        continue
                        
                    if not rel['relation_type'] or not isinstance(rel['relation_type'], str):
                        print(f"警告：跳过无效的关系类型: {rel['relation_type']}")
                        continue
                        
                    count = rel.get('count', 1)
                    if not isinstance(count, (int, float)) or count <= 0:
                        count = 1
                        print(f"警告：关系计数无效，使用默认值1: {rel['relation_type']}")
                    
                    parts = rel['relation_type'].split(' -> ')
                    if len(parts) != 3:
                        print(f"警告：跳过不完整的关系: {rel['relation_type']}")
                        continue
                        
                    entity1, relation, entity2 = parts
                    text = self._create_relation_text(entity1, relation, entity2)
                    relation_texts.append(text)
                    relation_details.append({
                        'parts': (entity1, relation, entity2),
                        'count': count,
                        'original': rel['relation'],
                        'text': text,
                        'subject_name': rel.get('subject_name', ''),
                        'object_name': rel.get('object_name', '')
                    })
                except Exception as e:
                    print(f"警告：处理关系时出错，已跳过: {str(e)}")
                    continue

            if not relation_texts:
                raise ValueError("没有有效的关系数据可以处理")

            print(f"成功处理 {len(relation_texts)} 个关系")
            
            # 生成嵌入向量
            print("正在生成文本嵌入...")
            try:
                encodings = self._encode_texts(relation_texts)
                if encodings is None or encodings.shape[0] == 0:
                    raise ValueError("生成嵌入向量失败")
            except Exception as e:
                raise ValueError(f"生成嵌入向量时出错: {str(e)}")

            # 执行聚类
            print("正在执行聚类...")
            try:
                clusters = self._cluster_relations(encodings, relation_details)
                if not clusters:
                    raise ValueError("聚类结果为空")
            except Exception as e:
                raise ValueError(f"聚类过程出错: {str(e)}")

            # 绘制聚类可视化图
            print("正在生成可视化图...")
            vis_output_file = output_file.replace('.json', '_clusters.png')
            try:
                self._plot_clusters(encodings, clusters, relation_details, vis_output_file)
                print(f"聚类可视化图已保存至：{vis_output_file}")
            except Exception as e:
                print(f"警告：生成可视化图时出错: {str(e)}")

            # 处理聚类结果
            print("正在处理聚类结果...")
            for cluster_id, indices in tqdm(clusters.items(), desc="处理聚类结果"):
                if not indices:
                    continue
                    
                try:
                    # 处理单元素簇
                    if len(indices) == 1:
                        idx = indices[0]
                        detail = relation_details[idx]
                        normalized_relations[detail['text']] = {
                            'cluster_id': cluster_id,
                            'original_relations': [detail['original']],
                            'representative_relation': detail['parts'][1],  # 只保留关系部分
                            'total_count': detail['count'],
                            'representative_count': detail['count'],
                            'avg_similarity_score': 1.0,
                            'cluster_size': 1,
                            'entity_types': {
                                'head': detail['parts'][0],
                                'tail': detail['parts'][2]
                            },
                            'examples': [{
                                'subject_name': detail['subject_name'],
                                'object_name': detail['object_name']
                            }]
                        }
                        continue

                    # 处理多元素簇
                    cluster_details = [relation_details[idx] for idx in indices]
                    embeddings = encodings[indices]
                    
                    # 计算代表关系
                    freq_map = defaultdict(int)
                    for detail in cluster_details:
                        freq_map[detail['text']] += detail['count']
                    
                    if not freq_map:
                        continue
                        
                    canonical_text = max(freq_map, key=lambda k: freq_map[k])
                    
                    # 获取代表关系的原始部分（不包含实体类型）
                    canonical_detail = next(d for d in cluster_details if d['text'] == canonical_text)
                    canonical_relation = canonical_detail['parts'][1]  # 只保留关系部分
                    
                    # 计算平均相似度
                    try:
                        canonical_emb = self._encode_texts([canonical_text])[0]
                        if canonical_emb is None:
                            raise ValueError("代表关系编码失败")
                        similarities = np.dot(embeddings, canonical_emb)
                        avg_similarity = float(np.mean(similarities))
                    except Exception as e:
                        print(f"警告：计算相似度时出错，使用默认值: {str(e)}")
                        avg_similarity = 0.5

                    # 收集示例和唯一的关系类型
                    unique_relations = set()
                    for detail in cluster_details:
                        unique_relations.add(detail['original'])

                    normalized_relations[canonical_text] = {
                        'cluster_id': cluster_id,
                        'original_relations': sorted(list(unique_relations)),
                        'representative_relation': canonical_relation,  # 使用只包含关系部分的文本
                        'total_count': sum(d['count'] for d in cluster_details),
                        'representative_count': freq_map[canonical_text],
                        'avg_similarity_score': avg_similarity,
                        'cluster_size': len(indices),
                        'entity_types': {
                            'head': cluster_details[0]['parts'][0],
                            'tail': cluster_details[0]['parts'][2]
                        },

                    }
                except Exception as e:
                    print(f"警告：处理簇 {cluster_id} 时出错: {str(e)}")
                    continue

            # 生成统计信息
            output_data = {
                'statistics': {
                    'original_relation_types': len(relations),
                    'normalized_relation_types': len(normalized_relations),
                    'reduction_percentage': round((1 - len(normalized_relations)/len(relations))*100, 2),
                },
                'normalized_relations': dict(sorted(
                    normalized_relations.items(),
                    key=lambda x: x[1]['total_count'],
                    reverse=True
                ))
            }

            # 保存结果
            print("正在保存结果...")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            print(f"处理完成！结果已保存至：{output_file}")
            print(f"原始关系数量: {len(relations)}")
            print(f"归一化后关系数量: {len(normalized_relations)}")
            print(f"归一化率: {output_data['statistics']['reduction_percentage']}%")

            # 构建关系映射词典
            relation_mapping = {}
            for rel_text, rel_info in normalized_relations.items():
                representative = rel_info['representative_relation']
                for original in rel_info['original_relations']:
                    relation_mapping[original] = representative

            # 生成新的jsonl文件
            jsonl_output_file = output_file.replace('.json', '_mapped.jsonl')
            print(f"正在生成映射后的jsonl文件：{jsonl_output_file}")
            
            with open(jsonl_output_file, 'w', encoding='utf-8') as f:
                for data in original_data:
                    new_data = {
                        'id': data.get('id', ''),
                        'triples': []
                    }
                    
                    if 'triples' in data:
                        for triple in data['triples']:
                            original_relation = triple['relation_type']
                            mapped_relation = relation_mapping.get(original_relation, original_relation)
                            
                            new_triple = {
                                'subject_type': triple['subject_type'],
                                'subject_name': triple['subject_name'],
                                'relation_type': mapped_relation,
                                'object_type': triple['object_type'],
                                'object_name': triple['object_name']
                            }
                            new_data['triples'].append(new_triple)
                    
                    f.write(json.dumps(new_data, ensure_ascii=False) + '\n')

            print(f"映射后的jsonl文件已保存至：{jsonl_output_file}")

        except Exception as e:
            print(f"处理过程中发生错误：{str(e)}")
            raise

def main():
    try:
        input_file = 'data/triples_qwen.json'
        output_file = 'data/qwen_cluster.json'
        
        print("启动关系归一化流程...")
        normalizer = EnhancedRelationNormalizer(
            min_cluster_size=2,      # 保持较小的最小簇大小
            min_samples=1,           # 保持较小的最小样本数
            similarity_threshold=0.5,  # 降低相似度阈值，使更多关系被归为一类
            cluster_selection_epsilon=0.7  # 增加epsilon值使聚类更宽松
        )
        normalizer.normalize_relations(input_file, output_file)
        
        print(f"处理完成！结果已保存至：{output_file}")
        
    except Exception as e:
        print(f"处理过程中发生错误：{str(e)}")

if __name__ == "__main__":
    main()