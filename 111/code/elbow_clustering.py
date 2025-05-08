import numpy as np
import json
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from kneed import KneeLocator
# 添加中文字体支持
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']  # 使用多个备选字体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['font.family'] = 'sans-serif'  # 设置字体族

def load_data(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 确保数据包含所需的字段
    if not isinstance(data, dict) or 'normalized_relations' not in data:
        raise ValueError("数据格式不正确，需要包含 'normalized_relations' 字段")
    
    # 将字典转换为列表格式，每个项目包含关系名称和其属性
    relations_list = []
    for relation_name, relation_data in data['normalized_relations'].items():
        relation_data['name'] = relation_name
        relations_list.append(relation_data)
    
    return relations_list

def pareto_analysis(data, pareto_threshold=0.8):
    """
    帕累托分析
    :param data: list或np.array，各簇的total_count值
    :param pareto_threshold: 帕累托阈值（默认80%）
    :return: (high_freq_idx, sorted_indices, sorted_data) 高频簇的索引及排序信息
    """
    data = np.array(data)
    # 按total_count降序排列
    sorted_indices = np.argsort(data)[::-1]
    sorted_data = data[sorted_indices]
    
    # 计算累积和
    cumsum = np.cumsum(sorted_data)
    total = cumsum[-1]
    
    # 筛选高频簇（达到pareto阈值）
    k = np.argmax(cumsum >= pareto_threshold * total) + 1
    high_freq_idx = sorted_indices[:k]
    
    return high_freq_idx, sorted_indices, sorted_data

def find_pareto_threshold(data):
    # 获取所有的total_count值
    counts = [item.get('total_count', 0) for item in data]
    
    # 运行帕累托分析
    high_idx, sorted_indices, sorted_data = pareto_analysis(counts)
    
    # 可视化
    plt.figure(figsize=(14, 7))  # 增加图表尺寸以容纳外置图例
    
    # 设置非线性x轴变换的函数
    def nonlinear_transform(x, breakpoint=1000, left_weight=0.5):
        max_x = len(counts)
        if x <= breakpoint:
            return x * left_weight / breakpoint
        else:
            return left_weight + (x - breakpoint) * (1 - left_weight) / (max_x - breakpoint)
    
    # 创建转换后的x坐标
    x_transformed = [nonlinear_transform(i) for i in range(len(counts))]
    
    # 创建主坐标轴和次坐标轴
    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax2 = ax1.twinx()
    
    # 设置颜色
    bar_color = '#ADD8E6'  # 淡蓝色
    line_color = '#FF8C00'  # 深橙色
    threshold_color = '#FFD700'  # 金色
    grid_color = '#E0E0E0'  # 浅灰色
    
    # 绘制柱状图（频次）
    bar_width = 0.008 if len(counts) > 1000 else 0.5 / len(counts)
    bars = ax1.bar(x_transformed, sorted_data, color=bar_color, alpha=0.7, width=bar_width)
    
    # 绘制累积分布曲线
    cumulative = np.cumsum(sorted_data) / np.sum(sorted_data) * 100
    line = ax2.plot(x_transformed, cumulative, color=line_color, linewidth=2.5, label='累计百分比')
    
    # 标注分层点
    k = len(high_idx)
    threshold_x = nonlinear_transform(k)
    threshold_y = cumulative[k-1]
    
    # 绘制阈值线
    ax1.axvline(x=threshold_x, color=threshold_color, linestyle='--', linewidth=1.5, alpha=0.8)
    
    # 在拐点处添加标记
    ax2.plot(threshold_x, threshold_y, 'o', color=line_color, markersize=8, 
            label=f'拐点 ({k}, {threshold_y:.1f}%)')
    
    # 设置标签和标题
    ax1.set_xlabel('关系簇', fontsize=11)
    ax1.set_ylabel('频次', fontsize=11, rotation=0, ha='right', va='center')
    ax2.set_ylabel('累计百分比 (%)', fontsize=11, rotation=0, ha='left', va='center')
    plt.title('关系簇帕累托分析', pad=20, fontsize=13)
    
    # 设置网格 - 只保留水平主要网格线
    ax1.grid(True, axis='y', color=grid_color, alpha=0.5, linestyle='-', which='major')
    ax1.grid(False, axis='x')
    
    # 调整坐标轴范围
    max_count = max(sorted_data)
    ax1.set_ylim([0, 2000])  # 频次轴范围略高于最大值
    ax2.set_ylim([0, 100])  # 百分比轴范围
    
    # 设置x轴刻度
    breakpoint = 1000
    original_ticks_before_1000 = list(range(0, breakpoint + 1, 100))
    original_ticks_after_1000 = list(range(breakpoint, len(counts) + 1, 1000))
    if original_ticks_after_1000[0] == breakpoint:
        original_ticks_after_1000 = original_ticks_after_1000[1:]
    all_original_ticks = original_ticks_before_1000 + original_ticks_after_1000
    
    transformed_tick_positions = [nonlinear_transform(pos) for pos in all_original_ticks]
    ax1.set_xticks(transformed_tick_positions)
    ax1.set_xticklabels([str(pos) for pos in all_original_ticks], rotation=45)
    
    # 添加阈值标注
    plt.text(threshold_x + 0.02, threshold_y, 
             f'← 关键阈值：前{k}项\n   占{threshold_y:.1f}%累计频次', 
             color='black', ha='left', va='center',
             bbox=dict(facecolor='white', edgecolor=threshold_color, alpha=0.8))
    
    # 合并图例并放置在图表内部右上角
    bars_legend = plt.Rectangle((0,0), 1, 1, fc=bar_color, alpha=0.7)
    threshold_line = plt.Line2D([0], [0], color=threshold_color, linestyle='--', linewidth=1.5)
    percentage_line = plt.Line2D([0], [0], color=line_color, linewidth=2.5)
    
    legend_elements = [
        (bars_legend, '频次'),
        (percentage_line, '累计百分比'),
        (threshold_line, f'帕累托阈值\n(前{k}项占{threshold_y:.1f}%)'),
    ]
    
    # 创建图例，放在右上角，添加半透明背景
    ax1.legend(*zip(*legend_elements), 
              loc='upper right',
              bbox_to_anchor=(0.98, 0.98),
              framealpha=0.8,
              edgecolor='none',
              fontsize=10)
    
    # 保存图片
    plt.savefig('pareto_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 返回阈值
    threshold = sorted_data[k-1] if k > 0 else 0
    
    return threshold, k

def filter_by_threshold(data, threshold):
    # 过滤数据
    filtered_data = [
        item for item in data
        if item.get('total_count', 0) >= threshold
    ]
    return filtered_data

def save_results(filtered_data, output_file, original_data, threshold, k):
    # 准备统计信息
    stats = {
        "统计信息": {
            "原始数据簇数量": int(len(original_data)),
            "阈值": float(threshold),
            "保留簇数量": int(k),
            "保留关系数量": int(len(filtered_data)),
            "过滤关系数量": int(len(original_data) - len(filtered_data))
        },
        "normalized_relations": {item['name']: {k: v for k, v in item.items() if k != 'name'} 
                               for item in filtered_data}
    }
    
    # 转换所有数值为Python原生类型
    def convert_to_native_types(obj):
        if isinstance(obj, dict):
            return {key: convert_to_native_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    stats = convert_to_native_types(stats)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

def main():
    input_file = 'qwen_normalized_relations.json'  # 输入文件路径
    output_file = 'filtered_output.json'  # 输出文件路径
    
    # 加载数据
    data = load_data(input_file)
    
    # 使用帕累托分析找到阈值
    threshold, k = find_pareto_threshold(data)
    print(f"帕累托阈值: {threshold} (保留前 {k} 个簇)")
    
    # 根据阈值过滤数据
    filtered_data = filter_by_threshold(data, threshold)
    
    # 保存结果
    save_results(filtered_data, output_file, data, threshold, k)
    print(f"已将过滤后的数据保存到: {output_file}")

if __name__ == "__main__":
    main() 