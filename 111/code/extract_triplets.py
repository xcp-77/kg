import os
import json
import requests
from typing import List, Dict
from langchain_community.llms import Tongyi
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def load_abstracts(json_file: str) -> List[Dict]:
    """从JSON文件加载abstract_filter字段和ID"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            abstracts = []
            if 'articles' in data and isinstance(data['articles'], list):
                for article in data['articles']:
                    abstract = article.get('abstract_filter', '')
                    item_id = str(article.get('id', ''))
                    
                    if abstract and item_id:
                        abstracts.append({
                            'id': item_id,
                            'abstract': abstract
                        })
            
            print(f"成功加载 {len(abstracts)} 条有效摘要" if abstracts else "警告：未找到任何有效的摘要数据")
            return abstracts
            
    except Exception as e:
        print(f"加载JSON文件失败: {str(e)}")
        return []

def build_extraction_chain():
    """构建关系抽取链"""
    system_template = SystemMessagePromptTemplate.from_template("""
    ## Role: Biochemical Entity Relationship Graph Construction System
    ### Target Entity Categories: 
    - Fruit (e.g., citrus fruits, orange, lemon)
    - Compound (e.g., naringin, citric acid, flavonoids)
    - Flavor (e.g., sweet, bitter, sour)
    - Enzyme (e.g., pectinase, amylase)
    - Reaction (e.g., oxidation, hydrolysis)
    - Metabolism (e.g., glycolysis, fermentation)
    
    ### Processing Requirements:
    1. Only process entity types within the whitelist
    2. Only capture relationships explicitly described in the text
    3. Output format:
    [
    {{
        "Entity Category 1": "Compound",
        "Entity 1": "Naringin",
        "Relationship": "Found in",
        "Entity Category 2": "Fruit",
        "Entity 2": "Citrus fruits"
    }}
    ]
    ### Example:
    Input: Naringin found in citrus fruits
    Output: [{{"Entity Category 1":"Compound","Entity 1":"Naringin","Relationship":"Found in","Entity Category 2":"Fruit","Entity 2":"Citrus fruits"}}]
    """)

    human_template = HumanMessagePromptTemplate.from_template("{text}")

    prompt_template = ChatPromptTemplate.from_messages([
        system_template,
        human_template
    ])
    
    # 添加超时设置
    model = Tongyi(
        model="qwen-plus-2025-01-25",
        api_key=os.getenv("TONGYI_API_KEY"),
        temperature=0.3,
        top_p=0.8,
        request_timeout=30  # 30秒超时
    )
    
    def parse_output(text: str) -> List[Dict]:
        """解析模型输出"""
        try:
            # 提取有效的JSON部分
            start = text.find('[')
            end = text.rfind(']') + 1
            json_str = text[start:end]
            
            data = json.loads(json_str)
            valid_relations = []
            for item in data:
                if all(key in item for key in ["Entity Category 1", "Entity 1", "Relationship", "Entity Category 2", "Entity 2"]):
                    valid_relations.append({
                        "source_type": item["Entity Category 1"],
                        "source": item["Entity 1"],
                        "relation": item["Relationship"],
                        "target_type": item["Entity Category 2"],
                        "target": item["Entity 2"]
                    })
            return valid_relations
        except Exception as e:
            print(f"解析错误: {str(e)}")
            return []

    return (
        RunnablePassthrough.assign(text=lambda x: x["text"])
        | prompt_template
        | model
        | StrOutputParser()
        | RunnableLambda(parse_output)
    )

def process_articles(start_id: str = None):
    """处理文章主流程（新增数据追加功能）"""
    chain = build_extraction_chain()
    output_file = "relations.json"
    
    # 加载全部数据
    all_articles = load_abstracts('pubmed_filter.json')
    if not all_articles:
        print("无法继续处理，请检查数据文件")
        return

    # 加载已处理数据
    processed_ids = set()
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        processed_ids.add(data['id'])
                    except json.JSONDecodeError:
                        continue
        except:
            pass

    # 获取起始位置
    if start_id:
        start_index = next((i for i, a in enumerate(all_articles) if a['id'] == start_id), None)
        if start_index is None:
            print(f"\n警告：未找到ID {start_id}")
            print(f"前5个可用ID示例: {', '.join(a['id'] for a in all_articles[:5])}")
            print("将从头开始处理...")
            start_index = 0
        else:
            print(f"从ID {start_id} 开始处理...")
    else:
        start_index = 0

    # 处理文章
    try:
        for idx, article in enumerate(all_articles[start_index:], start=1):
            article_id = article['id']
            
            # 跳过已处理文章
            if article_id in processed_ids:
                print(f"\n跳过已处理文章 ID: {article_id}")
                continue
                
            print(f"\n[{idx}/{len(all_articles)}] 处理文章 ID: {article_id}")
            print("摘要片段:", article['abstract'][:100].replace('\n', ' ') + "...")
            
            try:
                # 添加网络异常处理
                relations = chain.invoke({"text": article['abstract']})
                
                # 追加写入文件
                with open(output_file, 'a', encoding='utf-8') as f:
                    json.dump({
                        "id": article_id,
                        "relations": relations
                    }, f, ensure_ascii=False)
                    f.write('\n')  # 写入换行符分隔
                
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"网络连接异常: {str(e)}") from e
            except TimeoutError as e:
                raise RuntimeError("大模型响应超时") from e
            except Exception as e:
                raise RuntimeError(f"处理异常: {str(e)}") from e

        print(f"\n处理完成！结果已追加保存至 {output_file}")

    except Exception as e:
        # 获取最后处理的ID
        last_id = article_id if 'article_id' in locals() else "无"
        raise RuntimeError(f"{str(e)} (最后处理ID: {last_id})")

def main():
    """交互式入口"""
    print("="*40)
    print("生物医学关系抽取系统")
    print("="*40)
    
    # 获取用户输入
    start_id = input("\n请输入起始文章ID（直接回车将从头开始）: ").strip()
    
    try:
        print("\n启动处理流程...")
        process_articles(start_id)
    except RuntimeError as e:
        print(f"\n⚠️ 处理中断: {str(e)}")
    except KeyboardInterrupt:
        print(f"\n⏹️ 用户手动中断")
    except Exception as e:
        print(f"\n❌ 未处理异常: {str(e)}")

if __name__ == "__main__":
    main()