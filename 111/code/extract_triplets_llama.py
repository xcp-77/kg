import os
import json
import requests
from typing import List, Dict, Tuple
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv
from http import HTTPStatus
from dashscope import Generation

# 加载环境变量
load_dotenv()

# 定义文件路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
ABSTRACT_FILE = os.path.join(DATA_DIR, 'pubmed_abstract.json')
OUTPUT_FILE = os.path.join(DATA_DIR, 'triples_llama.jsonl')

class RelationExtractor:
    """关系抽取器类，封装所有相关功能"""
    
    def __init__(self):
        """初始化关系抽取器"""
        self.chain = self._build_extraction_chain()
        
    def _build_extraction_chain(self):
        """构建关系抽取链"""
        # 系统提示词模板
        system_template = SystemMessagePromptTemplate.from_template("""
        ## Role: Citrus Flavor Metabolite Knowledge Graph Construction System
        ### Target Entity Categories: 
        - Citrus (e.g., orange, lemon)
        - Compound (e.g., naringin, citric acid, flavonoids)
        - Flavor (e.g., sweet, bitter, sour)
        - Enzyme (e.g., pectinase, amylase)
        - Reaction (e.g., oxidation, hydrolysis)
        - Metabolism (e.g., glycolysis, fermentation)
        
        ### Processing Requirements: 
        1. Only process entity types within the whitelist
        2. Only extract relationships explicitly described in the text
        3. Output format:
        [
        {{"subject_type": "", "subject_name": "", "relation_type": "", "object_type": "", "object_name": ""}}
        ]
        ### Example:
        Input: Navelina oranges contains hesperidin (0.5-1.2% of fresh weight), a bioactive compound with proven antioxidant (ORAC 12,000 μmol TE/g) and vascular-protective effects. Its derivatives show therapeutic potential in inflammation and viral inhibition.
        Output: [{{"subject_type": "Citrus", "subject_name": "Navelina oranges", "relation_type": "hasCompound", "object_type": "Compound", "object_name": "hesperidin"}}]
        """)

        # 用户输入模板
        human_template = HumanMessagePromptTemplate.from_template("{text}")

        # 组合提示词模板
        prompt_template = ChatPromptTemplate.from_messages([
            system_template,
            human_template
        ])
        
        def _call_llama(text: str) -> str:
            """调用Llama模型进行关系抽取"""
            try:
                # 生成对话消息
                messages = prompt_template.format_messages(text=text)
                
                # 转换消息格式
                formatted_messages = [
                    {
                        "role": "system" if msg.type == "system" else "user",
                        "content": msg.content
                    }
                    for msg in messages
                ]
                
                # 调用Generation API
                response = Generation.call(
                    model='llama3.3-70b-instruct',
                    messages=formatted_messages,
                    api_key="sk-0a98608e71154f3daf615eb0d36260c3",
                    temperature=0.3,
                    top_p=0.8
                )
                
                # 处理响应
                if not response:
                    raise RuntimeError('API无响应')
                    
                print(f"API响应状态码: {response.status_code}")
                print(f"API响应内容: {response}")
                
                if response.status_code == HTTPStatus.OK:
                    if not hasattr(response, 'output'):
                        raise RuntimeError('API响应缺少output字段')
                        
                    if not hasattr(response.output, 'text'):
                        raise RuntimeError('API响应output缺少text字段')
                        
                    output = response.output.text
                    if not output:
                        raise RuntimeError('API响应text为空')
                        
                    print("调试信息 - 模型输出:", output)
                    return output
                else:
                    error_msg = f'API调用失败: {response.code if hasattr(response, "code") else "未知错误码"}'
                    if hasattr(response, 'message'):
                        error_msg += f' - {response.message}'
                    raise RuntimeError(error_msg)
            except Exception as e:
                print(f"模型调用错误: {str(e)}")
                raise
        
        def _parse_output(text: str) -> List[Dict]:
            """解析模型输出为结构化数据"""
            try:
                # 提取JSON部分
                start = text.find('[')
                end = text.rfind(']') + 1
                if start == -1 or end == 0:
                    return []
                
                json_str = text[start:end]
                data = json.loads(json_str)
                
                # 标准化输出格式
                valid_relations = []
                for item in data:
                    relation = {
                        "subject_type": item.get("subject_type", ""),
                        "subject_name": item.get("subject_name", ""),
                        "relation_type": item.get("relation_type", ""),
                        "object_type": item.get("object_type", ""),
                        "object_name": item.get("object_name", "")
                    }
                    if all(relation.values()):
                        valid_relations.append(relation)
                return valid_relations
            except Exception as e:
                print(f"解析错误: {str(e)}")
                return []

        # 构建处理链
        return (
            RunnablePassthrough.assign(text=lambda x: x["text"])
            | prompt_template
            | RunnableLambda(lambda x: _call_llama(x.messages[-1].content))  # 获取最后一条消息内容
            | RunnableLambda(_parse_output)
        )

    def extract_relations(self, text: str) -> List[Dict]:
        """从文本中抽取关系"""
        return self.chain.invoke({"text": text})

def load_abstracts(json_file: str = ABSTRACT_FILE) -> List[Dict]:
    """从JSON文件加载摘要数据
    
    Args:
        json_file: JSON文件路径，默认为配置的路径
        
    Returns:
        List[Dict]: 包含ID和摘要的列表
    """
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
        print(f"尝试加载的文件路径: {json_file}")
        return []

def process_articles(start_id: str = None):
    """处理文章主流程
    
    Args:
        start_id: 起始文章ID，如果为None则从头开始处理
    """
    # 初始化关系抽取器
    extractor = RelationExtractor()
    
    # 加载全部数据
    all_articles = load_abstracts()
    if not all_articles:
        print("无法继续处理，请检查数据文件")
        return

    # 加载已处理数据
    processed_ids = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
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
                # 抽取关系
                relations = extractor.extract_relations(article['abstract'])
                
                # 保存结果
                with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                    json.dump({
                        "id": article_id,
                        "relations": relations
                    }, f, ensure_ascii=False)
                    f.write('\n')
                
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"网络连接异常: {str(e)}") from e
            except TimeoutError as e:
                raise RuntimeError("大模型响应超时") from e
            except Exception as e:
                raise RuntimeError(f"处理异常: {str(e)}") from e

        print(f"\n处理完成！结果已追加保存至 {OUTPUT_FILE}")

    except Exception as e:
        # 获取最后处理的ID
        last_id = article_id if 'article_id' in locals() else "无"
        raise RuntimeError(f"{str(e)} (最后处理ID: {last_id})")

def main():
    """主函数，程序入口"""
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