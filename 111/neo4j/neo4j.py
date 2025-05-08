# -*- coding: utf-8 -*-

from typing import List, Dict, Any, Optional
from py2neo import Graph, Node, Relationship, Transaction
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np
import time
import logging
from pathlib import Path
import os
from dotenv import load_dotenv

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('neo4j_import.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

class Neo4jConfig:
    """Neo4j配置类"""
    def __init__(self):
        self.uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.user = os.getenv('NEO4J_USER', 'neo4j')
        self.password = os.getenv('NEO4J_PASSWORD', '78945612355zX!')
        self.data_file_path = os.getenv('DATA_FILE_PATH', 'data/qwen_cluster_mapped.jsonl')
        self.sentence_model = os.getenv('SENTENCE_MODEL', 'all-MiniLM-L6-v2')
        self.batch_size = int(os.getenv('BATCH_SIZE', '1000'))
        self.batch_delay = float(os.getenv('BATCH_DELAY', '0.1'))
        self.max_retries = int(os.getenv('MAX_RETRIES', '3'))
        self.retry_delay = float(os.getenv('RETRY_DELAY', '1.0'))

class Neo4jImporter:
    """Neo4j数据导入器"""
    def __init__(self, config: Neo4jConfig):
        self.config = config
        self.graph = None
        self.model = None

    def connect(self) -> bool:
        """连接到Neo4j数据库"""
        for attempt in range(self.config.max_retries):
            try:
                logger.info(f"尝试连接到Neo4j (尝试 {attempt + 1}/{self.config.max_retries})")
                self.graph = Graph(self.config.uri, auth=(self.config.user, self.config.password))
                self.graph.run("RETURN 1")  # 测试连接
                logger.info("Neo4j连接成功")
                return True
            except Exception as e:
                logger.error(f"连接失败: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                else:
                    return False

    def load_model(self) -> None:
        """加载句子转换模型"""
        try:
            logger.info(f"正在加载模型: {self.config.sentence_model}")
            self.model = SentenceTransformer(self.config.sentence_model)
            logger.info("模型加载成功")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise

    def load_data(self) -> List[Dict[str, Any]]:
        """加载JSONL数据"""
        data = []
        file_path = Path(self.config.data_file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="读取数据文件"):
                    try:
                        # 解析JSONL格式的每一行
                        item = json.loads(line.strip())
                        # 处理嵌套的triples
                        if 'triples' in item and isinstance(item['triples'], list):
                            for triple in item['triples']:
                                if self._validate_data_item(triple):
                                    data.append(triple)
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON解析错误: {e}, 行内容: {line.strip()}")
                    except Exception as e:
                        logger.error(f"数据处理错误: {e}")
            logger.info(f"成功加载 {len(data)} 条数据")
            return data
        except Exception as e:
            logger.error(f"文件读取错误: {e}")
            raise

    def _validate_data_item(self, item: Dict[str, Any]) -> bool:
        """验证数据项"""
        required_fields = ['subject_name', 'subject_type', 'object_name', 'object_type', 'relation_type']
        if not all(item.get(field) for field in required_fields):
            logger.warning(f"数据项缺少必要字段: {item}")
            return False
        return True

    def create_indexes(self) -> None:
        """创建必要的索引"""
        try:
            indexes = [
                "CREATE INDEX IF NOT EXISTS FOR (n:Node) ON (n.name)",
                "CREATE INDEX IF NOT EXISTS FOR (n:Subject) ON (n.name)",
                "CREATE INDEX IF NOT EXISTS FOR (n:Object) ON (n.name)"
            ]
            
            for index in indexes:
                self.graph.run(index)
            logger.info("索引创建成功")
        except Exception as e:
            logger.error(f"索引创建失败: {e}")
            raise

    def process_batch(self, batch: List[Dict[str, Any]]) -> None:
        """处理单个批次的数据"""
        tx = self.graph.begin()
        try:
            for item in batch:
                params = {
                    "s_name": item['subject_name'],
                    "o_name": item['object_name'],
                    "rel_type": item['relation_type']
                }

                # 创建/更新主语节点
                subject_query = f"""
                MERGE (s:`{item['subject_type']}` {{name: $s_name}})
                ON CREATE SET s.created_at = timestamp()
                ON MATCH SET s.last_seen = timestamp()
                """
                tx.run(subject_query, params)

                # 创建/更新宾语节点
                object_query = f"""
                MERGE (o:`{item['object_type']}` {{name: $o_name}})
                ON CREATE SET o.created_at = timestamp()
                ON MATCH SET o.last_seen = timestamp()
                """
                tx.run(object_query, params)

                # 创建关系
                relation_query = f"""
                MATCH (s:`{item['subject_type']}` {{name: $s_name}})
                MATCH (o:`{item['object_type']}` {{name: $o_name}})
                MERGE (s)-[r:`{item['relation_type']}`]->(o)
                ON CREATE SET r.created_at = timestamp()
                """
                tx.run(relation_query, params)

            tx.commit()
        except Exception as e:
            tx.rollback()
            logger.error(f"批处理失败: {e}")
            raise

    def import_data(self, data: List[Dict[str, Any]]) -> None:
        """导入数据到Neo4j"""
        total_batches = (len(data) + self.config.batch_size - 1) // self.config.batch_size
        
        for i in tqdm(range(0, len(data), self.config.batch_size), desc="导入数据"):
            batch = data[i:i + self.config.batch_size]
            for attempt in range(self.config.max_retries):
                try:
                    self.process_batch(batch)
                    if self.config.batch_delay > 0:
                        time.sleep(self.config.batch_delay)
                    break
                except Exception as e:
                    if attempt == self.config.max_retries - 1:
                        logger.error(f"批处理 {i//self.config.batch_size + 1} 失败: {e}")
                        raise
                    time.sleep(self.config.retry_delay)

def main():
    try:
        # 初始化配置
        config = Neo4jConfig()
        
        # 创建导入器实例
        importer = Neo4jImporter(config)
        
        # 连接数据库
        if not importer.connect():
            raise ConnectionError("无法连接到Neo4j数据库")
        
        # 创建索引
        importer.create_indexes()
        
        # 加载数据
        data = importer.load_data()
        if not data:
            raise ValueError("没有数据可导入")
        
        # 导入数据
        importer.import_data(data)
        
        logger.info("数据导入完成")
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        raise

if __name__ == "__main__":
    main()