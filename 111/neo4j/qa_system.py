from py2neo import Graph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import requests
import json
import os
from dotenv import load_dotenv
from .cypher_templates import CYPHER_TEMPLATES, ENTITY_TYPES

load_dotenv()

class DeepSeekLLM:
    def __init__(self, api_key):
        self.api_key = api_key
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        
    def __call__(self, prompt):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 确保prompt是字符串
        if hasattr(prompt, 'to_string'):
            prompt_text = prompt.to_string()
        else:
            prompt_text = str(prompt)
        
        data = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": 0.3  # 降低temperature以获得更稳定的输出
        }
        
        response = requests.post(self.api_url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"API调用失败: {response.text}")

class Neo4jQASystem:
    def __init__(self, uri="bolt://localhost:7687", username="neo4j", password="78945612355zX!", api_key="sk-f8a48804adb94c4a96181d5ebd6422d3"):
        try:
            self.graph = Graph(uri, auth=(username, password))
            # 测试连接
            self.graph.run("RETURN 1").data()
            
            self.llm = DeepSeekLLM(api_key)
            
            # 获取知识图谱结构
            self.labels = self._get_labels()
            self.relationship_types = self._get_relationship_types()
            self.graph_schema = self._get_graph_schema()
            
            # 初始化Cypher生成提示模板
            self.cypher_prompt = ChatPromptTemplate.from_template("""
            你是一个专业的Neo4j知识图谱查询助手。请根据用户的问题生成合适的Cypher查询语句。

            知识图谱结构：
            节点标签：{labels}
            关系类型：{relationship_types}

            以下是可用的查询模板和示例：

            {templates}

            实体类型定义：
            {entity_types}

            用户问题：{question}

            请根据上述模板和实体类型定义，生成一个合适的Cypher查询语句。
            注意：
            1. 只能使用上述节点标签和关系类型
            2. 查询应该返回相关的节点和关系，以便回答问题
            3. 只返回Cypher语句，不要包含其他解释
            4. 如果问题涉及多个方面，请使用多个MATCH子句
            5. 标签和关系类型中如果包含空格，需要用反引号(`)括起来
            6. 确保返回的字段名称有意义，便于理解查询结果
            7. 如果查询结果可能很多，请使用LIMIT限制返回数量
            8. 使用适当的WHERE子句来过滤结果
            9. 使用ORDER BY对结果进行排序（如果需要）
            10. 使用DISTINCT去除重复结果（如果需要）
            11. 使用apoc.text.fuzzyMatch进行模糊匹配
            """)
            
            # 初始化答案生成提示模板
            self.answer_prompt = ChatPromptTemplate.from_template("""
            你是一个专业的柑橘知识问答助手。请基于以下知识图谱信息回答问题。
            
            知识图谱信息：
            {context}
            
            用户问题：{question}
            
            请给出详细、准确的回答。如果信息不足，请说明。
            """)
            
            # 构建Cypher生成链
            self.cypher_chain = (
                {
                    "labels": lambda _: ", ".join([f"`{label}`" if " " in label else label for label in self.labels]),
                    "relationship_types": lambda _: ", ".join([f"`{rel}`" if " " in rel else rel for rel in self.relationship_types]),
                    "templates": lambda _: self._format_templates(),
                    "entity_types": lambda _: self._format_entity_types(),
                    "question": RunnablePassthrough()
                }
                | self.cypher_prompt
                | self.llm
                | StrOutputParser()
            )
            
            # 构建答案生成链
            self.answer_chain = (
                {"context": self._get_context, "question": RunnablePassthrough()}
                | self.answer_prompt
                | self.llm
                | StrOutputParser()
            )
            
        except Exception as e:
            print(f"初始化失败: {str(e)}")
            raise
    
    def _format_templates(self):
        """格式化查询模板"""
        formatted_templates = []
        for template_name, template_data in CYPHER_TEMPLATES.items():
            formatted_templates.append(f"模板名称：{template_name}")
            formatted_templates.append("匹配模式：")
            for pattern in template_data["patterns"]:
                formatted_templates.append(f"- {pattern}")
            formatted_templates.append("Cypher查询：")
            formatted_templates.append(template_data["cypher"].strip())
            formatted_templates.append("")
        return "\n".join(formatted_templates)
    
    def _format_entity_types(self):
        """格式化实体类型定义"""
        formatted_types = []
        for type_name, type_values in ENTITY_TYPES.items():
            formatted_types.append(f"{type_name}: {', '.join(type_values)}")
        return "\n".join(formatted_types)
    
    def _get_labels(self):
        """获取所有节点标签"""
        try:
            query = "CALL db.labels() YIELD label RETURN label"
            labels = [record['label'] for record in self.graph.run(query).data()]
            # 处理包含空格的标签
            return [f"`{label}`" if " " in label else label for label in labels]
        except Exception as e:
            print(f"获取节点标签时发生错误: {str(e)}")
            return []
    
    def _get_relationship_types(self):
        """获取所有关系类型"""
        try:
            query = "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
            rel_types = [record['relationshipType'] for record in self.graph.run(query).data()]
            # 处理包含空格的关系类型
            return [f"`{rel}`" if " " in rel else rel for rel in rel_types]
        except Exception as e:
            print(f"获取关系类型时发生错误: {str(e)}")
            return []
    
    def _get_graph_schema(self):
        """获取知识图谱结构信息"""
        try:
            schema = []
            
            # 获取节点标签和属性
            for label in self.labels:
                # 移除反引号以进行查询
                clean_label = label.strip('`')
                query = f"MATCH (n:`{clean_label}`) WITH n LIMIT 1 CALL apoc.meta.nodeTypeProperties(n) YIELD propertyName RETURN propertyName"
                props = [record['propertyName'] for record in self.graph.run(query).data()]
                schema.append(f"节点类型 {label} 的属性: {', '.join(props)}")
            
            return "\n".join(schema)
        except Exception as e:
            print(f"获取图谱结构时发生错误: {str(e)}")
            return "无法获取图谱结构信息"
    
    def _get_context(self, question):
        """从知识图谱中获取相关上下文"""
        try:
            # 生成Cypher查询
            cypher_query = self.cypher_chain.invoke(question)
            print(f"生成的Cypher查询: {cypher_query}")
            
            # 验证查询中的标签和关系类型是否存在于图谱中
            # 检查是否使用了不存在的标签
            for label in self.labels:
                if f":{label}" in cypher_query or f"`{label}`" in cypher_query:
                    # 如果使用了标签，确保它是有效的
                    if label not in self.labels and label.strip('`') not in self.labels:
                        raise Exception(f"生成的查询中使用了不存在的节点标签。可用标签: {', '.join(self.labels)}")
                    break
            
            # 检查是否使用了不存在的关系类型
            for rel_type in self.relationship_types:
                if f":{rel_type}" in cypher_query or f"`{rel_type}`" in cypher_query:
                    # 如果使用了关系类型，确保它是有效的
                    if rel_type not in self.relationship_types and rel_type.strip('`') not in self.relationship_types:
                        raise Exception(f"生成的查询中使用了不存在的关系类型。可用关系: {', '.join(self.relationship_types)}")
                    break
            
            # 执行初始查询
            initial_results = self.graph.run(cypher_query).data()
            
            if not initial_results:
                return "未找到相关信息"
            
            # 获取初始节点的ID
            node_ids = set()
            for result in initial_results:
                for key, value in result.items():
                    if isinstance(value, dict):
                        # 使用id()函数获取节点ID
                        id_query = f"MATCH (n) WHERE n.name = '{value.get('name', '')}' RETURN id(n) as node_id"
                        id_result = self.graph.run(id_query).data()
                        if id_result:
                            node_ids.add(id_result[0]['node_id'])
            
            if not node_ids:
                # 如果通过name找不到ID，尝试直接获取节点ID
                id_query = cypher_query.replace("RETURN n", "RETURN id(n) as node_id")
                id_results = self.graph.run(id_query).data()
                for result in id_results:
                    if 'node_id' in result:
                        node_ids.add(result['node_id'])
            
            if not node_ids:
                return "未找到相关节点"
            
            # 构建扩展查询
            expand_query = """
            MATCH (n)-[r]-(m)
            WHERE id(n) IN $node_ids
            RETURN n, r, m
            LIMIT 20
            """
            
            # 执行扩展查询
            expanded_results = self.graph.run(expand_query, node_ids=list(node_ids)).data()
            
            # 合并结果
            all_results = initial_results + expanded_results
            
            # 格式化结果
            context = []
            for result in all_results:
                # 处理节点
                node_info = {}
                for key, value in result.items():
                    if isinstance(value, dict):
                        if 'name' in value:
                            node_info[key] = value['name']
                        if 'description' in value:
                            node_info[key] = value['description']
                    else:
                        node_info[key] = value
                context.append(str(node_info))
            
            return "\n".join(context) if context else "未找到相关信息"
            
        except Exception as e:
            print(f"获取上下文时发生错误: {str(e)}")
            return f"获取上下文时发生错误: {str(e)}"
    
    def query(self, question):
        """执行查询并生成答案"""
        try:
            # 获取上下文
            context = self._get_context(question)
            print(f"获取到的上下文: {context}")
            
            # 生成答案
            answer = self.answer_chain.invoke(question)
            return answer
            
        except Exception as e:
            print(f"执行查询时发生错误: {str(e)}")
            return f"查询过程中发生错误: {str(e)}"
    
    def get_graph_info(self):
        """获取知识图谱信息"""
        try:
            # 获取节点标签
            labels_query = "CALL db.labels() YIELD label RETURN label"
            labels = [record['label'] for record in self.graph.run(labels_query).data()]
            
            # 获取关系类型
            rel_types_query = "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
            relationship_types = [record['relationshipType'] for record in self.graph.run(rel_types_query).data()]
            
            # 获取节点数量
            node_counts = {}
            for label in labels:
                # 处理包含空格的标签
                if " " in label:
                    count_query = f"MATCH (n:`{label}`) RETURN count(n) as count"
                else:
                    count_query = f"MATCH (n:{label}) RETURN count(n) as count"
                node_counts[label] = self.graph.run(count_query).data()[0]['count']
            
            # 获取关系数量
            rel_counts = {}
            for rel_type in relationship_types:
                # 处理包含空格的关系类型
                if " " in rel_type:
                    count_query = f"MATCH ()-[r:`{rel_type}`]->() RETURN count(r) as count"
                else:
                    count_query = f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count"
                rel_counts[rel_type] = self.graph.run(count_query).data()[0]['count']
            
            return {
                "labels": labels,
                "relationship_types": relationship_types,
                "node_counts": node_counts,
                "rel_counts": rel_counts
            }
            
        except Exception as e:
            print(f"获取图谱信息时发生错误: {str(e)}")
            return None 