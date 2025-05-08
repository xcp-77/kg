CYPHER_TEMPLATES = {
    # 查找特定名称的节点
    "find_node_by_name": {
        "patterns": [
            "查找名为{entity}的节点",
            "找到{entity}节点",
            "搜索名字是{entity}的节点"
        ],
        "cypher": "MATCH (n) WHERE apoc.text.fuzzyMatch(n.name, '{entity}') RETURN n LIMIT 10"
    },
    
    # 查找两个节点之间的关系
    "find_relationship": {
        "patterns": [
            "{entity1}和{entity2}之间的关系是什么",
            "查找{entity1}与{entity2}之间的关系",
            "{entity1}与{entity2}有什么联系"
        ],
        "cypher": """
        MATCH (n1)-[r]-(n2) 
        WHERE apoc.text.fuzzyMatch(n1.name, '{entity1}') 
        AND apoc.text.fuzzyMatch(n2.name, '{entity2}') 
        RETURN n1, r, n2 LIMIT 10
        """
    },
    
    # 查找具有特定关系的节点
    "find_nodes_with_relation": {
        "patterns": [
            "谁与{entity}有{relation}关系",
            "查找与{entity}存在{relation}关系的节点",
            "找出所有和{entity}{relation}相关的节点"
        ],
        "cypher": """
        MATCH (n1)-[r:{relation}]-(n2)
        WHERE apoc.text.fuzzyMatch(n1.name, '{entity}')
        RETURN n1, r, n2 LIMIT 10
        """
    },
    
    # 查找水果的风味特征
    "find_fruit_flavors": {
        "patterns": [
            "{fruit}有什么风味特征",
            "{fruit}的味道是什么样的",
            "描述{fruit}的风味"
        ],
        "cypher": """
        MATCH (f:Fruit)-[r:hasFlavor]->(fl:Flavor)
        WHERE apoc.text.fuzzyMatch(f.name, '{fruit}')
        RETURN f, r, fl LIMIT 10
        """
    },
    
    # 查找具有特定风味的水果
    "find_fruits_by_flavor": {
        "patterns": [
            "哪些水果具有{flavor}风味",
            "有{flavor}味道的水果有哪些",
            "找出带{flavor}风味的水果"
        ],
        "cypher": """
        MATCH (f:Fruit)-[r:hasFlavor]->(fl:Flavor)
        WHERE apoc.text.fuzzyMatch(fl.name, '{flavor}')
        RETURN f, r, fl LIMIT 10
        """
    },
    
    # 查找水果中的化合物
    "find_fruit_compounds": {
        "patterns": [
            "{fruit}含有哪些化合物",
            "{fruit}的化学成分有哪些",
            "查找{fruit}中的化合物"
        ],
        "cypher": """
        MATCH (f:Fruit)-[r:hasCompound]->(c:Compound)
        WHERE apoc.text.fuzzyMatch(f.name, '{fruit}')
        RETURN f, r, c LIMIT 10
        """
    },
    
    # 查找化合物的代谢途径
    "find_compound_metabolism": {
        "patterns": [
            "{compound}的代谢途径是什么",
            "{compound}如何被代谢",
            "查找{compound}的代谢过程"
        ],
        "cypher": """
        MATCH (c:Compound)-[p:precursor]->(m:Metabolism)-[mt:metabolite]->(mp:Compound)
        WHERE apoc.text.fuzzyMatch(c.name, '{compound}')
        RETURN c, p, m, mt, mp LIMIT 10
        """
    },
    
    # 查找酶催化的反应
    "find_enzyme_reactions": {
        "patterns": [
            "{enzyme}催化什么反应",
            "{enzyme}参与的反应是什么",
            "查找{enzyme}相关的反应"
        ],
        "cypher": """
        MATCH (e:Enzyme)-[c:catalyze]->(r:Reaction)
        WHERE apoc.text.fuzzyMatch(e.name, '{enzyme}')
        RETURN e, c, r LIMIT 10
        """
    },
    
    # 查找反应相关的化合物
    "find_reaction_compounds": {
        "patterns": [
            "{reaction}反应涉及哪些化合物",
            "{reaction}的反应物和产物是什么",
            "查找{reaction}相关的化合物"
        ],
        "cypher": """
        MATCH (c:Compound)-[p:participate]->(r:Reaction)-[pr:product]->(pc:Compound)
        WHERE apoc.text.fuzzyMatch(r.name, '{reaction}')
        RETURN c, p, r, pr, pc LIMIT 10
        """
    },
    
    # 按风味分类查询
    "find_flavor_by_category": {
        "patterns": [
            "有哪些{flavor_class}风味",
            "属于{flavor_class}的风味有哪些",
            "查找{flavor_class}类的风味"
        ],
        "cypher": """
        MATCH (f:Flavor)
        WHERE f.flavor_class = '{flavor_class}' OR f.flavor_subclass = '{flavor_class}'
        RETURN f LIMIT 10
        """
    },
    
    # 按化合物分类查询
    "find_compound_by_category": {
        "patterns": [
            "有哪些{compound_class}类化合物",
            "属于{compound_class}的化合物有哪些",
            "查找{compound_class}类的化合物"
        ],
        "cypher": """
        MATCH (c:Compound)
        WHERE c.compound_class = '{compound_class}'
        RETURN c LIMIT 10
        """
    },
    
    # 查找化合物激发的风味
    "find_compound_activated_flavors": {
        "patterns": [
            "{compound}会激发什么风味",
            "{compound}产生的味道是什么",
            "{compound}贡献了哪些风味"
        ],
        "cypher": """
        MATCH (c:Compound)-[r:activate]->(f:Flavor)
        WHERE apoc.text.fuzzyMatch(c.name, '{compound}')
        RETURN c, r, f LIMIT 10
        """
    },
    
    # 查找代谢途径相关的反应
    "find_metabolism_reactions": {
        "patterns": [
            "{metabolism}代谢包含哪些反应",
            "{metabolism}的反应过程是什么",
            "查找{metabolism}相关的反应步骤"
        ],
        "cypher": """
        MATCH (m:Metabolism)<-[c:constitutes]-(r:Reaction)
        WHERE apoc.text.fuzzyMatch(m.name, '{metabolism}')
        RETURN m, c, r LIMIT 10
        """
    },
    
    # 查找反应集中的具体反应
    "find_reaction_set_details": {
        "patterns": [
            "{reaction_set}包含哪些具体反应",
            "{reaction_set}的组成反应是什么",
            "查找{reaction_set}中的反应步骤"
        ],
        "cypher": """
        MATCH (rs:ReactionSet)<-[p:partof]-(r:Reaction)
        WHERE apoc.text.fuzzyMatch(rs.name, '{reaction_set}')
        RETURN rs, p, r LIMIT 10
        """
    },
    
    # 查找特定酶类型的酶
    "find_enzyme_by_category": {
        "patterns": [
            "有哪些{enzyme_class}类酶",
            "属于{enzyme_class}的酶有哪些",
            "查找{enzyme_class}类的酶"
        ],
        "cypher": """
        MATCH (e:Enzyme)
        WHERE e.enzyme_class = '{enzyme_class}'
        RETURN e LIMIT 10
        """
    },
    
    # 查找化合物参与的完整代谢路径
    "find_compound_metabolism_pathway": {
        "patterns": [
            "{compound}的完整代谢路径是什么",
            "从{compound}开始的代谢过程",
            "追踪{compound}的代谢转化"
        ],
        "cypher": """
        MATCH path = (c1:Compound)-[:precursor]->(m:Metabolism)-[:metabolite]->(c2:Compound)
        WHERE apoc.text.fuzzyMatch(c1.name, '{compound}')
        RETURN path LIMIT 10
        """
    }
}

# 扩展实体类型定义
ENTITY_TYPES = {
    "FRUIT": ["水果", "柑橘", "果实"],
    "FLAVOR": ["风味", "味道", "香气", "气味", "taste", "aroma"],
    "FLAVOR_CLASS": ["气味", "味觉", "水果香", "草本香", "油脂味", "香料味", "花香", "木香", 
                    "酸", "甜", "苦", "咸", "鲜"],
    "COMPOUND": ["化合物", "成分", "物质"],
    "COMPOUND_CLASS": ["香味化合物", "色素化合物", "味觉化合物", 
                      "酯类", "萜烯", "芳香族", "醇类", "醛类", "酮类", "内酯类",
                      "叶绿素", "花青素", "类胡萝卜素"],
    "ENZYME": ["酶", "催化剂"],
    "ENZYME_CLASS": ["氧化还原酶", "转移酶", "水解酶", "裂解酶", "异构酶", "合成酶", "转位酶"],
    "REACTION": ["反应", "过程"],
    "REACTION_CLASS": ["羧酸代谢", "核苷酸糖代谢", "芳香族化合物降解"],
    "METABOLISM": ["代谢", "代谢途径", "生化过程"],
    "METABOLISM_CLASS": ["碳水化合物代谢", "能量代谢", "脂质代谢", "核苷酸代谢", 
                        "氨基酸代谢", "糖类代谢", "维生素代谢", "萜烯代谢"]
} 