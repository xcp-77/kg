from py2neo import Graph
import sys
from py2neo.errors import ServiceUnavailable

def connect_to_neo4j():
    try:
        # 尝试使用不同的连接方式
        connection_url = "bolt://localhost:7687"
        auth = ("neo4j", "78945612355zX!")
        
        print(f"正在尝试连接到: {connection_url}")
        print(f"使用认证信息: 用户名=neo4j")
        
        # 设置连接超时
        graph = Graph(connection_url, auth=auth, secure=False, max_connection_lifetime=100)
        
        # 测试连接
        result = graph.run("RETURN 1").data()
        print("连接成功！")
        print(f"测试查询结果: {result}")
        return graph
        
    except ServiceUnavailable as e:
        print("\n连接失败：服务不可用")
        print("可能的原因：")
        print("1. Neo4j 服务未启动")
        print("2. Bolt 连接未启用")
        print("3. 端口 7687 被占用或未开放")
        print("\n请检查：")
        print("- 在 Neo4j 浏览器中执行 ':server status' 查看服务状态")
        print("- 检查 Neo4j 配置文件中的 Bolt 设置")
        print("- 确认防火墙设置")
        raise
        
    except Exception as e:
        print(f"\n连接失败，错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        raise

def test_connection():
    try:
        print("\n开始测试 Neo4j 连接...")
        graph = connect_to_neo4j()
        return True
    except Exception as e:
        print(f"\n测试失败: {e}")
        return False

if __name__ == "__main__":
    test_connection()
