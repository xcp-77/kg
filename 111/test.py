import openai
import os

# 设置 API 配置
openai.api_base = "https://api.linkapi.org/v1"
openai.api_key = "sk-Lc5Z2IqBvBfhb0oq0f1691AcD3554bAe91F32786F4DcEd1a"

try:
    # 尝试调用 API
    response = openai.ChatCompletion.create(
        model="deepseek-r1:1.5b",
        messages=[
            {"role": "user", "content": "你好，请回复'测试成功'"}
        ]
    )
    
    # 打印响应
    print("API 响应:", response.choices[0].message.content)
    print("测试成功！API 连接正常。")
    
except Exception as e:
    print("测试失败！错误信息:", str(e))
