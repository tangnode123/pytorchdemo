from transformers import pipeline

# 加载命名实体识别模型
ner_model = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

def extract_entities(text):
    # 使用NER模型提取命名实体
    entities = ner_model(text)
    return [entity['word'] for entity in entities]

# 模型输出的例子
model_output = "在2023年，OpenAI发布了一款名为ChatGPT-4的新型语言模型。该模型在自然语言处理方面取得了显著的进展。"

# 提取命名实体
entities = extract_entities(model_output)

# 打印结果
print("提取的命名实体：", entities)