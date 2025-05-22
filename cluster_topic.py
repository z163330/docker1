#import umap
import umap.umap_ as UMAP
import pandas as pd
from FlagEmbedding import BGEM3FlagModel
from bertopic import BERTopic
from hdbscan import HDBSCAN
import numpy as np
from process_data import process_text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_samples, silhouette_score
np.set_printoptions(threshold=np.inf)
pd.set_option('display.width', 800) # 设置字符显示宽度
pd.set_option('display.max_rows', None) # 设置显示最大行
pd.set_option('display.max_columns', None) # 设置显示最大列，None为显示所有列

SYSTEMPROMT_KEYPOINTS="你是一位经验丰富的文科【历史/政治/文学/哲学】阅卷老师，请从学生作答答案中根据其主题关键词和对应的文档中提取3-5个核心要点，使用规范相应学科语言风格，每个核心要点字数控制在50范围以内，要点语义连贯，符合主题关键词和文档内容事实，不能胡编乱造"
SYSTEMPROMT_SUBTRACT="你是一位经验丰富的文科【历史/政治/文学/哲学】阅卷老师，请从学生作答答案中根据其主题关键词和对应的文档中根据主题关键词进行摘要抽取，使用规范相应学科语言风格，摘要字数控制在100范围以内，语义连贯，符合主题关键词和文档内容事实，不能胡编乱造"

data_df=pd.read_csv("train_data.csv").dropna()
codes=data_df['code'].tolist()
texts=data_df['text'].tolist()
texts=process_text(texts)
from sentence_transformers import SentenceTransformer

# Create an embedding for each abstract
embedding_model = BGEM3FlagModel('/root/sdb/models/BAAI/bge-m3',use_fp16=True, trust_remote_code=True)
embeddings = embedding_model.encode(texts,batch_size=12, max_length=8192)['dense_vecs']


# We reduce the input embeddings from 1024 dimenions to 10 dimenions
umap_model = umap.UMAP( n_components=10, min_dist=0.0, metric='cosine', random_state=42)
reduced_embeddings = umap_model.fit_transform(embeddings)


# We fit the model and extract the clusters
hdbscan_model = HDBSCAN(
    min_cluster_size=50, metric='euclidean', cluster_selection_method='eom'
).fit(reduced_embeddings)
clusters = hdbscan_model.labels_
print(clusters)
cluster_silhouette_score = silhouette_score(reduced_embeddings, clusters)
print("cluster_silhouette_score:",cluster_silhouette_score)
import pandas as pd
from copy import deepcopy
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    verbose=True
).fit(texts, embeddings)
original_topics = deepcopy(topic_model.topic_representations_)
nr_topics_len=len(topic_model.get_topic_info()["Topic"].tolist())

def topic_differences(model, original_topics, nr_topics):
    """Show the differences in topic representations between two models """
    df = pd.DataFrame(columns=["Topic", "Original", "Updated"])
    for topic in range(nr_topics):
        # Extract top 5 words per topic per model
        og_words = " | ".join(list(zip(*original_topics[topic]))[0][:5])
        new_words = " | ".join(list(zip(*model.get_topic(topic)))[0][:5])
        df.loc[len(df)] = [topic, og_words, new_words]

    return df


from bertopic.representation import KeyBERTInspired

# Update our topic representations using KeyBERTInspired
representation_model = KeyBERTInspired()
topic_model.update_topics(texts, representation_model=representation_model)

# Show topic differences
topic_diff=topic_differences(topic_model, original_topics,nr_topics_len)
print(topic_diff)

from openai import OpenAI
def openai_gen(documents,keywords,systemprompt,flag):
    content_keypoints=f"我有一段包含以下文档的主题：文档{documents}，该文档主题由以下关键词描述：{keywords},根据以上信息，根据上述关键词列表里关键词,关键词必须包含在对应的文档文本内容；提取包含关键词的文档内容里2-4核心要点内容，格式如下：\
          [要点1|要点2|要点3|要点4|...]"
    content_subtractor=f"我有一段包含以下文档的主题：文档{documents}，该文档主题由以下关键词描述：{keywords},根据以上信息，根据上述关键词列表里关键词,关键词必须包含在对应的文档文本内容；根据主题关键词提取文档摘要，格式如下：\
         [摘要]"
    if flag=="keypoints":
        client = OpenAI(
            base_url='https://api.nuwaapi.com/v1',
            # sk-xxx替换为自己的key
            api_key='sk-jCrfrid9o986ZY2cnV7BvPEa1S5SjKjdyPczDMIOnPwjD1YL'
        )
        completion = client.chat.completions.create(
          model="gpt-4o",
          messages=[
            {"role": "system", "content": systemprompt},
            {"role": "user", "content":content_keypoints }
          ]
        )
        print(completion.choices[0].message)
        return completion.choices[0].message.content
    if flag=="subtractor":
        client = OpenAI(
            base_url='https://api.nuwaapi.com/v1',
            # sk-xxx替换为自己的key
            api_key='sk-jCrfrid9o986ZY2cnV7BvPEa1S5SjKjdyPczDMIOnPwjD1YL'
        )
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": systemprompt},
                {"role": "user", "content": content_subtractor}
            ]
        )
        print(completion.choices[0].message)
        return completion.choices[0].message.content

topic_info_df=topic_model.get_topic_info()
clusters_key_points=[]
clusters_subtractor=[]
for topic_id in topic_info_df['Topic']:
    keywords=topic_info_df.loc[topic_info_df['Topic']==topic_id]['Representation'].tolist()
    documents = topic_info_df.loc[topic_info_df['Topic'] == topic_id]['Representative_Docs'].tolist()
    keyPoints=openai_gen(documents,keywords,SYSTEMPROMT_KEYPOINTS,"keypoints")
    subtractor =openai_gen(documents, keywords, SYSTEMPROMT_SUBTRACT, "subtractor")
    clusters_key_points.append(keyPoints)
    clusters_subtractor.append(subtractor)
final_df=topic_info_df.drop("Representation",axis=1)
final_df["keyPoints"]=clusters_key_points
final_df["abtract"]=clusters_subtractor
final_df.to_csv('cluster_keypoint.csv')

d={}
for topic_id in topic_info_df['Topic'].tolist():
    id_docs_inds=[i for i, x in enumerate(clusters.tolist()) if x == topic_id]
    d[topic_id]=[texts[j] for j in id_docs_inds]

import json
with open('result_cluster.json', 'w') as fp:
    json.dump(d, fp,ensure_ascii=False)








