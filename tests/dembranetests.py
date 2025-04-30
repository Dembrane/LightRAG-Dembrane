from lightrag.lightrag import LightRAG
from typing import Optional, Any
import numpy as np
from litellm import completion, embedding
from dotenv import load_dotenv
import os
from lightrag.kg.shared_storage import initialize_pipeline_status
import asyncio
import uuid
import nest_asyncio
from lightrag.kg.postgres_impl import PostgreSQLDB
from dataclasses import dataclass
import random
import json
from lightrag.lightrag import QueryParam
nest_asyncio.apply()

load_dotenv()

LIGHTRAG_LITELLM_MODEL = os.getenv("LIGHTRAG_LITELLM_MODEL")
LIGHTRAG_LITELLM_API_KEY = os.getenv("LIGHTRAG_LITELLM_API_KEY")
LIGHTRAG_LITELLM_API_VERSION = os.getenv("LIGHTRAG_LITELLM_API_VERSION")
LIGHTRAG_LITELLM_API_BASE = os.getenv("LIGHTRAG_LITELLM_API_BASE")
LIGHTRAG_LITELLM_EMBEDDING_MODEL = os.getenv("LIGHTRAG_LITELLM_EMBEDDING_MODEL")
LIGHTRAG_LITELLM_EMBEDDING_API_KEY = os.getenv("LIGHTRAG_LITELLM_EMBEDDING_API_KEY")
LIGHTRAG_LITELLM_EMBEDDING_API_VERSION = os.getenv("LIGHTRAG_LITELLM_EMBEDDING_API_VERSION")
LIGHTRAG_LITELLM_EMBEDDING_API_BASE = os.getenv("LIGHTRAG_LITELLM_EMBEDDING_API_BASE")
os.environ["POSTGRES_HOST"] = "localhost"
os.environ["POSTGRES_PORT"] = "5432"
os.environ["POSTGRES_USER"] = "dembrane"
os.environ["POSTGRES_PASSWORD"] = "dembrane"
os.environ["POSTGRES_DATABASE"] = "dembrane"
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "admin@dembrane"
# os.environ["WORKSPACE"] = "test_dembrane"


test_text = """
Introducing Dembrane Echo
We are excited to share Dembrane Echo with you - Our first product of a series of DemTech tools designed to minimise the gap between decision makers and those impacted by their choices. 
Say Hello to Echo
notion image
 
One thing that keeps our team up at night is seeing good ideas die. We often see organisations and communities struggling to effectively include people in their decision-making processes. They hold meetings, send surveys, and genuinely want to hear from everyone, but when it comes to using all that input, people's insights into their neighbourhoods, workplaces, and communities get lost in translation or buried in reports that no one had time to read.
 
We built Echo to make it simple to engage a large group of stakeholders. Record and ask.
We think Echo can make your work with communities more meaningful and effective. Want to see how? Book a call with Jules to try it out.
 
A simple portal, by design
notion image
 
Echo works like a really good listener - it captures conversations naturally and helps make sense of them. When someone shares their experience about living on a busy street, or their idea for making the local park more welcoming, those insights are transcribed in a robust and secure way in the EU, and stored in a dedicated dashboard. We kept things simple: Participants can just open a link or scan a QR code and start talking. No apps to install means minimal barriers getting in the way of the conversation.
From conversations to actionable data, in minutes
notion image
 
In the dashboard, conversations can be turned into actionable insights that decision-makers can use. With Echo you can create custom views to explore the conversation data. The AI organises relevant quotes and insights into clear clusters, making it easy to spot both common themes and unique perspectives that might otherwise get missed. Or use the AI chat to dive deep into particular themes across all your sessions - just select the conversations you want to explore and ask.
Privacy first, always
We recognise the importance of a transparent and secure process in handling these deeply personal inputs. When using Echo, we don't know who said what: only the content is stored. The data is not used for anything other than each hosts own analysis. All of Dembrane Echo, including the AI models we run, are hosted on private Azure servers based in the EU. The platform is fully GDPR compliant and we are in the midst of obtaining our ISO27001 and other industry specific certifications.
Where Echo is used today
We are proud that even in its early stages, Echo is already making waves across different communities. In Brabant, it helped amplify the voices of more than seventy young people during their Jongerentop youth summit. In 's-Hertogenbosch, over 200 citizens council participants used Echo to share their stories with each other. Setup takes under 5 minutes, and insights appear within 2 minutes of conversations. Most importantly, participants consistently report feeling more heard and valued than they would with traditional methods (audio recording with no ECHO feedback, or with manually taken notes or post-its).
 
Building better together
Echo is currently in testing phase, and we're working closely with our partners to refine the experience. Our mission is to build democratic infrastructure that empowers every voice to participate in decisions that impact their community. While there's lots of work ahead, Echo represents an important step toward this mission.
 
We look forward to sharing further progress with you soon. In the meantime we hope you enjoy using Echo as much as we do.
 
The Dembrane Team
Built by Dembrane in Europe
Published December 10th, 2024
"""

@dataclass
class TestData:
    test_id: str
    chunk_id: str
    
    def to_dict(self) -> dict:
        return {
            "test_id": self.test_id,
            "chunk_id": self.chunk_id
        }

pg_db = PostgreSQLDB(config={
        "host": os.environ["POSTGRES_HOST"],
        "port": os.environ["POSTGRES_PORT"],
        "user": os.environ["POSTGRES_USER"],
        "password": os.environ["POSTGRES_PASSWORD"],
        "database": os.environ["POSTGRES_DATABASE"]
    })
async def llm_model_func(
    prompt: str, 
    system_prompt: Optional[str] = None, 
    history_messages: Optional[list[dict]] = None, 
    **kwargs: Any
) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    chat_completion = completion(
        model=f"{LIGHTRAG_LITELLM_MODEL}",  # litellm format for Azure models
        messages=messages,
        temperature=kwargs.get("temperature", 0.2),
        api_key=LIGHTRAG_LITELLM_API_KEY,
        api_version=LIGHTRAG_LITELLM_API_VERSION,
        api_base=LIGHTRAG_LITELLM_API_BASE
    )
    return chat_completion.choices[0].message.content

async def embedding_func(texts: list[str]) -> np.ndarray:
    # Bug in litellm forcing us to do this: https://github.com/BerriAI/litellm/issues/6967
    nd_arr_response = []
    for text in texts:
        temp = embedding(
            model=f"{LIGHTRAG_LITELLM_EMBEDDING_MODEL}",
            input=text,
            api_key=str(LIGHTRAG_LITELLM_EMBEDDING_API_KEY),
            api_version=str(LIGHTRAG_LITELLM_EMBEDDING_API_VERSION),
            api_base=str(LIGHTRAG_LITELLM_EMBEDDING_API_BASE),
        )
        nd_arr_response.append(temp['data'][0]['embedding'])
    return np.array(nd_arr_response)

async def initialize_rag():
    lightrag = LightRAG(
            working_dir=None,
            llm_model_func=llm_model_func,
            embedding_func=embedding_func,
            kv_storage="PGKVStorage",
            doc_status_storage="PGDocStatusStorage",
            graph_storage="Neo4JStorage",
            vector_storage="PGVectorStorage",
            vector_db_storage_cls_kwargs={
                "cosine_better_than_threshold": 0.2
            },
            namespace_prefix="test_dembrane"
        )
    await lightrag.initialize_storages()
    await initialize_pipeline_status()
    return lightrag

async def fetch_data(sql: str):
    await pg_db.initdb()
    res = await pg_db.query(sql, multirows=True)
    return res

def test_insert_data(count: int = 1):
    sample_test_text = test_text + '@' * random.randint(0,200) #md5 hash will be the same for the same text
    test_id = str(uuid.uuid4())
    lightrag = asyncio.run(initialize_rag())
    lightrag.insert(input=sample_test_text, 
                    ids=[test_id],
                    file_paths=[f"test_file_path_{count}"])
    sql = f"SELECT * FROM lightrag_doc_chunks WHERE full_doc_id = '{test_id}'"
    res = asyncio.run(fetch_data(sql))
    assert len(res) >= 1
    chunk_id = res[0]['id']
    sql = f"SELECT * FROM lightrag_chunk_graph_map WHERE chunk_id = '{chunk_id}'"
    res = asyncio.run(fetch_data(sql))
    assert len(res) >= 1
    print("Insertion successful")
    return TestData(test_id=test_id, chunk_id=chunk_id)



def test_multiple_insert_data(run_count: int = 10):
    for count in range(run_count):
        # read test_data from file. If file does not exist, create it and add test_data
        if not os.path.exists('test_data.json'):
            test_data_list = []
        else:
            with open('test_data.json', 'r') as f:
                test_data_list = json.load(f)
        
        test_data = test_insert_data(count)
        # Convert TestData to dictionary before appending
        test_data_list.append(test_data.to_dict())
        with open('test_data.json', 'w') as f:
            json.dump(test_data_list, f)

def test_query_for_prompt_generation():
    # Load the most recent test data from file
    with open('test_data.json', 'r') as f:
        test_data_list = json.load(f)
        if not test_data_list:
            raise ValueError("No test data available. Run test_insert_data first.")
        latest_test_data = test_data_list[-1]  # Get the most recent test data
    
    lightrag = asyncio.run(initialize_rag())
    res = lightrag.query(query="What is Dembrane Echo?",
                        param=QueryParam(
                            only_need_prompt = True,
                            ids=[latest_test_data["test_id"]],
                            mode="mix"
                        ))
    assert len(res) >= 1

async def test_deletion_by_file_path():
    with open('test_data.json', 'r') as f:
        test_data_list = json.load(f)
        if not test_data_list:
            raise ValueError("No test data available. Run test_insert_data first.")
        latest_test_data = test_data_list[-1]  # Get the most recent test data
    lightrag = asyncio.run(initialize_rag())
    await lightrag.adelete_by_doc_id(doc_id=latest_test_data["test_id"])


if __name__ == "__main__":
    run_count = 1
    test_data_list = []
    if os.path.exists('test_data.json'):
        with open('test_data.json', 'r') as f:
            test_data_list = json.load(f)
    if len(test_data_list) < run_count:
        test_multiple_insert_data(run_count=run_count)
    test_query_for_prompt_generation()
    asyncio.run(test_deletion_by_file_path())


