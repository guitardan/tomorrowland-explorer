import os, requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from random import shuffle, random
from re import sub
from time import perf_counter, sleep
from pymongo import MongoClient
#from pymongo.collection import MutableMapping # inherit from to make a custom Mongo-serializable class
from dotenv import load_dotenv
from numpy import save

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore # AstraDB # Chroma # FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQAWithSourcesChain
#from langchain.text_splitter import RecursiveCharacterTextSplitter # TODO
#splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=0)
#splits = splitter.split_documents(documents)

class WebPage():
    def __init__(self, url: str, soup: BeautifulSoup):
        self.url = url
        title_elem = soup.find('title')
        self.title = title_elem.text if title_elem else ''
        main_elem = soup.find('main')
        self.main_text = sub(r'\n+', '\n', main_elem.text) if main_elem else ''
        self.text_blocks = []
        for text_block in soup.select('div[class^="TextBlockDefault_text__"]'):
            self.text_blocks.append(text_block.text)
    
    def to_dict(self) -> dict:
        return {
            'url': self.url,
            'title': self.title,
            'main_text': self.main_text,
            'text_blocks': self.text_blocks
        }

def get_links(url: str, keyword: str) -> set[str]:
    links = set()
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            if href.startswith('/'):
                parsed_url = urlparse(url)
                href = f'{parsed_url.scheme}://{parsed_url.netloc}{href}'
            elif href.startswith('mailto:'):
                continue
            if keyword in href:
                links.add(fix_any_duplicate_domain(href))
    except requests.RequestException as e:
        print(f'Error fetching {url}: {e}')
    return links

def fix_any_duplicate_domain(txt: str) -> str:
    # https://store.tomorrowland.com//store.tomorrowland.com/cdn/shop/...
    if txt.count('//') <= 1:
        return txt
    return txt.split('//')[-1]

def get_sub_links(max_sub_links:int = 500) -> set[str]:
    print('spidering links...')
    links = set()
    new_links = {BASE_DOMAIN}
    while new_links - links:
        sub_links = set()
        for link in new_links - links:
            sub_links.update(get_links(link, KEYWORD))
            print(f'{len(sub_links)} found on {link}')
            if len(sub_links) > max_sub_links:
                links.update(new_links)
                links.update(sub_links)
                print(f'... {len(links)} crawled.')
                return links

        links = new_links
        new_links = sub_links

def get_contents(url: str) -> WebPage:
    if not urlparse(url).scheme:
        url = f'https://{url}'
    response = requests.get(url)
    try:
        html = response.content.decode("utf-8")
    except UnicodeDecodeError as e:
        print(f'Error decoding {url} to UTF-8: {e}')
        return ''
    soup = BeautifulSoup(html, 'html.parser')
    return WebPage(url, soup)

def get_mongo_collection():
    client = MongoClient(os.environ['MONGODB_CONN_STR'])
    database = client.get_database(os.environ['MONGODB_DATABASE'])
    return database.get_collection(KEYWORD)

def scrape():
    links = list(get_sub_links(MAX_LINKS))
    with open('links.txt', 'w') as f:
        f.writelines([f'{link}\n' for link in links])
    #shuffle(links)

    #idx = 0
    #if not os.path.exists(keyword):
    #    os.mkdir(keyword)
    max_wait_s = 2
    for url in links: # [:10]:
        #print(url)
        sleep(max_wait_s * random()) 
        page = get_contents(url)
        if page:
            #with open(os.path.join(keyword, f'{idx}.txt'), 'w') as f:
            #    f.write(txt)
            #idx += 1
            collection.insert_one(page.to_dict())

def local_scraped_txts_to_langchain_docs() -> list[Document]:
    docs = []
    for fn in os.listdir(KEYWORD):
        with open(os.path.join(KEYWORD, fn), 'r') as f:
            txt = f.read()
        docs.append(
            Document(
                page_content=txt,
                metadata={'source': f'{BASE_DOMAIN}/{fn}'} # dummy val, required by VectorStore
                )
            )
    return docs

def mongo_to_langchain_docs() -> list[Document]:
    lc_docs = []
    for doc in collection.find({}):
        page_content = ''
        if doc['title']:
            page_content += doc['title'] + '\n\n'
        if doc['main_text']:
            page_content += doc['main_text'] + '\n'
        if doc['text_blocks']:
            page_content += ''.join(doc['text_blocks'])
        lc_docs.append(Document(
            id=doc['_id'],
            page_content=page_content,
            metadata={
                "source": doc['url'],
                "title": doc['title']
                }))
    return lc_docs

def get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    # ResourceExhausted: "Request a higher quota limit." => limit the number of docs passed with num_docs?
    # better to locally cache the db and limit calls to the embedding generation API endpoint
    embedding_model = GoogleGenerativeAIEmbeddings(
        google_api_key=os.environ['GOOGLE_API_KEY'],
        model="models/text-embedding-004"
        ) 
    #vector = embedding_model.embed_query("hello, world!")
    #print(vector[:5])

    #pip install -U langchain-huggingface # ERROR: Cannot install langchain-huggingface because these package versions have conflicting dependencies.
    #from transformers import SentenceTransformer 
    #embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    #from langchain_ollama import OllamaEmbeddings
    #embedding_model = OllamaEmbeddings(model="llama3")
    return embedding_model # SentenceTransformer("all-MiniLM-L6-v2") # 

def limit_docs(lc_docs: list[Document]) -> list[Document]:
    # prefer non-"store" merchandise-related docs, but include enough to reach num_docs'''
    store_docs = [d for d in lc_docs if 'store' in d.metadata['source']]
    non_store_docs= [d for d in lc_docs if 'store' not in d.metadata['source']]
    ret = non_store_docs
    ret.extend(store_docs[:NUM_DOCS-len(non_store_docs)])
    return ret

def create_db():
    global db
    st = perf_counter()
    if os.path.exists(LOCAL_DB_PATH):
        db = InMemoryVectorStore.load(LOCAL_DB_PATH, get_embedding_model())
        print(f'InMemoryVectorStore exists, loaded in {perf_counter() - st:.2f}s')
        return
    #lc_docs = local_scraped_txts_to_langchain_docs(keyword)
    lc_docs = mongo_to_langchain_docs()
    #lc_docs = limit_docs(lc_docs)

    db = InMemoryVectorStore.from_documents(
        lc_docs,
        get_embedding_model(),
        collection_name='docs',
        #persist_directory=LOCAL_DB_PATH,
    )
    print(f'InMemoryVectorStore built in {perf_counter() - st:.2f}s')
    db.dump(LOCAL_DB_PATH)
    #db.persist() # AttributeError: 'InMemoryVectorStore' object has no attribute 'persist' -> only Chroma

def perform_query(query: str) -> str:
    try:
        db
        print("database does not yet exist")
    except NameError:
        create_db()

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # other params...
    )

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        #chain_type="stuff",
        retriever=db.as_retriever(),
        chain_type_kwargs={"verbose": True}
    )

    result = chain(
        {"question": query},
        return_only_outputs=True
        )
    return result

BASE_DOMAIN = 'https://www.tomorrowland.com'
KEYWORD = 'tomorrowland'
MAX_LINKS = 2_000 # 1_000 # 250 # 100 # 
NUM_DOCS = 1_500
LOCAL_DB_PATH = 'vector_store_db'

load_dotenv('.env')
collection = get_mongo_collection()
# collection.delete_many({})

#scrape()
