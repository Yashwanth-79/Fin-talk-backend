import os
from flask import Flask, request, jsonify, send_from_directory # Import send_from_directory
from flask_cors import CORS  # Import CORS
from langchain_groq import ChatGroq
from langchain_community.tools import BraveSearch
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
from datetime import datetime
from transformers import pipeline
from typing import List, Dict
import re
from pydantic.v1 import BaseModel
from neo4j import GraphDatabase
import json

# --------------------------- API KEYS & CONFIG ---------------------------
NEWS_API_KEY = "ca232d0ca1de49f5ba86731e0d6839d4"
FMP_API_KEY = "1PCcILly9sZFq7lSTu9eDNxOxy85Rdl1"
GROQ_API_KEY = "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"
BRAVE_API_KEY = "BSAHkqCaI8WeYS23hfogkBl19fy-tjn"

NEO4J_URI = "neo4j+s://59b6d873.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "XHs0ISCcQrrGYkg580ENlVIjdxFZOZkq6W0VcP7IkBk"

# --------------------------- Flask App Initialization ---------------------------
app = Flask(__name__, static_folder='client') # Set static folder
CORS(app)  # Enable CORS for all routes

# --------------------------- FETCH FUNCTIONS ---------------------------
def fetch_news(query, page=1, page_size=30):
    """Fetch news articles using NewsAPI."""
    BASE_URL = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "apiKey": NEWS_API_KEY,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": page_size,
        "page": page
    }
    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None

# --------------------------- CLEANING FUNCTIONS ---------------------------
def clean_news_data(raw_news_data):
    """Clean news data by removing HTML tags and normalizing dates using standard Python."""
    articles = raw_news_data.get("articles", [])
    cleaned_data = []
    for article in articles:
        description = article.get("description", "") or ""
        content = article.get("content", "") or ""
        try:
            cleaned_description = re.sub(r'<[^>]*>', '', description)
            cleaned_content = re.sub(r'<[^>]*>', '', content)
            cleaned_data.append({
                "title": article.get("title", "Untitled"),
                "description": cleaned_description,
                "content": cleaned_content,
                "published_at": datetime.strptime(article["publishedAt"], "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d %H:%M:%S")
                if article.get("publishedAt") else "Unknown",
                "source": article.get("source", {}).get("name", "Unknown"),
                "url": article.get("url", "No URL")
            })
        except Exception as e:
            print(f"Error cleaning article: {e}")
    return cleaned_data

# --------------------------- NLP MODELS ---------------------------
ner_pipe = pipeline("token-classification", model="dbmdz/bert-large-cased-finetuned-conll03-english")
sentiment_pipe = pipeline("sentiment-analysis", model="ProsusAI/finbert")
qa_pipe = pipeline("question-answering", model="mrm8488/bert-tiny-finetuned-squadv2")

def extract_entities_hf(text: str) -> str:
    entities = ner_pipe(text)
    return "; ".join([f"{entity['word']} ({entity['entity']})" for entity in entities])

def analyze_sentiment_hf(text: str) -> str:
    result = sentiment_pipe(text)
    return result[0]['label'] if result else "unknown"

def extract_relationships_hf(context: str, question: str) -> str:
    response = qa_pipe(question=question, context=context)
    return response.get('answer', '')

# --------------------------- MAIN DATA PROCESSING FUNCTION ---------------------------
def data_process(initial_query):
    print("Fetching news articles...")
    raw_news_data = fetch_news(initial_query)
    result_strings = []
    if raw_news_data:
        cleaned_news = clean_news_data(raw_news_data)
        for article in cleaned_news:
            title = article["title"]
            content = article["content"]
            entities = extract_entities_hf(content)
            sentiment = analyze_sentiment_hf(content)
            relationship = extract_relationships_hf(content, "What is the relationship described in this text?")
            result_strings.append({
                "Title": title,
                "Entities": entities,
                "Sentiment": sentiment,
                "Relationship": relationship,
                "Published At": article["published_at"],
                "Source": article["source"],
                "URL": article["url"]
            })
    else:
        print("Failed to fetch news articles.")
    return result_strings

# --------------------------- LLM AND EMBEDDING MODEL ---------------------------
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, max_tokens=None, max_retries=2, groq_api_key=GROQ_API_KEY)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
tool = BraveSearch.from_api_key(api_key=BRAVE_API_KEY, search_kwargs={"count": 10})
vector_store = None
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

def ingest_data_to_faiss(data, embeddings):
    global vector_store
    docs = [f"""
        Title: {item['Title']}
        Entities: {item['Entities']}
        Sentiment: {item['Sentiment']}
        Relationship: {item['Relationship']}
        Source: {item['Source']}
        URL: {item['URL']}
    """ for item in data]
    split_docs = text_splitter.split_text("\n".join(docs))
    vector_store = FAISS.from_texts(split_docs, embeddings)
    print("Data ingested into FAISS.")

# --------------------------- PROMPT TEMPLATE ---------------------------
template = """
You are a financial expert assistant. Answer the user's question politely and formally.
Let the output be well-structured in only bullet points. and be very brief
Answer only finance-related questions.

Chat History:
{chat_history}

Context:
{context}

Question:
{question}
"""
prompt = PromptTemplate(template=template, input_variables=["context", "question", "chat_history"])

def create_rag_chain(llm, vector_store, tool, prompt):
    retriever = vector_store.as_retriever(search_kwargs={"k": 8})
    return ({"context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)), "question": RunnablePassthrough(), "chat_history": RunnablePassthrough()} | prompt | llm | StrOutputParser())

# --------------------------- NEO4J FUNCTIONS ---------------------------
def create_nodes_and_relationships(news_data: List[Dict]):
    """
    Convert the result string into nodes and relationships to be added to Neo4j.
    :param news_data: List of dictionaries containing the cleaned and processed news data.
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    with driver.session() as session:
        for article in news_data:
            try:
                title = article["Title"]
                entities = article["Entities"].split("; ") if article.get("Entities") else []
                sentiment = article.get("Sentiment", "")
                relationship = article.get("Relationship", "")
                source = article.get("Source", "")

                # Create article node
                session.run(
                    "MERGE (a:Article {title: $title, source: $source})",
                    title=title,
                    source=source
                )

                # Create nodes for each entity and create relationships to the article
                for entity in entities:
                    if " (" in entity:
                        entity_name, entity_type = entity.rsplit(" (", 1)
                        entity_type = entity_type.rstrip(")")
                        session.run(
                            """
                            MERGE (e:Entity {name: $name, type: $type})
                            MERGE (a:Article {title: $title})-[:MENTIONS]->(e)
                            """,
                            name=entity_name.strip(),
                            type=entity_type.strip(),
                            title=title
                        )

                # Create sentiment node and relationship
                if sentiment:
                    session.run(
                        """
                        MERGE (s:Sentiment {sentiment: $sentiment})
                        MERGE (a:Article {title: $title})-[:HAS_SENTIMENT]->(s)
                        """,
                        sentiment=sentiment,
                        title=title
                    )

                # Create relationship node (if any)
                if relationship:
                    session.run(
                        """
                        MERGE (r:Relationship {description: $relationship})
                        MERGE (a:Article {title: $title})-[:DESCRIBES]->(r)
                        """,
                        relationship=relationship,
                        title=title
                    )

            except Exception as e:
                print(f"Error creating nodes or relationships for article '{title}': {e}")

    driver.close()

def generate_html_visualization(cypher_query: str, output_file: str = "graph.html"):
    """
    Generates an HTML file with the graph visualization using vis.js.
    :param cypher_query: Cypher query to fetch graph data.
    :param output_file: Name of the output HTML file.
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    with driver.session() as session:
        try:
            # Fetch graph data from Neo4j
            result = session.run(cypher_query)
            nodes = {}
            edges = []

            for record in result:
                for node in record["p"].nodes:
                    node_id = node.element_id # Use element_id
                    label = node.get("title", node.get("name", str(node.element_id))) # Use element_id
                    sentiment = node.get("sentiment", "")  # Get sentiment
                    if node_id not in nodes:
                        nodes[node_id] = {"id": node_id, "label": f"{label}\n({sentiment})"} # Include sentiment in label

                for relationship in record["p"].relationships:
                    edges.append({
                        "from": relationship.start_node.element_id, # Use element_id
                        "to": relationship.end_node.element_id, # Use element_id
                        "label": relationship.type
                    })

            nodes_list = list(nodes.values())

            # Generate HTML with vis.js
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Graph Visualization</title>
                <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"></script>
                <style>
                    #network {{
                        width: 100%;
                        height: 600px;
                        border: 1px solid lightgray;
                    }}
                </style>
            </head>
            <body>
                <div id="network"></div>
                <script>
                    var nodes = new vis.DataSet({json.dumps(nodes_list)});
                    var edges = new vis.DataSet({json.dumps(edges)});

                    var container = document.getElementById("network");
                    var data = {{ nodes: nodes, edges: edges }};
                    var options = {{
                        nodes: {{
                            shape: "dot",
                            size: 20,
                            font: {{ size: 14 }}
                        }},
                        edges: {{
                            arrows: "to",
                            smooth: true
                        }},
                        physics: {{
                            stabilization: false
                        }}
                    }};

                    var network = new vis.Network(container, data, options);
                </script>
            </body>
            </html>
            """

            # Write the HTML content to a file
            with open(output_file, "w") as file:
                file.write(html_content)

            print(f"Graph visualization saved to {output_file}. Open it in a browser to view.")

        except Exception as e:
            print(f"Error generating HTML visualization: {e}")

    driver.close()

def process_and_visualize_data(news_data):
    """
    Process the raw data, populate Neo4j, and visualize the graph.
    :param news_data: Raw cleaned data from the news API.
    """
    # Step 1: Convert the data to Neo4j compatible format (nodes & relationships)
    create_nodes_and_relationships(news_data)

    # Step 2: Visualize the graph by running Cypher queries
    cypher_query_sentiment = "MATCH p=()-[r:HAS_SENTIMENT]->() RETURN p LIMIT 25;"
    cypher_query_relationship = "MATCH p=()-[r:DESCRIBES]->() RETURN p LIMIT 25;"

    print("Generating HAS_SENTIMENT graph visualization...")
    generate_html_visualization(cypher_query_sentiment, "sentiment_graph.html")
    
    print("Generating DESCRIBES graph visualization...")
    generate_html_visualization(cypher_query_relationship, "relationship_graph.html")
    
    # Read both graph files
    with open("sentiment_graph.html", "r") as file:
        sentiment_html = file.read()
        
    with open("relationship_graph.html", "r") as file:
        relationship_html = file.read()
        
    return sentiment_html, relationship_html

# --------------------------- FLASK APP ROUTES ---------------------------
chat_history = []
data = None
rag_chain = None

@app.route('/', methods=['GET'])
def index():
    return "Welcome to the Fin-Talk"

@app.route('/initial_query', methods=['POST'])
def initial_query():
    global data, rag_chain
    req_data = request.get_json()
    query = req_data.get("query")
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    data = data_process(query)
    if data:
        ingest_data_to_faiss(data, embeddings)
        rag_chain = create_rag_chain(llm, vector_store, None, prompt)
        return jsonify({"message": "Data processed and ingested successfully"}), 200
    else:
        return jsonify({"error": "Failed to process data"}), 500

@app.route('/conversational_query', methods=['POST'])
def conversational_query():
    req_data = request.get_json()
    query = req_data.get("query")
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    if not rag_chain:
        return jsonify({"error": "No initial query data found. Run initial_query first."}), 400
    
    response = rag_chain.invoke(query, chat_history="\n".join(chat_history))
    chat_history.append(f"User: {query}\nAssistant: {response}")
    
    return jsonify({"response": response}), 200

@app.route('/chat_history', methods=['GET'])
def get_chat_history():
    return jsonify({"chat_history": chat_history}), 200

@app.route('/show_graph', methods=['GET'])
def show_graph():
    if not data:
        return jsonify({"error": "No graph data available. Run initial_query first."}), 400
    
    sentiment_html, relationship_html = process_and_visualize_data(data)
    
    return jsonify({"sentiment_html": sentiment_html, "relationship_html": relationship_html}), 200

# Serve static files (including graph HTML)
@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('client', path)

# --------------------------- MAIN EXECUTION ---------------------------
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)