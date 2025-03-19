# AAK-WebMindRAG
WebMind 🧠 – Smart HTML Chunking for RAG Built for Confluence page processing, WebMind handles Confluence pages, parsing multiple pages at once while preserving structure. It creates adaptive chunks, updates Faiss DB in batches, and offers an interactive Streamlit UI for exploration. 🚀

TO Run this APP :
1- You need to generate a API key from Your Confluence Account and pass it in the backed_service class. 
2- You need to Either Run it in ec2 Instance and have access rights to call the bedrock agent
3- Or if running in Local , you ned to have Ollama installed and qwen2.5:7b Downloaded in your local machine
4- Install all required libraries in for streamlit app,langchain_aws,langchain_core,faiss,sentencetransformers,ollama,boto3 (if running in ec2),bedrock.
5- pass the PageId as the input , this pageId should be of a parent confluence page which has multiple child pages, so that when app readds the page, it automatically reads all the child pages of the parent page and creates the chunks for all of them and Insert into Vector DB.  
