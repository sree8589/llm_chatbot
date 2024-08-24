from setuptools import setup, find_packages

setup(
    name="llm_chatbot",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "ctransformers==0.2.5",
        "sentence-transformers==2.2.2",
        "pinecone-client",
        "langchain==0.0.225",
        "flask",
        "pypdf",
        "python-dotenv",
    ],
)

