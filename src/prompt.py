prompt_template = """
Use the following pieces of information to answer the user's question.
If the context does not provide enough information to answer the question, respond with: "Sorry, I don't have enough information to answer that."

Context: {context}
Question: {question}

Provide the answer below, based on the context provided. If the context doesn't sufficiently address the question, use the fallback response.

Answer:
"""
