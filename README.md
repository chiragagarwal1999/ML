# ML
Instructions for RAG (Retreival Augmented Generation):
1. Download a LLM supported by GPT4ALL and place it inside the models folder.
2. Rename example.env to .env and mention the model name inside it as the 'MODEL_PATH' parameter.
3. Install the requirements
4. If you have a GPU, uncomment the gpu config lines in privateGPT.py
5. Run the following command in terminal : streamlit run privateGPT.py

Instructions for Analytics using PANDAS
1. Rename example.env to .env and mention your openai api key in the file.
2. This uses openai model currently as currently could not find any opensource model that could write pandas query precisely correct.
3. Run the following command : python analyticsGPT.py
