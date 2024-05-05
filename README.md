### 📚 Sectral (Mistral-7B_instruct.v0.1) 🚀

Get the exact Kaggle Notebook to run here: [Sectral Kaggle Notebook](https://www.kaggle.com/code/anshsarkar18/sectral-mistral-7b-based-rag)

**Retrieval augmented generation (RAG)** is a natural language processing (NLP) technique that combines the strengths of both retrieval- and generative-based artificial intelligence (AI) models. Based on the task assigned, the easiest and best suited way to extract insightful data without huge amounts of data analysis, in a short period of time, while being resistant to events as, changes in data formats, layouts, etc, was to implement a RAG based on a relatively lightweight model (`Mistral 7B`  in our case).

The following notebook contains the code blocks required for the functioning of the CLI program that initializes a RAG at the backend and provides a chat based interactive interface to the user. The user has the ability to **load** data corresponding to a particular stock and then ask **questions** related to it. The user also has the ability to create and visualize important financial metrics via **plots** and get **insights** from the model itself.

Even after my best efforts to create the system completely locally so that it could be easily deployed to a server without the requirement of a GPU, it turned out that even after quantization of the model (reduction in weight precisions), the model still required a GPU to function. Hence the decision was later taken to shift to an interactive kaggle notebook and provide a chat interface that is as intuitive and easy to use as possible, while making sure that the required results were being generated.

**Dependency Installation**: The following code block consists of the various libraries and frameworks required for the SectralCLI UI and Interface to function correctly.

### Required Configuration
Load a model without downloading it - We leverage the Mistral model provided by Kaggle. In my case, my current setup, does not include a GPU which made it almost impossible even to train a quantized version of the Mistral 7B model. Hence, the required settings for this Kaggle Notebook to function correctly are,

- Internet: `On` (Toggle)
- Accelerator: `GPU P100`
- Model: mistral 7b-instruct-v0.1-hf/1 (by kaggle)
- Persistence: Files Only

### Components
The entire notebook is divided into the following major sections:
- **SecConfig** : Contains the configurations and other parameters required by the various components
  - Internally uses the [edgartools](https://github.com/dgunning/edgartools) library to gather facts and information in the form of pandas dataframes
- **SecPlot**: Contains functions required to plot graphs and financial data related to a particular stock
  - Internally uses the `matplotlib` and `pandas` libraries to create, plot and store the figures / graphs locally for reference
- **SecDB**: Contains functions required to create / load the vector database indexes depending on availability
  - Internally uses `FAISS` (Facebook AI Similarity Search) and stores the vector embeddings in the `HuggingFaceEmbeddings` format.
- **SecCLI**: Responsible to bring all the components together. Initializes the model, loads data and manages the interactive session
  - Internally uses all the previously listed components: `SecDB`, `SecPlot` & `SecConfig` to create an interactive user session.

**Code Formatting and Structure**: All the components strictly follow an Object Oriented Programming (OOP) approach wherein every member function and associated parameters are neatly encapsulated within classes. The files have also been formatted using `black` (python formatter) to follow the `PEP 8` standard.