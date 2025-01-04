# EnviroSummarization
This project uses the BART model to summarize environmental content, extracting key insights on climate change, biodiversity, and pollution. Fine-tuned for summarization, BART's robust text generation capabilities provide concise, coherent summaries, aiding researchers and policymakers in understanding critical issues.

This project employs the BART model for summarizing environmental content, focusing on topics such as climate change, biodiversity, and pollution. By fine-tuning BART on synthetic data, we enable concise and coherent summaries, aiding researchers, policymakers, and decision-makers.

## Technologies Used  

- **Transformers**: Utilized for the implementation and fine-tuning of the BART model.  
- **Streamlit**: Provides an interactive interface for summarizing environmental texts.  
- **NLP (Natural Language Processing)**: Applied to process and analyze large volumes of textual data.  
- **Pandas**: Handles data manipulation and preparation tasks.  
- **Hugging Face**: Powers the transformer-based model fine-tuning and inference.  
- **JSON Dataset**: A pre-generated dataset in JSON format is provided for training and testing.  

## Model: BART  
BART is a transformer-based sequence-to-sequence model combining a bidirectional (BERT-like) encoder and an autoregressive (GPT-like) decoder. Pre-trained on a text reconstruction task, BART is highly effective for text summarization and generation. Fine-tuning it on domain-specific data ensures accurate, topic-relevant summaries.

## Dataset  
The dataset contains columns for topics, articles, and summaries. Summaries were synthetically generated using the Gemini API, ensuring high-quality data aligned with environmental literature. No separate evaluation dataset was used; the model shows good results when evaluated on the training dataset.

## Key Features  
- Extracts concise summaries from environmental literature.  
- Fine-tuned for domain-specific topics like climate change, pollution, and biodiversity.  
- Shows good results, supporting analysis and decision-making.

## Usage  
This project highlights the utility of transformer-based models like BART for summarizing large-scale text datasets in the environmental domain.


## Quick Start

1. **Run the Notebook and Save the Model:**

   - Open the provided Jupyter Notebook file (`train_model.ipynb`) in your preferred environment (e.g., Jupyter Notebook, JupyterLab, Google Colab).  
   - Execute the notebook cells to fine-tune the BART model on the provided dataset.  
   - Once training is complete, save the fine-tuned model locally or in a designated output directory (e.g., `./environmental_summarizer_final`).  

2. **Set Up Your Environment:**

   **Windows:**
   - Open your terminal and navigate to your project directory:  
     ```bash
     cd path/to/your/project
     ```
   - Create a virtual environment named `venv`:  
     ```bash
     python -m venv venv
     ```
   - Activate the virtual environment:  
     ```bash
     .\venv\Scripts\activate
     ```

   **Unix/macOS:**
   - Open your terminal and navigate to your project directory:  
     ```bash
     cd path/to/your/project
     ```
   - Create a virtual environment named `venv`:  
     ```bash
     python3 -m venv venv
     ```
   - Activate the virtual environment:  
     ```bash
     source venv/bin/activate
     ```

3. **Install Dependencies:**

   With your virtual environment activated, install the required dependencies by running:  
   ```bash
   pip install pdfplumber
   pip install streamlit
   pip install transformers
   pip install torch
   ```

4. **Run Code:**
  ```bash
  streamlit run app.py
  ```
   

