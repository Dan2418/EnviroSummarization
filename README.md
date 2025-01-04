# EnviroSummarization
This project uses the BART model to summarize environmental content, extracting key insights on climate change, biodiversity, and pollution. Fine-tuned for summarization, BART's robust text generation capabilities provide concise, coherent summaries, aiding researchers and policymakers in understanding critical issues.

This project employs the BART model for summarizing environmental content, focusing on topics such as climate change, biodiversity, and pollution. By fine-tuning BART on synthetic data, we enable concise and coherent summaries, aiding researchers, policymakers, and decision-makers.

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

