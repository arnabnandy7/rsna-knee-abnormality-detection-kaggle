# Scripts

Data preparation, label extraction, DICOM I/O benchmarks, and weight download utilities.

## Files

* **`extract_labels_llm.py`**: Extracts 12 continuous finding probabilities from unstructured radiology report text using a local LLM via Ollama API.
* **`extract_labels_v2.py`**: High-throughput rule-based radiology report parser and regular expression label extractor.
* **`download_dinov2.py`**: Utility to download pretrained DINOv2 weights for cross-series feature extraction.
* **`_fastread.py`**: Standalone benchmark engine evaluating single-pass DICOM headers and memory-mapped file descriptor reuse (`keepfd`, `onepass`, `dedupe`).
