# 🛒 Multimodal E-Commerce Product Categorization using a Hybrid CNN-BiLSTM Network

## 📌 Project Overview
This project implements a **Multimodal (Many-to-One) Deep Learning System** designed to automatically classify e-commerce products into their correct primary categories (`masterCategory`)[cite: 1, 2]. Instead of relying on a single data source, the system simultaneously analyzes two completely different data types: product images and textual descriptions[cite: 1, 2].

The implementation details, workflow structure, and analytical evaluations are fully documented in the source file **"hibrit_ysa_model_gelistirme.ipynb"**[cite: 1, 2], following the explicit criteria outlined in the **"Hibrit_YSA_Proje_Odevi_Istek_Raporu.docx"**[cite: 1].

### 💡 Why a Hybrid Approach?
* **Limitations of Single Modality:** A pure image-based model (CNN) captures shapes and colors but misses fine-grained technical details (e.g., distinguishing between visually identical 16GB and 64GB flash drives)[cite: 1, 2]. Conversely, a pure text model (RNN/LSTM) cannot perceive the style, pattern, or texture of a product, which is often the most distinguishing feature in categories like clothing or furniture[cite: 1, 2].
* **The Hybrid Advantage:** This architecture builds a cognitive bridge between pixels and text strings[cite: 1, 2]. By processing each input via specialized branches and fusing their knowledge, the network overcomes the natural performance ceilings of isolated single-stream models[cite: 1, 2].

## 📊 Dataset & Scope
* **Source:** Fashion Product Images Dataset (Small subset via Kaggle)[cite: 2].
* **Inputs:** Product Images (JPEG matrices) & Product Display Names (Text strings)[cite: 1, 2].
* **Target:** `masterCategory` (Multi-class target featuring categories such as *Apparel, Accessories, Footwear, and Personal Care*)[cite: 1, 2].
* **Data Handling:** Implements a stratified subset of 10,000 samples to maintain the authentic class distribution while ensuring hardware efficiency during optimization on a standard T4 GPU environment[cite: 2].

## 🧠 Model Architecture
The architecture consists of a dual-branch neural network designed to capture heterogeneous features[cite: 1, 2]:

1. **📷 Visual Branch (CNN):** Processes product images to extract high-level spatial features including edges, boundaries, textures, and color structures[cite: 1, 2].
2. **📝 Textual Branch (BiLSTM):** Processes tokenized titles and short descriptions, utilizing bidirectional memory cells to capture contextual semantic relationships from both past and future words in the sequence[cite: 1, 2].
3. **🔀 Feature Fusion Layer:** The dense feature vectors outputted by both branches are combined using **Feature Concatenation**[cite: 1, 2]. 
4. **🎯 Classification Head:** The fused multimodal embedding is routed into shared fully connected (Dense) layers with dropout regularization, culminating in a Softmax output layer that computes final class probabilities[cite: 1, 2].

## 📈 Performance Evaluation & Handling Class Imbalance
The dataset reflects a realistic e-commerce landscape, characterized by heavy class imbalance (e.g., the *Apparel* class vastly dominating over minority classes)[cite: 1, 2]. To tackle this bias:
- **Metrics:** Classification Accuracy alone is treated as insufficient; instead, **Macro F1-Score** is used as the reliable success benchmark[cite: 1, 2].
- **Error Analysis:** A comprehensive **Confusion Matrix** is analyzed at the end of the evaluation phase to identify precisely which category pairs the model confuses, allowing for targeted architectural and feature refinements[cite: 1, 2].
