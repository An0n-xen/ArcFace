# ArcFace Training in PyTorch: Deep Face Recognition

> "Face recognition is not just about classification; it's about discrimination."

### Paper
[ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)

### Reference
[InsightFace Repository](https://github.com/deepinsight/insightface)

---

## 📐 Project Overview
This repository contains a PyTorch implementation of **ArcFace**, the state-of-the-art method for face recognition.

While traditional Softmax loss separates classes, it does not force features to be compact. ArcFace introduces an **Additive Angular Margin** to the loss function, effectively pushing intra-class samples closer together and inter-class samples further apart on a hypersphere.

## 🧠 The "Secret Sauce": Geodesic Distance
The core challenge in face recognition is **Open-Set Recognition** (identifying faces the model has never seen before).

To solve this, this implementation enforces a geometric constraint:
1.  **Normalize:** Both features and weights are normalized to lie on a hypersphere.
2.  **Margin:** An angular margin ($m$) is added to the angle between the feature and the ground truth weight.
3.  **Result:** The model learns highly discriminative features that generalize to unseen faces.

## 🛠 Key Implementations
* **ArcFace Loss Module:** Custom PyTorch `nn.Module` implementing the additive angular margin penalty (including $\cos(\theta + m)$ logic).
* **Backbone:** Utilized ResNet (or IR-SE) architectures modified for feature extraction (112x112 input).
* **Training Loop:** Implemented data loading for face datasets (e.g., CASIA-WebFace / VGG-Face2) and optimization steps.

## 📊 Visualizing the Hypersphere
*The image below demonstrates how ArcFace clusters embeddings more tightly than Softmax.*

![Embedding Visualization](/assets/embedding_clusters.png)
*(Note: Generate a 2D/3D scatter plot of your embeddings here)*
