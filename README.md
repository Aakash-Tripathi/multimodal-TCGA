# multimodal-TCGA

Things to showcase:

1. Multimodal is better than uni-modal
2. Fine-tuning a pre-trained model is better than training from scratch
3. Better embeddings yield better results

**TASK:** Predicting survival of cancer patients using multimodal data from TCGA

The Cancer Genome Atlas (TCGA) is a landmark cancer genomics program that sequenced and molecularly characterized over 11,000 cases of primary cancer samples. The goal of TCGA is to improve the ability to diagnose, treat and prevent cancer through an understanding of the molecular basis of the disease. TCGA data is organized into six data types: clinical, biospecimen, genomic, epigenomic, transcriptomic, and proteomic. The data is available through the [Genomic Data Commons (GDC) Data Portal](https://portal.gdc.cancer.gov/) and [Imaging Data Commons (IDC) Data Portal](https://portal.imaging.datacommons.cancer.gov/).

In this project, we will use multimodal data from TCGA to predict survival of cancer patients. We will use the following data types:

1. Clinical data
2. Pathology whole slide images (WSI)
3. Radiology images

We use pretrained foundational models to generate embeddings for each modality. To evaluate the performance of the embeddings generated, we use the embeddings to train a simple neural network model for some classification/regression task. The best performing model is then used as the embedding generator for the multimodal transformer model.

## Embedding Models

### Clinical data

To generate the embeddings for the clinical data, we study the performance of the following foundational models that are trained on the clinical data:

1. [GatorTron](placeholder)
2. [ClinicalT5](placeholder)

### Pathology WSI and Radiology images

To generate the embeddings for the pathology WSI and radiology images, we use the [REMEDIS](placeholder) model.