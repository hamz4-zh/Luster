# Attribute-Controlled Jewelry Image Synthesis via LoRA

This repository contains the code, evaluation scripts, and dataset generation pipeline for a custom Stable Diffusion XL model fine-tuned using LoRA to generate highly accurate jewelry images..

## Performance
Our custom domain-adapted model achieved a 97.5% attribute accuracy score, outperforming DALL-E 3 (92.5%) and Base SDXL 1.0 (85.0%).. It was particularly effective at maintaining perfect geometric cut accuracy for gemstones across all test prompts..

## Dataset Taxonomy
The training dataset consists of 1,267 images generated programmatically.. The attribute taxonomy covers 54 distinct classes across four axes to avoid concept bleeding.:
* **Gemstone:** Diamond, Emerald, Sapphire.
* **Cut Shape:** Round, Princess, Cushion.
* **Metal:** Yellow Gold, White Gold, Rose Gold.
* **Setting:** Solitaire, Pavé.

## Training Details
The LoRA fine-tuning was completed locally in approximately 1.4 hours using an NVIDIA RTX 5070 Ti.. 
* **Base Model:** stabilityai/stable-diffusion-xl-base-1.0.
* **LoRA Rank:** 4.
* **Learning Rate:** 1e-4 (Constant).
* **Mixed Precision:** fp16.

*Note: Checkpoint 1000 was selected as the optimal model, as further training to 1500 steps showed signs of data autophagy and detail degradation*..

## Author

Hamza Alzahrani

This was an individual project completed as part of a team coursework arrangement, where each member led a separate project for a shared course submission. Thanks to my teammates, Turki Alshuaibi, Khalid Alomair, Ahmad Alakhdhar, Aseel Alsaid, and Anas Alghamdi, for their support during that semester.
