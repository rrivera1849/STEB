# Dataset Download Instructions

This document provides instructions for downloading datasets that require special access or subscriptions.

# 1. Fisher Speech Transcript Dataset

## Overview
The Fisher Speech Transcript Dataset is used for speaker attribution tasks. It consists of conversational speech transcripts from the Fisher English Training Speech Transcript corpus. Note that speech transcripts are a different modality (spoken language) from the other STEB datasets, which are written language. However, Aggazzotti et al. (2024) found authorship models to perform fairly well out-of-the-box. Another advantage of this dataset beyond testing model generalizability to another modality is the levels of topic control (base/hard/harder) to help encourage reliance on more invariant speaker style features.

## Requirements
- **LDC Subscription**: You need a subscription to the Linguistic Data Consortium (LDC) to access these datasets
- **Two datasets required**:
  - LDC2004T19: [Fisher English Training Speech Part 1 Transcripts](https://catalog.ldc.upenn.edu/LDC2004T19)
  - LDC2005T19: [Fisher English Training Part 2 Transcripts](https://catalog.ldc.upenn.edu/LDC2005T19)

## Installation Steps

### Step 1: Obtain LDC Subscription
1. Visit the [LDC website](https://www.ldc.upenn.edu/)
2. Create an account or log in
3. Subscribe to or purchase access to:
   - LDC2004T19 (Fisher English Training Speech Part 1 Transcripts)
   - LDC2005T19 (Fisher English Training Part 2 Transcripts)

### Step 2: Download the Datasets
1. Log into your LDC account
2. Navigate to your subscribed datasets
3. Download both LDC2004T19 and LDC2005T19
4. Extract the downloaded archives to accessible directories

### Step 3: Configure Paths
The Fisher dataset requires configuration in the speech-attribution project. For detailed setup instructions, refer to the [speech-attribution GitHub](https://github.com/caggazzotti/speech-attribution).

**Key configuration points:**
- Set `fisher_dir1` to the directory containing LDC2004T19 data
- Set `fisher_dir2` to the directory containing LDC2005T19 data
- Set `work_dir` to where trial datasets and results will be stored (STEB/raw_datasets/)

Example structure:
```
fisher_dir1: /path/to/LDC2004T19
fisher_dir2: /path/to/LDC2005T19
work_dir: ./raw_datasets/fisher_speaker_attribution
```

### Step 4: Process the Dataset

1. **Split datasets**: Split the corpus by speaker into train/val/test
2. **Create trials**: Generate positive and negative trial pairs
3. **Add transcripts to trials**: Retrieve and format transcripts for trials

Result: 6 files (3 for the BBN transcription, 3 for the LDC transcription; 1 for each difficulty level)
- bbn_test_base_trials.npy
- bbn_test_hard_trials.npy
- bbn_test_harder_trials.npy
- ldc_test_base_trials.npy
- ldc_test_hard_trials.npy
- ldc_test_harder_trials.npy

### Step 5: Integrate with STEB
Now the trials are ready for STEB and can be used to evaluate your models following the directions on the [STEB GitHub](https://github.com/rrivera1849/STEB/). Manually add them to a folder under raw_datasets (e.g., fisher_speaker_attribution/) or point your config data_dir to where they live.


---



---

## Troubleshooting

### Fisher Dataset
- **LDC access issues**: Contact LDC support or verify your subscription status
- **Path configuration errors**: Ensure absolute paths are used in config files
- **Processing errors**: Refer to the speech-attribution README for detailed troubleshooting

---



