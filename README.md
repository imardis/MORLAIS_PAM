# Marine Species Identification Using Passive Acoustic Monitoring

## Introduction

The application of machine learning classifiers to data from passive acoustic monitoring devices focuses on identifying marine species based on their unique sounds, known as ecological clicks. The work emphasizes the separation of species, particularly bottlenose dolphins and Risso's dolphins, in alignment with the MORLAIS project.

## Methodology

### Phase 1: Identification of Clicks

- **Input:** Audio files from hydrophones, saved as WAV files.
- The extraction of ecological clicks from these files, accomplished using PAMGuard software and signal processing, involves storing the information in binary files (PGDF extension).

### Phase 2: Labeling the Clicks

- **Goal:** The creation of a labelled dataset for supervised learning models.
- Using PAMGuard software, experts in marine biology check and label clicks within PGDF files based on waveform and frequency. This labelled data is then saved in a SQLite file.

#### Note:

Skipping Phase 2 is possible when using a pre-trained  classifier.

### Phase 3: Data Preparation

- A well-constructed dataset for machine learning is designed by combining PGDF binary files with SQLite files containing labels.
- The selection of desired labels and the retention of only relevant input and output data is performed. The organized dataset is then placed in a Data folder for simplicity.

## Machine Learning Model

The research explores the ability of machine learning models to identify marine species using raw signal waveforms.

- Investigation of two deep neural networks: recurrent networks and networks based on convolution filters.
- The input to the models includes both raw waveforms and spectrograms.
- The models are trained on AWS servers, resulting in optimal settings considering a large number of parameters and hyperparameters.

#### Note:

For new species or datasets, the training process must be repeated for optimal results.

## Folders

- **Data Folder:** Contains the data file extracted for the machine learning model.
- **Model Folder:** Contains codes related to the machine learning model.
- **Notebook Folder:** Contains corresponding Jupyter notebook files.

## Performance Test

To assess the performance of the Jupyter Notebook file's model, see the **Testing Performance.ipynb**.

## Future Development

Considering the good performance of the model, it is suggested to explore further development opportunities.
