# Self-Supervised Autoencoder Network for Robust Heart Rate Extraction from Noisy Photoplethysmogram

**Paper Links:** [arXiv](https://arxiv.org/abs/2504.09132) [[pdf](https://arxiv.org/pdf/2504.09132)]

## Abstract
Biosignals can be viewed as mixtures measuring particular physiological events, and blind source separation (BSS) aims to extract underlying source signals from mixtures. This paper proposes a self-supervised multi-encoder autoencoder (MEAE) to separate heartbeat-related source signals from photoplethysmogram (PPG), enhancing heart rate (HR) detection in noisy PPG data. The MEAE is trained on PPG signals from a large open polysomnography database without any pre-processing or data selection. The trained network is then applied to a noisy PPG dataset collected during the daily activities of nine subjects. The extracted heartbeat-related source signal significantly improves HR detection as compared to the original PPG. The absence of pre-processing and the self-supervised nature of the proposed method, combined with its strong performance, highlight the potential of BSS in biosignal analysis.

<p align="center">
    <img src="assets/figure_1.png" alt="drawing" width="100%" height="100%"/>
  <p align="center">
      Figure 1. Diagram outlining the architecture of the proposed multi-encoder autoencoder used in this study to extract heart-related source signal from photoplethysmogram.
  </p>
</p>
<p align="center">
    <img src="assets/figure_2.png" alt="drawing" width="55%" height="55%"/>
    <p align="center">
      Figure 2. Diagram showing the inference step for the proposed method. An encoder is selected, and the output from all other encoders are masked with zeros. Then the encodings are passed to the decoder, yielding a source prediction corresponding to the selected encoder. $\hat{S}_n$ is the nth predicted source corresponding to the nth encoder, $E_n$.
  </p>
</p>

# Experiments
This repository is heavily based upon prior work that can be found [here](https://github.com/webstah/self-supervised-bss-via-multi-encoder-ae).

You can request access to the Multi-Ethnic Study of Atherosclerosis (MESA) Sleep study[^1][^2] data [here](https://sleepdata.org/datasets/mesa). After downloading the dataset, set the `DATA_PATH` variable and run: `python generate_mesa_data.py`

To train a model with our configuration use the following command: `python trainer.py experiment_config=mesa_ppg_bss`

After training, manual inspection is required to determine which encoder produces the desired cardiovascular-related source.

The noisy PPG dataset used in this study is not publicly available due to the absence of permission from the study participants and approval from the relevant Institutional Review Board (IRB). 

# Results
| Method      | None             | ICA              | NMF              | BRDAE            | MEAE (Ours)                |
|-------------|------------------|------------------|------------------|------------------|--------------------|
| $\uparrow$ Correlation | $0.407 \pm 0.177$ | $0.602 \pm 0.214$ | $0.576 \pm 0.181$ | $0.686 \pm 0.147$ | $\textbf{0.740} \pm 0.153$ |
| $\downarrow$ RMSE (BPM)  | $14.4 \pm 10.6$   | $8.6 \pm 10.4$    | $8.2 \pm 6.0$     | $6.1 \pm 8.2$     | $\textbf{4.9} \pm 5.1$     |

Table 1: Comparison of heart rates from various methods against reference ECG HR. Bold text indicates best result for a given metric.

<p align="center">
    <img src="assets/figure_3.png" alt="drawing" width="55%" height="55%"/>
    <p align="center">
      Figure 3. Detection of heart rate from electrocardiogram (top), original photoplethysmogram (middle), and the first source signal (bottom) generated from the optimized multi-encoder autoencoder network. The detected peaks corresponding to each signals are used to generate beat-by-beat heart rates for each cardiac cycle.
  </p>
</p>
<p align="center">
    <img src="assets/figure_5a.png" alt="drawing" width="45%" height="45%"/><img src="assets/figure_5b.png" alt="drawing" width="45%" height="45%"/>
    <p align="center">
      Figure 4. Bland-Altman plots comparing heart rate detection from the original photoplethysmogram (left) and the selected source signal (right) against the reference electrocardiogram..
  </p>
</p>

## Model and config download
The MEAE model used in our experiments can be found [here](https://drive.google.com/file/d/1z4gHzVAsF3FKio2O0_7h1NTSYMe7f4EO/view?usp=sharing) on Google Drive.

Extract the model and config file into a `./checkpoints` folder within the repository. Go to the example usage notebook here: `notebooks/test_model.ipynb`


### References
[^1]: Zhang GQ, Cui L, Mueller R, Tao S, Kim M, Rueschman M, Mariani S, Mobley D, Redline S. The National Sleep Research Resource: towards a sleep data commons. J Am Med Inform Assoc. 2018 Oct 1;25(10):1351-1358. doi: 10.1093/jamia/ocy064. PMID: 29860441; PMCID: PMC6188513.
[^2]: Chen X, Wang R, Zee P, Lutsey PL, Javaheri S, Alcántara C, Jackson CL, Williams MA, Redline S. Racial/Ethnic Differences in Sleep Disturbances: The Multi-Ethnic Study of Atherosclerosis (MESA). Sleep. 2015 Jun 1;38(6):877-88. doi: 10.5665/sleep.4732. PMID: 25409106; PMCID: PMC4434554.