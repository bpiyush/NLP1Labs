1. Create a `conda` environment
   ```bash
   conda create -n nlp1labs
   conda activate nlp1labs
   ```
2. Install `python3.9`
   ```bash
   conda install python=3.9 -y
   ```
3. Install relevant packages
   ```bash
   conda install --file env.yaml
   conda install numpy matplotlib jupyter jupyterlab ipdb tqdm natsort pandas
   conda install nltk
   conda install scikit-learn
   ```
