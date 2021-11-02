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
   conda install numpy matplotlib jupyter jupyterlab ipdb tqdm natsort
   conda install nltk
   conda install scikit-learn
   ```
<!-- 4. Download data
   ```bash
   cd lab1/
   mkdir data/
   cd data/
   wget https://gist.githubusercontent.com/bastings/d47423301cca214e3930061a5a75e177/raw/5113687382919e22b1f09ce71a8fecd1687a5760/reviews.json
   ``` -->