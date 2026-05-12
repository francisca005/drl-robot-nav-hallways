# rl-webots

git clone https://github.com/francisca005/drl-robot-nav-hallways.git
cd drl-robot-nav-hallways

python -m venv venv
.\venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install tensorboard

mkdir models -ErrorAction SilentlyContinue
mkdir logs -ErrorAction SilentlyContinue

num terminal
webots .\worlds\smart-wheelchairs.wbt

parar a simulação


nourto terminal
cd drl-robot-nav-hallways
.\venv\Scripts\activate
python .\src\rl-server.py --new

começas a simulação no webots


para ter as training curves da expriência fazer

python .\src\plot_training_curves.py --experiment e2_features
ou 
python .\src\plot_training_curves.py --experiment e1_cnn

sempre que se correr a um novo teste, guardar a pasta positions anterior e apagar. No fim do teste guardar a nova pasta position com o nome da expriência para não perder os caminhos