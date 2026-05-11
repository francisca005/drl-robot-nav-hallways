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