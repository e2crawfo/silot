# Install dps
git clone https://github.com/e2crawfo/dps.git
cd dps
git checkout aaai_2020
pip install -r requirements.txt
pip install -e .

# Install auto_yolo
cd ../
git clone https://github.com/e2crawfo/auto_yolo.git
cd auto_yolo
git checkout aaai_2020
pip install -r requirements.txt
pip install -e .

# Optional: install SQAIR.
cd ../
git clone https://github.com/e2crawfo/sqair.git
cd sqair
git checkout aaai_2020
pip install -r requirements.txt
pip install -e .

# Install silot
cd  ../
pip install -r requirements.txt
pip install -e .
