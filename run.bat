@echo off

docker-compose up -d
python run_gui.py
docker-compose down
