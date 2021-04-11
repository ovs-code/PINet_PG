export FLASK_APP=main.py
source venv/bin/activate
redis-server &
python worker.py &
flask run --port=8080 --host=0.0.0.0
trap 'kill $(jobs -p)' EXIT
