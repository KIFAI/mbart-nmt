nohup python -m uvicorn trans_api:app --host=0.0.0.0 --port 5000 --workers 4 > nohup_app.out &
