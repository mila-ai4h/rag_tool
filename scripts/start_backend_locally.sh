#!/bin/bash

cd ../
source .env
uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload
