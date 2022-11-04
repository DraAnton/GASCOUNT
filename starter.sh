#!/bin/bash
uvicorn app:app --reload --host $(hostname -I | cut -d' ' -f 1)