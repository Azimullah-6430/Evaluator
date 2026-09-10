import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from app.evaluation.evaluator import EvaluationAgent

load_dotenv()

app = Flask(__name__)
