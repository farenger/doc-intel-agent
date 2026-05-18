#!/usr/bin/env python
# coding: utf-8

# ============================================================
# DOC INTEL AGENT - Flask API
# ============================================================

import os
import json
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from doc_intel_agent import agent

# ============================================================
# Flask App Setup
# ============================================================

app = Flask(__name__)

# ============================================================
# Config
# ============================================================

UPLOAD_FOLDER = r"C:\Users\manas\OneDrive\Documents\Python Agent Project Folder"

ALLOWED_EXTENSIONS = {"pdf", "txt"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ============================================================
# Helper Functions
# ============================================================

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )

# ============================================================
# Health Check Route
# ============================================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "running"
    })

# ============================================================
# Document Processing Route
# ============================================================

@app.route("/process-document", methods=["POST"])
def process_document():

    try:

        # ----------------------------------------------------
        # Check if file exists in request
        # ----------------------------------------------------

        if "file" not in request.files:
            return jsonify({
                "error": "No file uploaded"
            }), 400

        file = request.files["file"]

        # ----------------------------------------------------
        # Validate filename
        # ----------------------------------------------------

        if file.filename == "":
            return jsonify({
                "error": "No file selected"
            }), 400

        # ----------------------------------------------------
        # Validate extension
        # ----------------------------------------------------

        if not allowed_file(file.filename):
            return jsonify({
                "error": "Only PDF and TXT files allowed"
            }), 400

        # ----------------------------------------------------
        # Save uploaded file
        # ----------------------------------------------------

        filename = secure_filename(file.filename)

        file_path = os.path.join(
            UPLOAD_FOLDER,
            filename
        )

        file.save(file_path)

        print(f"\n[INFO] File saved to: {file_path}")

        # ----------------------------------------------------
        # Invoke Agent
        # ----------------------------------------------------

        print("[INFO] Starting agent...")

        response = agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": f"Process this document: {file_path}"
                }
            ]
        })

        # ----------------------------------------------------
        # Debug Messages
        # ----------------------------------------------------

        print("\n========== AGENT RESPONSE ==========\n")

        for msg in response["messages"]:

            print(f"TYPE: {msg.type}")
            print(f"CONTENT:\n{msg.content}")
            print("-----------------------------------")

        # ----------------------------------------------------
        # Extract JSON from Tool Messages
        # ----------------------------------------------------

        result_json = None

        for msg in reversed(response["messages"]):

            if msg.type == "tool":

                try:
                    result_json = json.loads(msg.content)
                    break

                except json.JSONDecodeError:
                    continue

        # ----------------------------------------------------
        # Return Tool JSON if found
        # ----------------------------------------------------

        if result_json:
            return jsonify(result_json)

        # ----------------------------------------------------
        # Fallback to Last AI Message
        # ----------------------------------------------------

        final_message = response["messages"][-1].content

        clean_message = (
            final_message
            .strip()
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

        try:

            parsed_json = json.loads(clean_message)

            return jsonify(parsed_json)

        except json.JSONDecodeError:

            return jsonify({
                "raw_response": final_message
            })

    # ========================================================
    # Global Exception Handler
    # ========================================================

    except Exception as e:

        print(f"\n[ERROR] {str(e)}")

        return jsonify({
            "error": str(e)
        }), 500

# ============================================================
# Main Entry
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=False,
        host="0.0.0.0",
        port=5000
    )