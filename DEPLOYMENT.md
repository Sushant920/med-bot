# 🚀 Streamlit Cloud Deployment Guide

This guide will help you deploy your **AI-CPA MedBot** application to Streamlit Cloud.

## 1. Prerequisites
- A **GitHub Account**.
- Your code pushed to a GitHub repository.
- A **Streamlit Cloud Account** (sign up at [share.streamlit.io](https://share.streamlit.io/)).

## 2. Configuration Steps

### Step 1: Push Code to GitHub
Ensure all your latest changes (including `requirements.txt`) are pushed to your GitHub repository.

### Step 2: Deploy on Streamlit Cloud
1.  Log in to [Streamlit Cloud](https://share.streamlit.io/).
2.  Click **"New app"**.
3.  Select your GitHub repository, branch (usually `main`), and for **"Main file path"**, enter:
    ```
    src/app.py
    ```
    *(Note: The default is usually `streamlit_app.py`, so you MUST change this)*.

### Step 3: Configure Secrets (CRITICAL)
Your app uses **Secrets** to securely manage the Database connection and Google API Key. You must set these up in the Streamlit Cloud dashboard.

1.  After clicking "Deploy" (or in the App Settings > Secrets), go to the **"Secrets"** section.
2.  Paste the following configuration into the Secrets text area.
    -   **Replace** the values with your actual cloud credentials.

    ```toml
    # --- Google Gemini API Key ---
    GOOGLE_API_KEY = "your-google-api-key-here"

    # --- Database Connection (PostgreSQL) ---
    [database]
    url = "postgresql://user:password@host:port/dbname"
    
    # Note: If you don't have a cloud DB yet, the app will 
    # look for a local SQLite file, which DOES NOT PERSIST on Cloud.
    # We highly recommend using a free PostgreSQL instance (e.g., Neon, Supabase).
    ```

### Step 4: Reboot
Once secrets are saved, Streamlit should automatically restart the app. If not, click **"Reboot"** in the "Manage app" menu.

## 3. Verify Deployment
-   **Login**: Try logging in (default admin: `admin`/`admin123` if you haven't changed it).
-   **Dashboard**: Ensure charts load (this confirms `faers_drug_summary.csv` is being found).
-   **Patient Entry**: Use the "Generate Clinical Analysis" feature to test the Gemini API connection.
-   **Explainability**: Check the "Explainability" tab to ensure SHAP values are calculated (confirms `feature_template.csv` is found).

## Troubleshooting
-   **"FileNotFoundError"**: Check the logs. The app now uses robust path finding to locate `colabupload` and `models` folders relative to `src/app.py`. Ensure these folders are in your git repo.
-   **"ProgrammingError (psycopg2)"**: We fixed the `check_same_thread` issue. If you see DB errors, double-check your `[database] url` format in secrets. It should start with `postgresql://`.
