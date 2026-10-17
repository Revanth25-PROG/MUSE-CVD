# MUSE-CVD: Full Cloud Deployment Guide (Vercel + Render)

This guide provides step-by-step instructions to deploy:
* **Backend (FastAPI + Machine Learning Models)** on **Render**
* **Frontend (React + Vite + Tailwind CSS)** on **Vercel**

---

## Step 1: Initialize Git and Push to GitHub

From your local machine in PowerShell:

```powershell
# 1. Navigate to the project root
cd C:\Users\24761\.gemini\antigravity\scratch\MUSE-CVD

# 2. Initialize git repository (if not already initialized)
git init

# 3. Add all files
git add .

# 4. Commit changes
git commit -m "Complete MUSE-CVD research framework with models and UI"

# 5. Create a new repository on GitHub (e.g. named 'MUSE-CVD')
# Then link and push your code:
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/MUSE-CVD.git
git push -u origin main
```

---

## Step 2: Deploy the Backend on Render

1. Log in to your [Render Dashboard](https://dashboard.render.com/).
2. Click **New +** and select **Web Service**.
3. Choose **Build and deploy from a Git repository** and connect your `MUSE-CVD` GitHub repository.
4. Fill in the deployment configuration:
   * **Name**: `muse-cvd-backend` (or your choice)
   * **Region**: Choose the closest region (e.g., *Oregon (US West)* or *Frankfurt*)
   * **Branch**: `main`
   * **Root Directory**: `backend`
   * **Runtime**: `Python 3`
   * **Build Command**: `pip install -r requirements.txt`
   * **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   * **Instance Type**: `Free`
5. Click **Create Web Service**.
6. Wait 2–3 minutes for the build to finish. Once live, Render will give you a public URL like:
   `https://muse-cvd-backend.onrender.com`
7. Test the health endpoint in your browser:
   `https://muse-cvd-backend.onrender.com/api/health`  
   *(Expected response: `{"status":"healthy","model_loaded":true,"calibrated_model_loaded":true}`)*

> [!NOTE]
> Render free-tier web services spin down after 15 minutes of inactivity. When a request arrives, it spins up automatically within 20–30 seconds. The frontend has been configured with an extended timeout to seamlessly accommodate this.

---

## Step 3: Deploy the Frontend on Vercel

1. Log in to your [Vercel Dashboard](https://vercel.com/dashboard).
2. Click **Add New…** > **Project**.
3. Under **Import Git Repository**, select your `MUSE-CVD` repository.
4. Configure the project settings:
   * **Project Name**: `muse-cvd` (or your choice)
   * **Framework Preset**: `Vite`
   * **Root Directory**: Click **Edit** and select the `frontend` folder.
   * **Build Command**: `npm run build`
   * **Output Directory**: `dist`
5. **Environment Variables**:
   Expand the **Environment Variables** section and add:
   * **Key**: `VITE_API_URL`
   * **Value**: Your Render backend URL (e.g., `https://muse-cvd-backend.onrender.com`)
6. Click **Deploy**.
7. Vercel will build the frontend in ~30 seconds and provide your live public URL (e.g. `https://muse-cvd.vercel.app`).

---

## Step 4: Verification and Live Inspection

1. Open your new Vercel website link in any browser.
2. Observe the top navigation bar: the badge will show **API Active**.
3. Click on **Prediction Workspace**:
   * Select any clinical preset (e.g., *Suspected Acute MI Profile*).
   * Click **Run Model Prediction & Explain**.
   * Verify the real-time prediction, probability breakdown, and local SHAP feature attributions.
4. Browse the **Model Evaluation**, **Dataset Explorer**, **Explainable AI**, and **Research Manuscript** pages to inspect all interactive metrics.

---

## Troubleshooting & FAQ

* **Q: The frontend says "API Offline" or loading takes 20 seconds on first request.**  
  * **Answer:** This is standard for Render's free tier when spinning up from sleep mode. Wait 20 seconds and click again; it will stay warm during your active session.
* **Q: I get a CORS error in the browser console.**  
  * **Answer:** The FastAPI backend is configured with open CORS (`allow_origins=["*"]`), which permits all Vercel domains out of the box.
* **Q: How do I update my deployed application?**  
  * **Answer:** Simply run `git add .`, `git commit -m "Update"`, and `git push origin main`. Both Render and Vercel automatically trigger continuous deployment on git push!
