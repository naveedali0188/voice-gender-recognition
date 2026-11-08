# Gender Recognition from Voice

Classify a WAV voice recording as male or female using 13 mean MFCC features and a scikit-learn Random Forest model.

Live demo: https://voice.vorextdigitalai.com (available after deployment and DNS setup)

> **Model note:** If you train without audio in both dataset folders, the training script creates a synthetic placeholder so the app can start. Its predictions are not meaningful. Add a properly labeled dataset and retrain before presenting results as real classifications.

## Features

- WAV uploads up to 16 MB
- MFCC feature extraction with LibROSA at 16 kHz mono
- Random Forest classification with confidence scores
- Uploaded files are deleted after each request
- Responsive, accessible interface

## Tech stack

Flask, Flask-CORS, LibROSA, scikit-learn, Pandas, NumPy, vanilla HTML/CSS/JavaScript, and Gunicorn.

## Local setup

Use Python 3.10.13. Put labeled recordings in `dataset/male/` and `dataset/female/`, then train the model:

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate
pip install -r requirements.txt
python train_model.py
python app.py
```

Open http://localhost:5000. For a useful model, include a representative, consented, labeled dataset in both folders before training. Dataset quality affects accuracy; confidence scores are model probabilities and are not calibrated.

## Deploy to Render

1. Push this project folder as the root of a GitHub repository.
2. In Render, choose **New → Web Service** and connect the repository.
3. Set **Root Directory** to `.` if this repository contains only the app.
4. Set **Build Command** to `pip install -r requirements.txt`.
5. Set **Start Command** to `gunicorn app:app`.
6. Choose Python **3.10.13** and the **Free** instance type.
7. Deploy, then test the Render `onrender.com` URL. The live URL will be assigned by Render.

## Custom domain

In the Render service, open **Settings → Custom Domains** and add `voice.vorextdigitalai.com`. Copy the DNS target Render displays. In Cloudflare or GoDaddy DNS, create a CNAME with name `voice` and that target; if using Cloudflare, set proxy status to **DNS only** while Render verifies the domain. Render provisions SSL after verification.

## Main site integration

Add the navigation link, CTA card, and SEO tags from the integration snippet supplied with the project instructions to the Vorext Digital AI main site. Link the tool to https://voice.vorextdigitalai.com once the custom domain is live.

## Author

Vorext Digital AI
