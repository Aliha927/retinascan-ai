# RetinaScan AI

Explainable Diabetic Retinopathy screening with generative AI clinical reporting.

Upload a retinal fundus image → get a severity classification, a Grad-CAM heatmap
showing exactly what the model looked at, and generative AI reports (clinician +
patient English + patient Urdu) — all in one clinical dashboard.

## Project Structure

```
retinascan-ai/
├── backend/
│   ├── main.py              # FastAPI app: /api/screen, /api/history
│   ├── model.py              # ResNet-50 loading + inference
│   ├── utils.py               # CLAHE preprocessing + Grad-CAM
│   ├── train.py                # Fine-tuning script
│   ├── sort_dataset.py          # CSV dataset -> folder-per-class converter
│   ├── requirements.txt
│   └── .env.example
└── frontend/
    ├── src/
    │   ├── App.jsx            # Main dashboard UI
    │   ├── main.jsx
    │   └── index.css
    ├── index.html
    ├── package.json
    ├── vite.config.js
    ├── tailwind.config.js
    └── postcss.config.js
```

## Setup

### 1. Supabase setup (database + authentication)

1. supabase.com pe free account banao, naya project create karo
2. Project dashboard mein: Settings -> API -> yahan se **Project URL** aur **anon public key** copy karo
3. Dashboard mein SQL Editor kholo, `backend/supabase_schema.sql` ka pura content paste karke Run karo — yeh `scans` table bana dega
4. Authentication -> Providers mein Email provider already enabled hota hai by default — kuch aur karne ki zaroorat nahi

### 2. Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env and paste your real GEMINI_API_KEY, SUPABASE_URL, and SUPABASE_KEY
# Never commit this file
```

### 2. Get a trained model checkpoint

The backend needs `checkpoints/dr_model.pth` to give real predictions. Two paths:

**A. Train it yourself (recommended for the hackathon):**
```bash
# If your dataset is APTOS-style (CSV + single image folder):
python sort_dataset.py --csv train.csv --images_dir train_images --output_dir dataset

# Then fine-tune:
python train.py --data_dir dataset --epochs 5 --batch_size 32
```
This saves `checkpoints/dr_model.pth` automatically, and prints real validation
accuracy + a confusion matrix — keep that output, you'll want it for your
documentation and pitch deck.

**B. Skip training for now:** the app will still run (useful for testing the
full pipeline plumbing end-to-end), but predictions will be meaningless until
a real checkpoint exists. `model.py` prints a loud warning in this case — don't
demo without training first.

### 3. Run the backend

```bash
python main.py
# API live at http://localhost:8000
```

### 4. Frontend

```bash
cd frontend
cp .env.example .env
# Edit .env: set VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY (same Supabase project as backend)

npm install
npm run dev
# App live at http://localhost:5173 (or whatever port Vite prints)
```

## Notes

- **Real authentication** via Supabase Auth — email/password signup and login, session
  persists across page refreshes. New signups need to confirm via email before logging
  in (Supabase's default behavior).
- **Scan history is per-user and cloud-persistent** — stored in Supabase Postgres, tied
  to the logged-in user's email, survives backend restarts/redeploys (unlike the old
  SQLite version).
- **Safety guardrail:** predictions under 75% confidence trigger a warning badge
  recommending manual ophthalmologist review.
- **Generative reports** never state a confirmed diagnosis — they're explicitly
  framed as AI-assisted screening requiring clinician confirmation.
