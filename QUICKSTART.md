# Quick Start Guide

Get the Medical Impact Predictor running in 5 minutes!

## Prerequisites

- Python 3.8+
- Node.js 14+
- 4GB RAM

## Installation

### 1. Backend Setup (2 minutes)

```bash
# Install Python dependencies
pip install -r requirements.txt

# Start Flask server
python app.py
```

Server will run at `http://127.0.0.1:5000`

### 2. Frontend Setup (3 minutes)

```bash
# Navigate to frontend
cd frontend

# Install dependencies (first time only)
npm install

# Start development server
npm start
```

UI will open at `http://localhost:3000`

## Test the Application

1. Open `http://localhost:3000` in your browser
2. The form is pre-filled with example patient data
3. Click **"Generate Prediction"**
4. View results in the right panel:
   - Length of Stay (P10, P50, P90)
   - Estimated Costs
   - Risk Classification

## Example API Call

```bash
curl -X POST http://127.0.0.1:5000/predict_impact \
  -H "Content-Type: application/json" \
  -d '{
    "anchor_age": 70,
    "gender": "M",
    "admission_type": "EMERGENCY",
    "insurance": "Medicare",
    "primary_diagnosis": "Sepsis",
    "procedure_count": 2,
    "max_creatinine": 1.8,
    "min_hemoglobin": 9.5,
    "saps_ii_score": 35
  }'
```

## Configuration (Optional)

Edit `config/hospital_assumptions.py` to customize:
- `AVERAGE_DAILY_PATIENT_COST` (default: 2500)
- `SAPS_II_RISK_THRESHOLD` (default: 40)
- `AVERAGE_DAILY_PATIENT_COST_CURRENCY` (default: USD)

## Troubleshooting

### Backend Issues

**Error: "Model not found"**
- Ensure `models/advanced/*.joblib` files exist
- Re-run training: `python scripts/train_advanced_models.py`

**Error: "Port 5000 already in use"**
- Change port in `app.py`: `app.run(port=5001)`

### Frontend Issues

**Error: "Cannot connect to API"**
- Verify backend is running on port 5000
- Check CORS settings in `app.py`

**Error: "npm install fails"**
- Clear cache: `npm cache clean --force`
- Delete `node_modules` and retry

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore [API Documentation](README.md#-api-documentation)
- Check out [Contributing Guidelines](CONTRIBUTING.md)

## Need Help?

- Open an issue on GitHub
- Check existing issues for solutions
- Contact: shiva.chandra@example.com

Happy predicting! 🏥
