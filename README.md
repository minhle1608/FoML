##  Quick Start

Follow these steps to set up and run the application:

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train the Model
```bash
python -m src.train
```
*Note: Run this command from the project root directory*

### Step 3: Start the API Server
```bash
uvicorn app.API:app --reload
```
This will initialize the FastAPI backend server.

### Step 4: Launch the Web Interface
```bash
streamlit run app/interface.py
```

## 📋 Supported Car Brands

For accurate predictions, please use only the following car brands:

- Maruti
- Skoda
- Honda
- Hyundai
- Toyota
- Ford
- Renault
- Mahindra
- Tata
- Chevrolet
- Fiat
- Datsun
- Jeep
- Mercedes-Benz
- Mitsubishi
- Audi
- Volkswagen
- BMW
- Nissan
- Lexus
- Jaguar
- Land (Land Rover)
- MG
- Volvo
- Daewoo
- Kia
- Force
- Ambassador
- Ashok (Ashok Leyland)
- Isuzu
- Opel
- Peugeot
