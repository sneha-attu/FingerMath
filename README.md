FingerMath: AI-Powered Hand Gesture Calculator
FingerMath is a web-based, real-time calculator that uses hand gestures captured from a webcam to perform arithmetic operations and interact with the UI. Built with Python (Flask), OpenCV, and MediaPipe, it allows users to control and compute expressions using intuitive gestures—making calculation both interactive and accessible.

🚀 Features
Live Gesture Recognition – Detects and interprets hand gestures (0–5, +, -, *, /, =, C) in real time.

Manual Input Mode – Type calculations directly with keyboard or virtual calculator.

Responsive UI – Modern, mobile-friendly, and accessible interface.

History & Analytics – View calculation history and real-time analytics.

Speech Output – Reads results aloud for accessibility.

Dark Mode – Toggle between light and dark themes.

📸 Gesture Mappings
Gesture Pattern	Symbol	Meaning
Closed fist	0	Zero
Index finger	1	One
Index + Middle	2	Two
Index + Middle + Ring	3	Three
Four fingers (except thumb)	4	Four
All fingers open	5	Five
Thumb only	+	Addition
Thumb + Index	-	Subtraction
Thumb + Middle	*	Multiply
Index + Ring	/	Divide
Pinky only	=	Equals ('evaluate')
Thumb + Pinky	C	Clear
Tips: Hold a gesture steady for ~0.5–1 second for best recognition.

🖥️ Demo
![Demo Screenshot](: [insert a gif/screenshot of your tool in action here]*

🛠️ Tech Stack
Backend: Python, Flask, OpenCV, MediaPipe

Frontend: HTML5, CSS3, Bootstrap (custom), JavaScript (vanilla)

Other: Chart.js, FontAwesome, Google Fonts

📦 Installation
Dependencies
Python 3.8+

pip

1. Clone the Repository
bash
git clone https://github.com/your-username/hand-gesture-calculator.git
cd hand-gesture-calculator
2. Create Virtual Environment
bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
3. Install Requirements
bash
pip install -r requirements.txt
4. Download MediaPipe Models (auto when running the project)
5. Start the App
bash
python app.py
or with a production server:

bash
pip install waitress
python app.py
6. Open in Your Browser
Open http://localhost:5000

⚡ Usage
Click Start Camera to enable camera and gesture control.

Use the live view to show gestures; the digit/operator will reflect in the calculator display.

Use Manual Input tab to type or tap expressions using the virtual calculator.

Use Analytics to view your calculation stats.

🧩 Code Structure
text
hand-gesture-calculator/
│
├─ app.py                    # Flask app (main entry point)
├─ requirements.txt
├─ templates/
│   └─ index.html            # App HTML
├─ static/
│   ├─ style.css             # Custom CSS
│   └─ (icons/fonts/etc.)
├─ src/
│   └─ hand_calculator/
│       ├─ hands.py          # Hand tracking logic
│       ├─ evaluator.py      # Expression evaluation logic
│       ├─ gestures.py       # Gesture recognizer classes
│       ├─ main.py           # (Optional old CLI demo)
│       └─ ... (other modules)
└─ README.md                
