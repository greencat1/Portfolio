# 🍋 Little Lemon — Table Booking SPA

A responsive, accessible **React SPA** for a Mediterranean restaurant. Built as the final Capstone Project for the **Meta Front-End Developer Certificate**.

## 🚀 Features
- **Table Booking:** Date, time, guests, occasion — with live API fetching.
- **State Management:** Shared state via `useReducer` + lifted props.
- **Validation:** HTML5 + React (submit button disabled until valid).
- **Accessibility:** Semantic HTML, ARIA labels, keyboard-friendly.
- **Routing:** React Router v6 with a confirmation page.
- **Testing:** Jest & React Testing Library (100% pass).

## 🛠️ Stack
`React 18` · `React Router v6` · `CSS3 (Flex/Grid)` · `Jest`

## ⚡ Quick Start
```bash
git clone https://github.com/YOUR_USERNAME/little-lemon.git
cd little-lemon
npm install
npm start   # → http://localhost:3000
npm test    # → run unit tests
```

## 📂 Structure

```text
src/
├── pages/          # BookingPage, BookingForm, ConfirmedBooking
├── assets/         # Images & SVGs
├── Main.js         # Router + useReducer logic
└── index.js        # Entry point
public/
└── api.js          # Mock fetchAPI / submitAPI
```

## 🧠 How It Works

1. User selects a date → dispatch calls fetchAPI(date).
2. useReducer updates available times.
3. Form validates → user submits.
4. submitAPI() → navigates to /booking-confirmed.

