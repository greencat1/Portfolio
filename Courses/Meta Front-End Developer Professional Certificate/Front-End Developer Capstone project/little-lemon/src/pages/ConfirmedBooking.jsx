// src/pages/ConfirmedBooking.jsx
import { Link } from 'react-router-dom';

function ConfirmedBooking() {
    return (
        <div style={{ textAlign: 'center', padding: '50px' }}>
            <h1>Booking Confirmed! 🎉</h1>
            <p>Thank you for choosing Little Lemon. We look forward to seeing you!</p>
            <Link to="/">Return to Home</Link>
        </div>
    );
}

export default ConfirmedBooking;