import { useState } from 'react';
import './BookingForm.css';

function BookingForm({ availableTimes, dispatch, submitForm }) {
    const [date, setDate] = useState('');
    const [time, setTime] = useState('17:00');
    const [guests, setGuests] = useState(1);
    const [occasion, setOccasion] = useState('Birthday');

    const handleDateChange = (e) => {
        const selectedDate = e.target.value;
        setDate(selectedDate);
        dispatch({ type: 'UPDATE_TIMES', payload: selectedDate });
    };

    const handleSubmit = (e) => {
        e.preventDefault(); 
        
        const formData = {
            date,
            time,
            guests,
            occasion
        };

        submitForm(formData);
    };

    return (
        <form className="booking-form" onSubmit={handleSubmit}>
            <label htmlFor="res-date">Choose date</label>
            <input 
                type="date" 
                id="res-date" 
                value={date}
                onChange={handleDateChange} 
                required
            />

            <label htmlFor="res-time">Choose time</label>
            <select 
                id="res-time" 
                value={time}
                onChange={(e) => setTime(e.target.value)}
            >
                {availableTimes.map(t => (
                    <option key={t} value={t}>{t}</option>
                ))}
            </select>

            <label htmlFor="guests">Number of guests</label>
            <input 
                type="number" 
                id="guests" 
                min="1" 
                max="10"
                value={guests}
                onChange={(e) => setGuests(e.target.value)}
                required
            />

            <label htmlFor="occasion">Occasion</label>
            <select 
                id="occasion" 
                value={occasion}
                onChange={(e) => setOccasion(e.target.value)}
            >
                <option value="Birthday">Birthday</option>
                <option value="Anniversary">Anniversary</option>
            </select>

            <button type="submit">Make Your reservation</button>
        </form>
    );
}

export default BookingForm;