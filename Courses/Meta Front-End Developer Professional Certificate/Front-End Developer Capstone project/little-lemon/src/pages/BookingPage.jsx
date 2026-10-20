import BookingForm from './BookingForm';

function BookingPage({ availableTimes, dispatch, submitForm }) {
    return (
        <div className="booking-container">
            <h1>Book a Table</h1>
            <p>Please fill in the form below to reserve your table.</p>
            
            <BookingForm 
                availableTimes={availableTimes} 
                dispatch={dispatch} 
                submitForm={submitForm} 
            />
        </div>
    );
}

export default BookingPage;