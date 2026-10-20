/* global fetchAPI, submitAPI */   

import { useReducer } from 'react'; // Import useReducer
import { Routes, Route, Link, useNavigate } from 'react-router-dom';
import BookingPage from './pages/BookingPage'; 

import ConfirmedBooking from './pages/ConfirmedBooking';


// Image imports
import restaurantFood from './assets/restauranfood.jpg';
import greekSaladImg from './assets/greek salad.jpg';
import bruschettaImg from './assets/bruchetta.svg';
import lemonDessertImg from './assets/lemon dessert.jpg';
import basketIcon from './assets/basket .svg';



// 1. Initialize the available times
export const initializeTimes = () => {
    // Create a Date object for today
    const today = new Date();
    
    // The API might expect a specific format (e.g., YYYY-MM-DD). 
    // If it returns undefined or errors, try passing a specific date string.
    // For now, let's pass the date object.
    return fetchAPI(today);
};

// 2. Create the reducer to handle state changes
export const updateTimes = (state, action) => {
    switch (action.type) {
        case 'UPDATE_TIMES':
            const dateObject = new Date(action.payload);
            return fetchAPI(dateObject);
        default:
            return state;
    }
};

function Specials() {
    return (
        <section className="specials-section">
            <div className="specials-header">
                <h2>This week specials!</h2>
                <Link to="/order" className="reserve-btn" aria-label="On Click">Online Menu</Link>
            </div>

            <div className="specials-grid">
                
                <article className="special-card">
                    <img src={greekSaladImg} alt="Greek Salad" className="card-image" />
                    <div className="card-content">
                        <div className="card-header">
                            <h3>Greek salad</h3>
                            <span className="price">$12.99</span>
                        </div>
                        <p>The famous greek salad of crispy lettuce, peppers, olives and our Chicago style feta cheese...</p>
                        <Link to="/order" className="delivery-link">
                            Order a delivery <img src={basketIcon} alt="basket" />
                        </Link>
                    </div>
                </article>

                <article className="special-card">
                    <img src={bruschettaImg} alt="Bruschetta" className="card-image" />
                    <div className="card-content">
                        <div className="card-header">
                            <h3>Bruchetta</h3>
                            <span className="price">$5.99</span>
                        </div>
                        <p>Our Bruschetta is made from grilled bread that has been smeared with garlic and seasoned with salt and olive oil.</p>
                        <Link to="/order" className="delivery-link">
                            Order a delivery <img src={basketIcon} alt="basket" />
                        </Link>
                    </div>
                </article>

                <article className="special-card">
                    <img src={lemonDessertImg} alt="Lemon Dessert" className="card-image" />
                    <div className="card-content">
                        <div className="card-header">
                            <h3>Lemon Dessert</h3>
                            <span className="price">$5.00</span>
                        </div>
                        <p>This comes straight from grandma's recipe book, every last ingredient has been sourced and is as authentic as can be imagined.</p>
                        <Link to="/order" className="delivery-link">
                            Order a delivery <img src={basketIcon} alt="basket" />
                        </Link>
                    </div>
                </article>

            </div>
        </section>
    );
}

function HomePage() {
    return (
        <>
            <section id="main_section">
                <div className="hero-text">
                    <h1>Little Lemon</h1>
                    <h2>Chicago</h2>
                    <p>
                        Little Lemon is a family-owned Mediterranean restaurant in Chicago, serving authentic recipes passed down through generations. 
                        From crispy falafel to creamy hummus, every dish is made with the freshest ingredients, love, and respect for tradition. 
                        Join us for lunch, dinner, or a family gathering — and enjoy a warm, welcoming taste of the Mediterranean.
                    </p>
                    <Link to="/booking" className="reserve-btn" aria-label="On Click">Reserve a table</Link>
                </div>
                <div className="hero-image-container">
                    <img src={restaurantFood} alt="Little Lemon Restaurant" />
                </div>
            </section>

            <Specials />

            <section className="about-section">
                <div className="about-text">
                    <h2>About</h2>
                    <p>
                        Little Lemon is owned by two Italian brothers, Mario and Adrian, who moved to the United States to pursue their shared dream of owning a restaurant.
                        To craft the menu, Mario relies on family recipes and his experience as a chef in Italy.
                        Adrian does all the marketing for the restaurant and led the effort to expand the menu beyond classic Italian to incorporate additional cuisines from the Mediterranean region.
                    </p>
                </div>
                <div className="about-images">
                    {/* Add your images here when ready */}
                </div>
            </section>
        </>
    );
}

function Main() {
    const [availableTimes, dispatch] = useReducer(updateTimes, initializeTimes());

    const navigate = useNavigate();

    const submitForm = (formData) => {
        const isSuccessful = submitAPI(formData);
        
        if (isSuccessful) {
            navigate('/booking-confirmed');
        } else {
            alert('Something went wrong. Please try again.');
        }
    };

    return (
        <main>
            <Routes>
                <Route path="/" element={<HomePage />} />
                
                <Route 
                    path="/booking" 
                    element={
                        <BookingPage 
                            availableTimes={availableTimes} 
                            dispatch={dispatch} 
                            submitForm={submitForm} // 3. Передаем submitForm
                        /> 
                    } 
                />
                <Route path="/booking-confirmed" element={<ConfirmedBooking />} />
            </Routes>
        </main>
    );
}

export default Main;


