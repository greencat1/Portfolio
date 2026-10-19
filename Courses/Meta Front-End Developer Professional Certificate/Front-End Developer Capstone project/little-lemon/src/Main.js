import restaurantFood from './assets/restauranfood.jpg';
import greekSaladImg from './assets/greek salad.jpg';
import bruschettaImg from './assets/bruchetta.svg';
import lemonDessertImg from './assets/lemon dessert.jpg';
import basketIcon from './assets/basket .svg';

function Specials() {
    return (
        <section className="specials-section">
            <div className="specials-header">
                <h2>This week specials!</h2>
                <button className="reserve-btn">Online Menu</button>
            </div>

            <div className="specials-grid">
                
                <article className="special-card">
                    <img src={greekSaladImg} alt="Greek Salad" className="card-image" />
                    <div className="card-content">
                        <div className="card-header">
                            <h3>Greek salad</h3>
                            <span className="price">$12.99</span>
                        </div>
                        <p>
                            The famous greek salad of crispy lettuce, peppers, olives and our Chicago style feta cheese, garnished with crunchy garlic and rosemary croutons.
                        </p>
                        <a href="/order" className="delivery-link">
                            Order a delivery <img src={basketIcon} alt="basket" />
                        </a>
                    </div>
                </article>

                <article className="special-card">
                    <img src={bruschettaImg} alt="Bruschetta" className="card-image" />
                    <div className="card-content">
                        <div className="card-header">
                            <h3>Bruchetta</h3>
                            <span className="price">$5.99</span>
                        </div>
                        <p>
                            Our Bruschetta is made from grilled bread that has been smeared with garlic and seasoned with salt and olive oil.
                        </p>
                        <a href="/order" className="delivery-link">
                            Order a delivery <img src={basketIcon} alt="basket" />
                        </a>
                    </div>
                </article>

                <article className="special-card">
                    <img src={lemonDessertImg} alt="Lemon Dessert" className="card-image" />
                    <div className="card-content">
                        <div className="card-header">
                            <h3>Lemon Dessert</h3>
                            <span className="price">$5.00</span>
                        </div>
                        <p>
                            This comes straight from grandma's recipe book, every last ingredient has been sourced and is as authentic as can be imagined.
                        </p>
                        <a href="/order" className="delivery-link">
                            Order a delivery <img src={basketIcon} alt="basket" />
                        </a>
                    </div>
                </article>

            </div>
        </section>
    );
}

function Main() {
    return (
        <main>
            <section id='main_section'>
                <div className="hero-text">
                <h1>Little Lemon</h1>
                <p>Chicago</p>
                <p>Little Lemon is a family-owned Mediterranean restaurant in Chicago, serving authentic recipes passed down through generations. 
                    From crispy falafel to creamy hummus, every dish is made with the freshest ingredients, love, and respect for tradition. 
                    Join us for lunch, dinner, or a family gathering — and enjoy a warm, welcoming taste of the Mediterranean.
                    </p>
                <button className="reserve-btn">Reserve a table</button>
                </div>
                <div className="hero-image-container">
                <img src={restaurantFood}/>
                </div>
                
            </section>

            <section>
                <div id='head_and_button'>
                <h2>This week specials!</h2>
                <button className="Menu">Online Menu</button>
                </div>
                <Specials></Specials>
            </section>

            <section>
                <h2>About</h2>
                {/* discription */}
            </section>
        </main>
    );
}

export default Main;