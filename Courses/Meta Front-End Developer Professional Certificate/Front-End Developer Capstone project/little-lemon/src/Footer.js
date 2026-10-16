import logo from './assets/Logo.svg';

function Footer() {
    return (
        <footer>
            <div>
                <img src={logo} alt="Little Lemon Logo" />
            </div>
            <div>
                <h3>Doormat Navigation</h3>
                <ul>
                    <li><a href="/">Home</a></li>
                    <li><a href="/about">About</a></li>
                    <li><a href="/menu">Menu</a></li>
                </ul>
            </div>
            <div>
                <h3>Contact</h3>
                <p>Address: 123 Lemon St</p>
                <p>Phone: +1 (234) 567-890</p>
                <p>Email: info@littlelemon.com</p>
            </div>
            <div>
                <h3>Social Media Links</h3>
                <ul>
                    <li><a href="https://facebook.com">Facebook</a></li>
                    <li><a href="https://instagram.com">Instagram</a></li>
                    <li><a href="https://twitter.com">Twitter</a></li>
                </ul>
            </div>
        </footer>
    );
}

export default Footer;