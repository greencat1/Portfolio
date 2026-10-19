import logo from './assets/Logo.svg';


function Footer() {
    return (
        <footer className="footer">
            <div className="footer-container">
                {/* Column 1: Logo */}
                <div className="footer-col">
                    <img src={logo} alt="Little Lemon Logo" className="footer-logo" />
                </div>

                {/* Column 2: Navigation */}
                <div className="footer-col">
                    <h3>Doormat Navigation</h3>
                    <ul>
                        <li><a href="/">Home</a></li>
                        <li><a href="/about">About</a></li>
                        <li><a href="/menu">Menu</a></li>
                    </ul>
                </div>

                {/* Column 3: Contact */}
                <div className="footer-col">
                    <h3>Contact</h3>
                    <ul>
                        <li>Address: 123 Lemon St</li>
                        <li>Phone: +1 (234) 567-890</li>
                        <li>Email: info@littlelemon.com</li>
                    </ul>
                </div>

                {/* Column 4: Social Media */}
                <div className="footer-col">
                    <h3>Social Media Links</h3>
                    <ul>
                        <li><a href="https://facebook.com">Facebook</a></li>
                        <li><a href="https://instagram.com">Instagram</a></li>
                        <li><a href="https://twitter.com">Twitter</a></li>
                    </ul>
                </div>
            </div>
        </footer>
    );
}

export default Footer;