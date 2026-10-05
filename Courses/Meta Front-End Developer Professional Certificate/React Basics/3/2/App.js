// App.jsx
import { useState } from 'react';
import './App.css';

import LoginButton from './LoginButton';
import LogoutButton from './LogoutButton';
import HomePage from './HomePage';

function App() {
    const [loggedIn, setLoggedIn] = useState(false);

    const login = () => setLoggedIn(true);
    const logout = () => setLoggedIn(false);

    return (
        <>
            {loggedIn ? (
                <div>
                    <HomePage />
                    <LogoutButton logout={logout} />
                </div>
            ) : (
                <LoginButton login={login} />
            )}
        </>
    );
}

export default App;