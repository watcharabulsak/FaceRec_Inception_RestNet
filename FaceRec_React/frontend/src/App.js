import React from 'react';
import Video from './components/pages/dashboard';
import logo from './logo.png';
import './App.css';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} alt="logo"/>
        <Video />
      </header>
    </div>
  );
}

export default App;
