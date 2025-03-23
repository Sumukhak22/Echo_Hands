import React from "react";
import { Routes, Route } from 'react-router-dom';

import LoginSignup from "./pages/LoginSignup";
import HomePage from "./pages/homepage";

function App() {
  return (
    <Routes>

      <Route path="/login" element={<LoginSignup />} />
      <Route path="/" element={<HomePage />} />


    </Routes>
  );
}

export default App;
