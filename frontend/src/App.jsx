import React from "react";
import { Routes, Route } from 'react-router-dom';

import LoginSignup from "./pages/LoginSignup";
import HomePage from "./pages/homepage";
import Video from "./pages/Learn";
function App() {
  return (
    <Routes>

      <Route path="/login" element={<LoginSignup />} />
      <Route path="/" element={<HomePage />} />
      <Route path="/video" element={<Video />} />

    </Routes>
  );
}

export default App;
