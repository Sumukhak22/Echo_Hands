import React from "react";
import { Routes, Route } from "react-router-dom";

import LoginSignup from "./pages/LoginSignup";
import HomePage from "./pages/homepage";
import IDk from "./pages/Learn";
import ProtectedRoute from "./components/ProtectedRoute";


function App() {
  return (
    <Routes>
      {/* Load login first */}
      <Route path="/login" element={<LoginSignup />} />

      {/* Protect HomePage and Video Route */}
      <Route
        path="/"
        element={
          <ProtectedRoute>
            <HomePage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/video"
        element={
          <ProtectedRoute>
            <IDk />
          </ProtectedRoute>
        }
      />
    </Routes>
  );
}

export default App;
