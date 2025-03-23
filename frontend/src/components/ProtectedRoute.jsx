// src/components/ProtectedRoute.jsx
import React from "react";
import { Navigate } from "react-router-dom";

// Simple check function (returns true if user is logged in)
const isAuthenticated = () => {
  const userLoggedIn = sessionStorage.getItem("isLoggedIn");
  return userLoggedIn === "true"; // Returns true if logged in
};

const ProtectedRoute = ({ children }) => {
  return isAuthenticated() ? children : <Navigate to="/login" />;
};

export default ProtectedRoute;
