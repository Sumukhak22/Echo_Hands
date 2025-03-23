import React, { useState } from "react";
import { FaUser } from "react-icons/fa";
import { MdEmail } from "react-icons/md";
import { RiLockPasswordFill } from "react-icons/ri";
import "./LoginSignup.css";

const LoginSignup = () => {
  const [isLogin, setIsLogin] = useState(false);
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const handleAuth = async () => {
    if (!username || !password || (!isLogin && !email)) {
      alert("Please fill all fields");
      return;
    }

    const endpoint = isLogin ? "http://localhost:5000/login" : "http://localhost:5000/signup";
    const body = isLogin ? { username, password } : { username, email, password };

    try {
      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      const data = await response.json();
      alert(data.message || data.error);
    } catch (error) {
      console.error("Error during authentication:", error);
      alert("Something went wrong. Please try again later.");
    }
  };

  return (
    <div className="containers">
      <div className={`box ${isLogin ? "login-mode" : ""}`}>
        <div className="form-container">
          <h2>{isLogin ? "Login" : "Sign Up"}</h2>
          <div className="input-box">
            <FaUser className="icon" />
            <input 
              type="text" 
              placeholder="Username" 
              value={username} 
              onChange={(e) => setUsername(e.target.value)}
              required 
            />
          </div>
          {!isLogin && (
            <div className="input-box">
              <MdEmail className="icon" />
              <input 
                type="email" 
                placeholder="Email" 
                value={email} 
                onChange={(e) => setEmail(e.target.value)}
                required 
              />
            </div>
          )}
          <div className="input-box">
            <RiLockPasswordFill className="icon" />
            <input 
              type="password" 
              placeholder="Password" 
              value={password} 
              onChange={(e) => setPassword(e.target.value)}
              required 
            />
          </div>
          <button className="btn" onClick={handleAuth}>{isLogin ? "Login" : "Sign Up"}</button>
          {isLogin && (
            <div className="forgot-password">
              Lost Password? <span>Click Here!</span>
            </div>
          )}
          <p className="toggle-text">
            {isLogin ? "Don't have an account?" : "Already have an account?"}
            <span onClick={() => setIsLogin(!isLogin)}>
              {isLogin ? " Sign Up" : " Login"}
            </span>
          </p>
        </div>
      </div>
    </div>
  );
};

export default LoginSignup;
