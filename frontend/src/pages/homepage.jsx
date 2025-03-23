import React, { useState, useEffect } from 'react';
import './homepage.css';
import { useNavigate } from "react-router-dom";
import { FaRegWindowClose } from "react-icons/fa";
// Mock data - would be fetched from your MongoDB in production
const mockUserData = {
  name: "Alex",
  xp: 1250,
  level: 3,
  completedSections: ["Basic Greetings", "Numbers 1-10"],
  inProgressSections: ["Common Phrases"],
  accuracy: 87,
  lastSession: "2025-03-19T15:30:00Z"
};

const Avatar = ({ gender }) => {
  const avatarSrc = gender === 'male'
    ? 'https://api.dicebear.com/7.x/adventurer/svg?seed=male$'
    : 'https://api.dicebear.com/7.x/adventurer/svg?seed=female';

  return (
    <img
      src={avatarSrc}
      alt="avatar"
      className="w-40 h-40 rounded-full border-4 border-gray-300"
    />
  );
};

const learningPaths = [
  {
    id: "basic-greetings",
    title: "Basic Greetings",
    description: "Learn essential greetings like 'Hello', 'Goodbye', 'Thank you', and more.",
    xpRequired: 100,
    completed: true,
    progress: 100,
    icon: "👋"
  },
  {
    id: "numbers",
    title: "Numbers 1-10",
    description: "Master counting from 1 to 10 in sign language.",
    xpRequired: 150,
    completed: true,
    progress: 100,
    icon: "🔢"
  },
  {
    id: "common-phrases",
    title: "Common Phrases",
    description: "Learn everyday phrases for basic communication.",
    xpRequired: 200,
    completed: false,
    progress: 65,
    icon: "💬"
  },
  {
    id: "family-members",
    title: "Family Members",
    description: "Signs for family relationships like mother, father, sister, etc.",
    xpRequired: 250,
    completed: false,
    progress: 20,
    icon: "👪"
  },
  {
    id: "intermediate",
    title: "Intermediate Level",
    description: "Advanced concepts for more fluid conversations.",
    xpRequired: 500,
    completed: false,
    progress: 0,
    icon: "🌟"
  }
];

const achievements = [
  { id: 1, title: "First Steps", description: "Complete your first lesson", unlocked: true },
  { id: 2, title: "Perfect Form", description: "Get 100% accuracy on a sign", unlocked: true },
  { id: 3, title: "Consistent Learner", description: "Practice for 5 days in a row", unlocked: false },
  { id: 4, title: "Vocabulary Master", description: "Learn 50 different signs", unlocked: false },
];

const HomePage = () => {
  const [userData, setUserData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [feedbackVisible, setFeedbackVisible] = useState(false);
  const [selectedPath, setSelectedPath] = useState(null);
  const [gender, setGender] = useState('male'); // Fixed gender state

  const navigate = useNavigate(); // Initialize useNavigate

  useEffect(() => {
    // Simulate API call to fetch user data
    setTimeout(() => {
      setUserData(mockUserData);
      setIsLoading(false);
    }, 800);
  }, []);

  const handleContinueLearning = () => {
    // Find the first in-progress section
    const inProgress = learningPaths.find(
      (path) => !path.completed && path.progress > 0
    );
    if (inProgress) {
      setSelectedPath(inProgress);
      setFeedbackVisible(true);
    }
    // Redirect to /video page
    navigate("/video"); // Redirects to /video
  };

  const handleStartPath = (path) => {
    setSelectedPath(path);
    setFeedbackVisible(true);
  };

  const handleCloseFeedback = () => {
    setFeedbackVisible(false);
    setSelectedPath(null);
  };

  const calculateLevelProgress = () => {
    const xpForCurrentLevel = 1000;
    const xpForNextLevel = 2000;
    const currentLevelXp = userData.xp - xpForCurrentLevel;
    const xpNeededForNextLevel = xpForNextLevel - xpForCurrentLevel;
    return (currentLevelXp / xpNeededForNextLevel) * 100;
  };

  if (isLoading) {
    return (
      <div className="loading-container">
        <div className="spinner"></div>
        <p>Loading your sign language journey...</p>
      </div>
    );
  }

  return (
    <div className="homepage-container">
      <header className="header">
        <div className="logo">
          <h1>
            Echo<span className="highlight">Hands</span>
          </h1>
        </div>
        <nav className="navigation">
          <ul>
            <li><a href="#learn">Learn</a></li>
            <li><a href="#progress">Progress</a></li>
            <li><a href="#community">Community</a></li>
            <li><a href="#about">About</a></li>
          </ul>
        </nav>
        <div className="user-profile">
          <div className="user-avatar">{userData.name.charAt(0)}</div>
          <div className="user-info">
            <p className="user-name">{userData.name}</p>
            <p className="user-level">Level {userData.level}</p>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="hero-section">
        <div className="hero-content">
          <h1>
            Learn Sign Language
            <br />
            Through Interactive Practice
          </h1>
          <p>Master ASL with our camera-based, gamified learning experience</p>
          <button className="primary-button" onClick={handleContinueLearning}>
            Continue Learning
          </button>
        </div>

        <div className="hero-image">
          <div className="sign-demo">
            <Avatar gender={gender} />
          </div>
        </div>

        {/* <h1 className="text-4xl font-bold mb-8">Avatar Generator</h1> */}
        {/* <div className="mt-6"> */}
          {/* <button
            onClick={() => setGender('male')}
            className={`px-6 py-2 mx-2 rounded-lg ${
              gender === 'male' ? 'bg-blue-500 text-white' : 'bg-gray-300'
            }`}
          >
            Male
          </button>
          <button
            onClick={() => setGender('female')}
            className={`px-6 py-2 mx-2 rounded-lg ${
              gender === 'female' ? 'bg-pink-500 text-white' : 'bg-gray-300'
            }`}
          >
            Female
          </button> */}
        {/* </div> */}
      </section>

      {/* Learning Paths Section */}
      <section className="learning-paths-section" id="learn">
        <h2>Learning Paths</h2>
        <div className="paths-container">
          {learningPaths.map((path) => (
            <div
              key={path.id}
              className={`path-card ${path.completed ? 'completed' : ''} ${
                userData.inProgressSections.includes(path.title) ? 'in-progress' : ''
              }`}
            >
              <div className="path-icon">{path.icon}</div>
              <h3>{path.title}</h3>
              <p>{path.description}</p>
              <div className="path-progress">
                <div className="progress-bar">
                  <div
                    className="progress-fill"
                    style={{ width: `${path.progress}%` }}
                  ></div>
                </div>
                <span>{path.progress}%</span>
              </div>
              <div className="path-xp">
                <span className="xp-icon">⭐</span>
                <span>{path.xpRequired} XP</span>
              </div>
              {path.completed ? (
                <button className="secondary-button">Review</button>
              ) : path.progress > 0 ? (
                <button
                  className="primary-button"
                  onClick={() => handleStartPath(path)}
                >
                  Continue
                </button>
              ) : (
                <button
                  className="primary-button"
                  onClick={() => handleStartPath(path)}
                >
                  Start
                </button>
              )}
            </div>
          ))}
        </div>
      </section>

      {/* Feedback Modal */}
      {feedbackVisible && selectedPath && (
        <div className="feedback-modal">
          <div className="feedback-content">
            <button className="close-button" onClick={handleCloseFeedback}>
              <FaRegWindowClose size={24} color="#000000" />
            </button>

            <h2>{selectedPath.title}</h2>
            <div className="feedback-details">
              <div className="feedback-score">
                <h3>Current Accuracy</h3>
                <div className="score-circle">
                  <span>{userData.accuracy}%</span>
                </div>
              </div>
              <div className="feedback-tips">
                <h3>Tips for Improvement</h3>
                <ul>
                  <li>Keep your wrist relaxed for smoother transitions</li>
                  <li>Practice in front of a mirror to compare with examples</li>
                  <li>Focus on finger positioning for clearer signs</li>
                </ul>
              </div>
            </div>
            <button className="primary-button" onClick={handleContinueLearning}>
              Start Lesson
            </button>
          </div>
        </div>
      )}

      {/* Footer */}
      <footer className="footer">
        <div className="footer-content">
          <div className="footer-section">
            <h3>
              SignWave<span className="highlight">Learn</span>
            </h3>
            <p>Making sign language accessible for everyone through interactive, camera-based learning.</p>
          </div>
          <div className="footer-section">
            <h3>Quick Links</h3>
            <ul>
              <li><a href="#learn">Learning Paths</a></li>
              <li><a href="#progress">Your Progress</a></li>
              <li><a href="#community">Community</a></li>
              <li><a href="#about">About Us</a></li>
            </ul>
          </div>
          <div className="footer-section">
            <h3>Contact</h3>
            <p>Email: help@signwavelearn.com</p>
            <p>Support:XXXXXXXXX</p>
          </div>
        </div>
        <div className="footer-bottom">
          <p>&copy; 2025 SignWaveLearn. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
};

export default HomePage;
