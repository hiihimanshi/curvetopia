
// import Image from './components/Imageuploader';

import Main from './components/Main';
import Video from './pages/video';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';

import './App.css';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Main />} />
        <Route path="/video" element={<Video />} />
      </Routes>
    </Router>
  );
}

export default App;